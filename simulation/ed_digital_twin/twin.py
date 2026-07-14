"""SimPy-based Active Digital Twin for Emergency Department."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import simpy
from scipy import stats

from ed_digital_twin.distributions import (
    CONFIG,
    inter_arrival_time,
    needs_admission,
    needs_imaging,
    sample_esi,
    sample_service_time,
    generate_reference_service_samples,
)
from ed_digital_twin.sync import BidirectionalSync


class PatientStage(Enum):
    WAITING = "waiting"
    TRIAGE = "triage"
    CONSULTATION = "consultation"
    IMAGING = "imaging"
    DISPOSITION = "disposition"
    DEPARTED = "departed"


@dataclass
class Patient:
    id: int
    esi: int
    arrival_time: float
    stage: PatientStage = PatientStage.WAITING
    needs_img: bool = False
    needs_adm: bool = False
    remaining_service: float = 0.0
    wait_start: float = 0.0
    los: float = 0.0
    vitals: Dict[str, float] = field(default_factory=dict)


@dataclass
class EDMetrics:
    los_values: List[float] = field(default_factory=list)
    wait_by_esi: Dict[int, List[float]] = field(default_factory=lambda: {i: [] for i in range(1, 6)})
    occupancy_trace: List[Tuple[float, int]] = field(default_factory=list)
    max_occupancy: int = 0
    throughput: int = 0
    hourly_counts: List[int] = field(default_factory=list)
    critical_waits: List[float] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        los_arr = np.array(self.los_values) if self.los_values else np.array([0.0])
        crit_arr = np.array(self.critical_waits) if self.critical_waits else np.array([0.0])
        equity = self._equity_index()
        return {
            "mean_los": float(np.mean(los_arr)),
            "median_los": float(np.median(los_arr)),
            "std_los": float(np.std(los_arr)),
            "max_occupancy": float(self.max_occupancy),
            "throughput": float(self.throughput),
            "mean_critical_wait": float(np.mean(crit_arr)),
            "equity_index": equity,
            "p95_los": float(np.percentile(los_arr, 95)),
        }

    def _equity_index(self) -> float:
        """Higher is better: inverse of wait-time Gini across ESI groups."""
        mean_waits = []
        for esi in range(1, 6):
            waits = self.wait_by_esi.get(esi, [])
            if waits:
                mean_waits.append(np.mean(waits))
        if len(mean_waits) < 2:
            return 1.0
        arr = np.array(mean_waits)
        n = len(arr)
        sorted_arr = np.sort(arr)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_arr)) / (n * np.sum(sorted_arr)) - (n + 1) / n
        return float(max(0.0, 1.0 - gini))


class EmergencyDepartmentTwin:
    """Active Digital Twin with bidirectional sync and policy-driven dispatch."""

    def __init__(
        self,
        seed: int = 42,
        policy: Optional[Callable] = None,
        demand_multiplier: float = 1.0,
        resource_multiplier: float = 1.0,
        enable_sync: bool = True,
    ):
        self.seed = seed
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.policy = policy or fifo_policy
        self.demand_multiplier = demand_multiplier
        self.resource_multiplier = resource_multiplier
        self.enable_sync = enable_sync

        cap = CONFIG["capacity"]
        self.n_beds = max(1, int(cap["beds"] * resource_multiplier))
        self.n_consult = max(1, int(cap["consult_rooms"] * resource_multiplier))
        self.n_imaging = max(1, int(cap["imaging_rooms"] * resource_multiplier))
        self.n_triage = max(1, int(cap["triage_stations"] * resource_multiplier))

        self.duration = CONFIG["simulation"]["duration_hours"] * 60
        self.warmup = CONFIG["simulation"]["warmup_hours"] * 60

        self.env: Optional[simpy.Environment] = None
        self.patients: List[Patient] = []
        self.waiting_queue: List[Patient] = []
        self.metrics = EDMetrics()
        self.sync = BidirectionalSync()
        self.patient_counter = 0
        self.current_hour_counts: Dict[int, int] = {}

    def _generate_vitals(self, esi: int) -> Dict[str, float]:
        base_hr = {1: 130, 2: 110, 3: 95, 4: 85, 5: 78}[esi]
        base_spo2 = {1: 88, 2: 93, 3: 96, 4: 98, 5: 99}[esi]
        return {
            "heart_rate": base_hr + self.rng.normal(0, 5),
            "spo2": min(100, base_spo2 + self.rng.normal(0, 2)),
            "sbp": 120 - (esi - 3) * 10 + self.rng.normal(0, 8),
        }

    def _select_next_patient(self) -> Optional[Patient]:
        if not self.waiting_queue:
            return None
        return self.policy(self.waiting_queue, self.env.now)

    def _get_ed_state(self) -> Dict[str, Any]:
        queue_by_esi = {i: 0 for i in range(1, 6)}
        wait_by_esi = {i: 0.0 for i in range(1, 6)}
        for p in self.waiting_queue:
            queue_by_esi[p.esi] += 1
            wait_by_esi[p.esi] = max(wait_by_esi[p.esi], self.env.now - p.wait_start)
        active = sum(1 for p in self.patients if p.stage != PatientStage.DEPARTED)
        return {
            "queue_by_esi": queue_by_esi,
            "wait_times_by_esi": wait_by_esi,
            "beds_occupied": min(active, self.n_beds),
            "consult_occupied": sum(1 for p in self.patients if p.stage == PatientStage.CONSULTATION),
            "imaging_occupied": sum(1 for p in self.patients if p.stage == PatientStage.IMAGING),
            "patients_in_system": active,
            "hourly_arrivals": list(self.current_hour_counts.values()),
        }

    def _sync_cycle(self) -> None:
        state = self._get_ed_state()
        self.sync.pull_from_physical(state, self.env.now)
        n_scenarios = CONFIG["simulation"]["scenarios_per_minute"] // (
            60 // int(self.sync.sync_interval)
        )
        scenario_results = self.sync.run_stochastic_scenarios(
            max(10, n_scenarios), state, self.rng
        )
        self.sync.push_to_virtual(scenario_results)
        self.sync.log_sync()

    def _patient_process(self, patient: Patient):
        patient.wait_start = self.env.now
        self.waiting_queue.append(patient)

        while patient.stage != PatientStage.DEPARTED:
            selected = self._select_next_patient()
            if selected is None or selected.id != patient.id:
                yield self.env.timeout(0.5)
                continue

            self.waiting_queue.remove(patient)
            wait_time = self.env.now - patient.wait_start
            self.metrics.wait_by_esi[patient.esi].append(wait_time)
            if patient.esi <= 2:
                self.metrics.critical_waits.append(wait_time)

            if patient.stage == PatientStage.WAITING:
                patient.stage = PatientStage.TRIAGE
                svc = sample_service_time("triage", patient.esi, self.rng)
                yield self.env.timeout(svc)
                patient.stage = PatientStage.CONSULTATION
                patient.wait_start = self.env.now
                self.waiting_queue.append(patient)

            elif patient.stage == PatientStage.CONSULTATION:
                svc = sample_service_time("consultation", patient.esi, self.rng)
                patient.remaining_service = svc
                yield self.env.timeout(svc)

                if patient.needs_img:
                    patient.stage = PatientStage.IMAGING
                    patient.wait_start = self.env.now
                    self.waiting_queue.append(patient)
                else:
                    patient.stage = PatientStage.DISPOSITION
                    patient.wait_start = self.env.now
                    self.waiting_queue.append(patient)

            elif patient.stage == PatientStage.IMAGING:
                svc = sample_service_time("imaging", patient.esi, self.rng)
                yield self.env.timeout(svc)
                patient.stage = PatientStage.DISPOSITION
                patient.wait_start = self.env.now
                self.waiting_queue.append(patient)

            elif patient.stage == PatientStage.DISPOSITION:
                svc = sample_service_time("disposition", patient.esi, self.rng)
                yield self.env.timeout(svc)
                patient.stage = PatientStage.DEPARTED
                patient.los = self.env.now - patient.arrival_time
                if self.env.now > self.warmup:
                    self.metrics.los_values.append(patient.los)
                    self.metrics.throughput += 1

    def _arrival_process(self):
        while True:
            hour = (self.env.now / 60) % 24
            ia = inter_arrival_time(hour, self.rng)
            if self.demand_multiplier != 1.0:
                ia /= self.demand_multiplier
            yield self.env.timeout(ia)

            self.patient_counter += 1
            esi = sample_esi(self.rng)
            patient = Patient(
                id=self.patient_counter,
                esi=esi,
                arrival_time=self.env.now,
                needs_img=needs_imaging(esi, self.rng),
                needs_adm=needs_admission(esi, self.rng),
                vitals=self._generate_vitals(esi),
            )
            self.patients.append(patient)
            self.env.process(self._patient_process(patient))

            h = int(self.env.now // 60)
            self.current_hour_counts[h] = self.current_hour_counts.get(h, 0) + 1

            active = sum(1 for p in self.patients if p.stage != PatientStage.DEPARTED)
            self.metrics.max_occupancy = max(self.metrics.max_occupancy, active)
            if self.env.now > self.warmup:
                self.metrics.occupancy_trace.append((self.env.now, active))

    def _sync_process(self):
        interval = self.sync.sync_interval / 60.0  # convert to minutes
        while True:
            yield self.env.timeout(interval)
            self._sync_cycle()

    def _resource_triage(self):
        return simpy.Resource(self.env, capacity=self.n_triage)

    def run(self) -> EDMetrics:
        self.env = simpy.Environment()
        self.env.process(self._arrival_process())
        if self.enable_sync:
            self.env.process(self._sync_process())
        self.env.run(until=self.duration)
        return self.metrics

    @staticmethod
    def validate_fidelity(n_samples: int = 5000, seed: int = 42) -> Dict[str, Any]:
        """KS test on service times; MAPE on hourly arrival counts."""
        rng = np.random.Generator(np.random.PCG64(seed))
        ref = generate_reference_service_samples(n_samples, seed)
        sim_samples = {
            station: np.array(
                [sample_service_time(station, 3, rng) for _ in range(n_samples)]
            )
            for station in ref.keys()
        }
        ks_results = {}
        for station in ref:
            stat, pval = stats.ks_2samp(ref[station], sim_samples[station])
            ks_results[station] = {"statistic": float(stat), "p_value": float(pval)}

        twin = EmergencyDepartmentTwin(seed=seed)
        twin.run()
        observed = list(twin.current_hour_counts.values())
        expected_rate = CONFIG["arrival"]["base_rate_per_hour"]
        expected = [expected_rate] * len(observed) if observed else [expected_rate]
        if observed:
            mape = float(np.mean(np.abs(np.array(observed) - np.array(expected[: len(observed)])) / np.maximum(expected[: len(observed)], 1)) * 100)
        else:
            mape = 0.0

        return {"ks_tests": ks_results, "mape_hourly_arrivals_pct": mape}


# --- Dispatch policies ---

def fifo_policy(queue: List[Patient], now: float) -> Optional[Patient]:
    """First-In-First-Out baseline."""
    if not queue:
        return None
    return min(queue, key=lambda p: p.wait_start)


def esi_fifo_policy(queue: List[Patient], now: float) -> Optional[Patient]:
    """ESI-priority then FIFO (clinical standard)."""
    if not queue:
        return None
    min_esi = min(p.esi for p in queue)
    candidates = [p for p in queue if p.esi == min_esi]
    return min(candidates, key=lambda p: p.wait_start)


def srpt_policy(queue: List[Patient], now: float) -> Optional[Patient]:
    """Shortest Remaining Processing Time."""
    if not queue:
        return None
    return min(queue, key=lambda p: p.remaining_service if p.remaining_service > 0 else p.esi * 30)


def heuristic_policy(queue: List[Patient], now: float) -> Optional[Patient]:
    """ESI + predicted admission probability heuristic."""
    if not queue:
        return None

    def score(p: Patient) -> float:
        esi_weight = (6 - p.esi) * 10
        adm_weight = 5 if p.needs_adm else 0
        wait_weight = (now - p.wait_start) * 0.1
        return -(esi_weight + adm_weight + wait_weight)

    return max(queue, key=score)


POLICIES = {
    "esi_fifo": esi_fifo_policy,
    "fifo": fifo_policy,
    "srpt": srpt_policy,
    "heuristic": heuristic_policy,
}
