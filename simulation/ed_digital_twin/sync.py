"""Bidirectional synchronization layer for the active Digital Twin."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ed_digital_twin.distributions import CONFIG


@dataclass
class PhysicalState:
    """Snapshot of the physical ED plane."""

    timestamp: float = 0.0
    queue_by_esi: Dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(1, 6)})
    wait_times_by_esi: Dict[int, float] = field(default_factory=lambda: {i: 0.0 for i in range(1, 6)})
    beds_occupied: int = 0
    consult_occupied: int = 0
    imaging_occupied: int = 0
    patients_in_system: int = 0
    hourly_arrivals: List[int] = field(default_factory=list)


@dataclass
class VirtualState:
    """Snapshot of the digital twin plane."""

    timestamp: float = 0.0
    projected_occupancy: float = 0.0
    scenario_count: int = 0
    divergence: float = 0.0
    sync_latency_ms: float = 0.0


class BidirectionalSync:
    """
    Active bidirectional synchronizer between physical and virtual planes.
    Delta-t <= 30s per architecture specification.
    """

    def __init__(self, sync_interval: Optional[float] = None):
        self.sync_interval = sync_interval or CONFIG["simulation"]["sync_interval_seconds"]
        self.epsilon = 0.05  # divergence threshold
        self.physical = PhysicalState()
        self.virtual = VirtualState()
        self.sync_log: List[Dict[str, Any]] = []

    def pull_from_physical(self, ed_metrics: Dict[str, Any], sim_time: float) -> PhysicalState:
        """Ingest RTLS/EHR telemetry from physical plane."""
        self.physical = PhysicalState(
            timestamp=sim_time,
            queue_by_esi=ed_metrics.get("queue_by_esi", self.physical.queue_by_esi),
            wait_times_by_esi=ed_metrics.get("wait_times_by_esi", self.physical.wait_times_by_esi),
            beds_occupied=ed_metrics.get("beds_occupied", 0),
            consult_occupied=ed_metrics.get("consult_occupied", 0),
            imaging_occupied=ed_metrics.get("imaging_occupied", 0),
            patients_in_system=ed_metrics.get("patients_in_system", 0),
            hourly_arrivals=ed_metrics.get("hourly_arrivals", []),
        )
        return self.physical

    def push_to_virtual(self, scenario_results: Dict[str, Any]) -> VirtualState:
        """Update virtual plane with stochastic scenario projections."""
        self.virtual = VirtualState(
            timestamp=self.physical.timestamp,
            projected_occupancy=scenario_results.get("mean_occupancy", 0),
            scenario_count=scenario_results.get("n_scenarios", 0),
            divergence=self._compute_divergence(scenario_results),
            sync_latency_ms=scenario_results.get("latency_ms", 15.0),
        )
        return self.virtual

    def _compute_divergence(self, scenario_results: Dict[str, Any]) -> float:
        """Compute divergence between physical and projected occupancy."""
        phys_occ = self.physical.patients_in_system / max(
            CONFIG["capacity"]["beds"], 1
        )
        virt_occ = scenario_results.get("mean_occupancy", 0)
        return abs(phys_occ - virt_occ)

    def feedback_directive(self) -> Optional[Dict[str, Any]]:
        """Emit operational directive if divergence exceeds threshold."""
        if self.virtual.divergence > self.epsilon:
            return {
                "type": "recalibrate",
                "divergence": self.virtual.divergence,
                "timestamp": self.physical.timestamp,
            }
        return None

    def log_sync(self) -> None:
        self.sync_log.append({
            "timestamp": self.physical.timestamp,
            "physical_patients": self.physical.patients_in_system,
            "virtual_occupancy": self.virtual.projected_occupancy,
            "divergence": self.virtual.divergence,
            "latency_ms": self.virtual.sync_latency_ms,
        })

    def run_stochastic_scenarios(
        self,
        n_scenarios: int,
        current_state: Dict[str, Any],
        rng: np.random.Generator,
        horizon_minutes: float = 30.0,
        fast_cap: int = 15,
        use_simpy: bool = False,
    ) -> Dict[str, Any]:
        """Run N stochastic forward scenarios per sync cycle."""
        import time

        start = time.perf_counter()
        base_patients = current_state.get("patients_in_system", 0)
        beds = max(CONFIG["capacity"]["beds"], 1)
        n_run = min(n_scenarios, fast_cap)

        if not use_simpy:
            occupancies = []
            for _ in range(n_run):
                noise = rng.normal(0, 0.08)
                projected = base_patients * (1 + noise) + rng.poisson(1.5)
                occupancies.append(projected / beds)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return {
                "mean_occupancy": float(np.mean(occupancies)),
                "std_occupancy": float(np.std(occupancies)),
                "n_scenarios": n_run,
                "latency_ms": elapsed_ms,
            }

        from ed_digital_twin.twin import EmergencyDepartmentTwin, esi_fifo_policy

        occupancies: List[float] = []
        for _ in range(n_run):
            scenario_seed = int(rng.integers(0, 2**31 - 1))
            twin = EmergencyDepartmentTwin(
                seed=scenario_seed, policy=esi_fifo_policy, enable_sync=False
            )
            twin.duration = horizon_minutes
            twin.warmup = 0.0
            load_factor = base_patients / beds
            twin.demand_multiplier = 1.0 + load_factor * 0.3 + rng.normal(0, 0.05)
            metrics = twin.run()
            peak_occ = metrics.max_occupancy
            if metrics.occupancy_trace:
                peak_occ = max(o for _, o in metrics.occupancy_trace)
            occupancies.append(peak_occ / beds)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "mean_occupancy": float(np.mean(occupancies)),
            "std_occupancy": float(np.std(occupancies)),
            "n_scenarios": n_run,
            "latency_ms": elapsed_ms,
        }

    def run_lookahead(
        self,
        current_state: Dict[str, Any],
        rng: np.random.Generator,
        n_scenarios: int = 50,
        horizon_minutes: float = 120.0,
    ) -> Dict[str, Any]:
        """Full SimPy look-ahead for production State Manager."""
        return self.run_stochastic_scenarios(
            n_scenarios=n_scenarios,
            current_state=current_state,
            rng=rng,
            horizon_minutes=horizon_minutes,
            fast_cap=n_scenarios,
            use_simpy=True,
        )
