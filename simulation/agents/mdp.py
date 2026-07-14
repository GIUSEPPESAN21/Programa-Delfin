"""Markov Decision Process formulation for ED orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ed_digital_twin.distributions import CONFIG
from ed_digital_twin.twin import EmergencyDepartmentTwin, Patient, PatientStage


STATE_DIM = 17
ACTION_DIM = 6


def get_action_policies() -> List[Callable]:
    def esi_strict(queue, now):
        if not queue:
            return None
        return min(queue, key=lambda p: (p.esi, p.wait_start))

    def critical_boost(queue, now):
        if not queue:
            return None
        for p in queue:
            if p.esi <= 2 and (now - p.wait_start) > 15:
                return p
        return min(queue, key=lambda p: (p.esi, p.wait_start))

    def fast_track(queue, now):
        esi45 = [p for p in queue if p.esi >= 4 and p.stage == PatientStage.WAITING]
        if esi45:
            return min(esi45, key=lambda p: p.wait_start)
        return min(queue, key=lambda p: (p.esi, p.wait_start))

    def imaging_priority(queue, now):
        img = [p for p in queue if p.stage == PatientStage.IMAGING]
        if img:
            return min(img, key=lambda p: p.esi)
        return min(queue, key=lambda p: (p.esi, -p.wait_start))

    def admission_priority(queue, now):
        adm = [p for p in queue if p.needs_adm]
        if adm:
            return min(adm, key=lambda p: p.esi)
        return min(queue, key=lambda p: (p.esi, p.wait_start))

    def balanced(queue, now):
        def score(p):
            return (6 - p.esi) * 5 + (now - p.wait_start) * 0.05
        return max(queue, key=score)

    return [
        esi_strict,
        critical_boost,
        fast_track,
        imaging_priority,
        admission_priority,
        balanced,
    ]


@dataclass
class MDPConfig:
    gamma: float = 0.99
    alpha: float = 2.0
    beta: float = 0.5
    gamma_occ: float = 0.3
    delta: float = 0.2
    lambda_fair: float = 1.0


class EDOrchestrationEnv:
    """
    MDP environment: M = (S, A, P, R, gamma)
    Wraps Digital Twin for DRL training with shortened episodes.
    """

    def __init__(
        self,
        seed: int = 0,
        episode_hours: float = 8.0,
        mdp_config: Optional[MDPConfig] = None,
    ):
        self.seed = seed
        self.episode_hours = episode_hours
        self.mdp = mdp_config or MDPConfig(**{
            k: CONFIG["reward_weights"].get(
                "lambda_fair" if k == "lambda_fair" else k, v
            )
            for k, v in MDPConfig().__dict__.items()
            if k != "gamma"
        })
        rw = CONFIG["reward_weights"]
        self.mdp = MDPConfig(
            gamma=CONFIG["dqn"]["gamma"],
            alpha=rw["alpha"],
            beta=rw["beta"],
            gamma_occ=rw["gamma"],
            delta=rw["delta"],
            lambda_fair=rw["lambda_fair"],
        )
        self.policies = get_action_policies()
        self.twin: Optional[EmergencyDepartmentTwin] = None
        self.current_action = 0
        self.step_count = 0
        self.max_steps = 24

    def _encode_state(self, twin: EmergencyDepartmentTwin) -> np.ndarray:
        state = twin._get_ed_state()
        q = state["queue_by_esi"]
        w = state["wait_times_by_esi"]
        cap = CONFIG["capacity"]
        hour = (twin.env.now / 60) % 24 if twin.env else 0

        vec = [
            q.get(i, 0) / 10.0 for i in range(1, 6)
        ] + [
            min(w.get(i, 0) / 120.0, 1.0) for i in range(1, 6)
        ] + [
            state["beds_occupied"] / cap["beds"],
            state["consult_occupied"] / cap["consult_rooms"],
            state["imaging_occupied"] / cap["imaging_rooms"],
            hour / 24.0,
            state["patients_in_system"] / cap["beds"],
            np.mean([p.vitals.get("spo2", 97) for p in twin.patients[-20:]]) / 100.0 if twin.patients else 0.97,
            np.mean([p.vitals.get("heart_rate", 80) for p in twin.patients[-20:]]) / 200.0 if twin.patients else 0.4,
        ]
        return np.array(vec[:STATE_DIM], dtype=np.float32)

    def _compute_reward(self, metrics, twin: EmergencyDepartmentTwin) -> float:
        rw = self.mdp
        w_crit = np.mean(metrics.critical_waits) if metrics.critical_waits else 0
        los = np.mean(metrics.los_values[-10:]) if metrics.los_values else 0
        occ = metrics.max_occupancy / CONFIG["capacity"]["beds"]
        thru = metrics.throughput / max(twin.env.now / 60, 1) if twin.env else 0
        unfair = 1.0 - metrics._equity_index() if hasattr(metrics, "_equity_index") else 0

        r = (
            -rw.alpha * (w_crit / 60.0)
            - rw.beta * (los / 120.0)
            - rw.gamma_occ * occ
            + rw.delta * thru
            - rw.lambda_fair * unfair
        )
        return float(r)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.seed = seed
        self.current_action = 0
        self.step_count = 0
        policy = self.policies[self.current_action]

        self.twin = EmergencyDepartmentTwin(seed=self.seed, policy=policy)
        self.twin.duration = self.episode_hours * 60
        self.twin.warmup = min(60, self.twin.duration * 0.1)

        import simpy
        self.twin.env = simpy.Environment()
        self.twin.env.process(self.twin._arrival_process())
        self.twin.env.process(self.twin._sync_process())

        return self._encode_state(self.twin)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        action = int(action) % ACTION_DIM
        self.current_action = action
        policy = self.policies[action]

        if self.twin is None:
            raise RuntimeError("Call reset() first")

        self.twin.policy = policy
        segment_duration = self.twin.duration / self.max_steps
        target_time = min(
            self.twin.env.now + segment_duration,
            self.twin.duration,
        )
        self.twin.env.run(until=target_time)
        self.step_count += 1

        state = self._encode_state(self.twin)
        reward = self._compute_reward(self.twin.metrics, self.twin)
        done = self.twin.env.now >= self.twin.duration - 1
        info = self.twin.metrics.summary()

        return state, reward, done, info

    def run_full_episode(self, action_sequence: Optional[List[int]] = None) -> Dict[str, float]:
        """Run complete episode with fixed or dynamic policy."""
        self.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = (
                action_sequence[self.step_count]
                if action_sequence and self.step_count < len(action_sequence)
                else self.current_action
            )
            _, reward, done, info = self.step(action)
            total_reward += reward
        info["total_reward"] = total_reward
        return info


def dqn_policy_factory(action: int) -> Callable:
    policies = get_action_policies()
    return policies[action % ACTION_DIM]
