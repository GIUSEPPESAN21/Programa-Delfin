"""Real-time state manager with bidirectional digital twin sync."""

from __future__ import annotations

import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ed_orchestrator.api.schemas import EDStateSnapshot, LookaheadResult
from ed_orchestrator.config import CONFIG

_SIM_PATH = Path(__file__).resolve().parents[3] / "simulation"
if str(_SIM_PATH) not in sys.path:
    sys.path.insert(0, str(_SIM_PATH))

from ed_digital_twin.sync import BidirectionalSync  # noqa: E402


class StateManager:
    """Ingests telemetry, maintains twin sync, runs look-ahead projections."""

    def __init__(self, sync_interval: Optional[float] = None):
        self.sync = BidirectionalSync(sync_interval=sync_interval)
        self.latest_snapshot: Optional[EDStateSnapshot] = None
        self.rng = np.random.default_rng(42)
        self._state_dim = 17

    async def ingest_telemetry(self, snapshot: EDStateSnapshot) -> str:
        snapshot_id = str(uuid.uuid4())
        self.latest_snapshot = snapshot
        ed_state = self._to_ed_dict(snapshot)
        self.sync.pull_from_physical(ed_state, snapshot.timestamp.timestamp())
        return snapshot_id

    async def run_lookahead(self, horizon_minutes: int = 120) -> LookaheadResult:
        if self.latest_snapshot is None:
            raise RuntimeError("No telemetry ingested yet")
        ed_state = self._to_ed_dict(self.latest_snapshot)
        n_scenarios = min(50, CONFIG["simulation"].get("scenarios_per_minute", 500) // 10)
        result = self.sync.run_lookahead(
            ed_state, self.rng, n_scenarios=n_scenarios, horizon_minutes=horizon_minutes
        )
        self.sync.push_to_virtual(result)
        self.sync.log_sync()
        return LookaheadResult(
            mean_occupancy=result["mean_occupancy"],
            std_occupancy=result["std_occupancy"],
            n_scenarios=result["n_scenarios"],
            latency_ms=result["latency_ms"],
            horizon_minutes=horizon_minutes,
        )

    def get_current_state_vector(self) -> np.ndarray:
        if self.latest_snapshot is None:
            return np.zeros(self._state_dim, dtype=np.float32)
        return self._encode_snapshot(self.latest_snapshot)

    def _to_ed_dict(self, snapshot: EDStateSnapshot) -> Dict[str, Any]:
        return {
            "queue_by_esi": snapshot.queue_by_esi,
            "wait_times_by_esi": snapshot.wait_times_by_esi,
            "beds_occupied": snapshot.beds_occupied,
            "consult_occupied": snapshot.consult_occupied,
            "imaging_occupied": snapshot.imaging_occupied,
            "patients_in_system": snapshot.patients_in_system,
            "hourly_arrivals": snapshot.hourly_arrivals,
        }

    def _encode_snapshot(self, snapshot: EDStateSnapshot) -> np.ndarray:
        cap = CONFIG["capacity"]
        hour = snapshot.timestamp.hour + snapshot.timestamp.minute / 60.0
        q = snapshot.queue_by_esi
        w = snapshot.wait_times_by_esi
        vitals = snapshot.vitals_summary
        vec = (
            [q.get(i, 0) / 10.0 for i in range(1, 6)]
            + [min(w.get(i, 0) / 120.0, 1.0) for i in range(1, 6)]
            + [
                snapshot.beds_occupied / cap["beds"],
                snapshot.consult_occupied / cap["consult_rooms"],
                snapshot.imaging_occupied / cap["imaging_rooms"],
                hour / 24.0,
                snapshot.patients_in_system / cap["beds"],
                vitals.get("spo2", 97) / 100.0,
                vitals.get("heart_rate", 80) / 200.0,
            ]
        )
        return np.array(vec[: self._state_dim], dtype=np.float32)
