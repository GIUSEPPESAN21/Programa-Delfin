"""Basic tests for ED orchestrator."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "simulation"))


def test_gym_env_reset_step():
    from ed_orchestrator.rl.env import EDOrchestrationGymEnv

    env = EDOrchestrationGymEnv({"seed": 1, "episode_hours": 2.0})
    obs, _ = env.reset(seed=1)
    assert obs.shape == (17,)
    obs2, reward, done, truncated, info = env.step(0)
    assert obs2.shape == (17,)
    assert isinstance(reward, float)


def test_state_manager_encode():
    from datetime import datetime

    from ed_orchestrator.api.schemas import EDStateSnapshot
    from ed_orchestrator.digital_twin.state_manager import StateManager

    mgr = StateManager()
    snap = EDStateSnapshot(
        timestamp=datetime.utcnow(),
        queue_by_esi={1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        patients_in_system=15,
        beds_occupied=10,
    )
    vec = mgr._encode_snapshot(snap)
    assert vec.shape == (17,)
    assert np.all(vec >= 0)


def test_api_health():
    from fastapi.testclient import TestClient
    from ed_orchestrator.api.main import app

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
