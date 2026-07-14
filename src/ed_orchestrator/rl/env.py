"""Gymnasium environment for RLlib training."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

_SIM_PATH = Path(__file__).resolve().parents[3] / "simulation"
if str(_SIM_PATH) not in sys.path:
    sys.path.insert(0, str(_SIM_PATH))

from agents.mdp import ACTION_DIM, STATE_DIM, EDOrchestrationEnv  # noqa: E402


class EDOrchestrationGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None):
        super().__init__()
        cfg = config or {}
        self._env = EDOrchestrationEnv(
            seed=cfg.get("seed", 0),
            episode_hours=cfg.get("episode_hours", 8.0),
        )
        self.observation_space = spaces.Box(0.0, 1.0, shape=(STATE_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(ACTION_DIM)

    def reset(self, *, seed=None, options=None):
        obs = self._env.reset(seed=seed)
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self._env.step(int(action))
        return obs, reward, done, False, info


def register_env():
    gym.register(
        id="EDOrchestration-v0",
        entry_point="ed_orchestrator.rl.env:EDOrchestrationGymEnv",
    )
