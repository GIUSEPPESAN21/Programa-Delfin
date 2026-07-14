"""Gymnasium environment wrapper for ED orchestration MDP."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from agents.mdp import ACTION_DIM, STATE_DIM, EDOrchestrationEnv


class EDOrchestrationGymEnv(gym.Env):
    """Standard Gymnasium interface for RLlib and custom training loops."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        seed: int = 0,
        episode_hours: float = 8.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self._env = EDOrchestrationEnv(seed=seed, episode_hours=episode_hours)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_DIM)
        self._seed = seed

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._seed = seed
        obs = self._env.reset(seed=self._seed)
        return obs, {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, done, info = self._env.step(action)
        return obs, reward, done, False, info

    def render(self):
        return None

    def close(self):
        return None
