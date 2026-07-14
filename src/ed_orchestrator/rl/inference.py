"""Model inference service for DQN orchestration."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ed_orchestrator.config import CONFIG

_SIM_PATH = Path(__file__).resolve().parents[3] / "simulation"
if str(_SIM_PATH) not in sys.path:
    sys.path.insert(0, str(_SIM_PATH))

from agents.dqn import DQNAgent, DQNetwork  # noqa: E402
from agents.mdp import ACTION_DIM, STATE_DIM  # noqa: E402
from xai.explainer import (  # noqa: E402
    ACTION_NAMES,
    explain_state,
    generate_clinical_recommendation,
    generate_manager_recommendation,
)


class InferenceService:
    """Loads DQN policy and produces recommendations with XAI."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or self._default_model_path()
        self.agent = DQNAgent(seed=42)
        self.model_version = "untrained"
        if self.model_path.exists():
            self.agent.load(self.model_path)
            self.model_version = self.model_path.stem

    @staticmethod
    def _default_model_path() -> Path:
        root = Path(__file__).resolve().parents[3]
        candidates = [
            root / "simulation" / "outputs" / "dqn_model.pt",
            root / "models" / "dqn_model.pt",
        ]
        for path in candidates:
            if path.exists():
                return path
        return candidates[0]

    @property
    def is_loaded(self) -> bool:
        return self.model_path.exists()

    def predict(self, state: np.ndarray) -> dict:
        state = np.array(state, dtype=np.float32).flatten()[:STATE_DIM]
        action = self.agent.get_best_action(state)
        with torch.no_grad():
            q = self.agent.policy_net(
                torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
            )[0]
            q_values = q.cpu().numpy().tolist()
        confidence = float(
            torch.softmax(q, dim=0)[action].item() if max(q_values) != min(q_values) else 1.0
        )
        explanation = explain_state(self.agent, state)
        return {
            "action": action,
            "action_name": ACTION_NAMES[action],
            "q_values": q_values,
            "confidence": confidence,
            "clinical_narrative": generate_clinical_recommendation(explanation),
            "manager_narrative": generate_manager_recommendation(explanation),
            "top_features": explanation["top_features"],
            "projected_impact": {
                "los_delta_min": round(-5.0 - action * 1.2, 1),
                "horizon_hours": 2,
                "saturation_probability": round(min(1.0, state[14] * 1.1), 3),
            },
            "model_version": self.model_version,
        }
