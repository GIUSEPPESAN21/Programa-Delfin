"""XAI module: SHAP-based explanations for DRL orchestration decisions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from agents.mdp import STATE_DIM, EDOrchestrationEnv
from agents.dqn import DQNAgent, DQNetwork


STATE_FEATURE_NAMES = [
    "cola_ESI1", "cola_ESI2", "cola_ESI3", "cola_ESI4", "cola_ESI5",
    "espera_ESI1", "espera_ESI2", "espera_ESI3", "espera_ESI4", "espera_ESI5",
    "ocupacion_camas", "ocupacion_consultorios", "ocupacion_imagenologia",
    "hora_dia", "pacientes_sistema", "SpO2_promedio", "FC_promedio",
]

ACTION_NAMES = [
    "Prioridad ESI estricta",
    "Refuerzo casos críticos (ESI 1-2)",
    "Fast-track ESI 4-5",
    "Prioridad imagenología",
    "Prioridad admisión hospitalaria",
    "Balance clínico-operativo",
]


def explain_state(agent: DQNAgent, state: np.ndarray) -> Dict[str, Any]:
    """
    Compute SHAP-like feature attributions using perturbation-based approximation.
    Falls back gracefully if full SHAP library unavailable.
    """
    state = np.array(state, dtype=np.float32).flatten()[:STATE_DIM]
    action = agent.get_best_action(state)

    try:
        import shap

        def model_fn(x):
            with __import__("torch").no_grad():
                t = __import__("torch").FloatTensor(x)
                if hasattr(agent, "device"):
                    t = t.to(agent.device)
                q = agent.policy_net(t)
                return q.cpu().numpy()

        background = np.zeros((1, STATE_DIM), dtype=np.float32)
        explainer = shap.KernelExplainer(model_fn, background)
        shap_values = explainer.shap_values(state.reshape(1, -1), nsamples=100)
        if isinstance(shap_values, list):
            attributions = shap_values[action][0]
        else:
            attributions = shap_values[0, action] if shap_values.ndim == 3 else shap_values[0]
    except Exception:
        attributions = _perturbation_shap(agent, state, action)

    top_indices = np.argsort(np.abs(attributions))[::-1][:5]
    top_features = [
        {
            "feature": STATE_FEATURE_NAMES[i],
            "value": float(state[i]),
            "attribution": float(attributions[i]),
        }
        for i in top_indices
    ]

    return {
        "action": action,
        "action_name": ACTION_NAMES[action],
        "top_features": top_features,
        "attributions": attributions.tolist(),
    }


def _perturbation_shap(agent: DQNAgent, state: np.ndarray, action: int) -> np.ndarray:
    """Simple perturbation-based attribution fallback."""
    import torch

    baseline = np.zeros_like(state)
    with torch.no_grad():
        base_q = agent.policy_net(
            torch.FloatTensor(baseline).unsqueeze(0).to(agent.device)
        )[0, action].item()
        attributions = np.zeros(STATE_DIM)
        for i in range(STATE_DIM):
            perturbed = baseline.copy()
            perturbed[i] = state[i]
            q = agent.policy_net(
                torch.FloatTensor(perturbed).unsqueeze(0).to(agent.device)
            )[0, action].item()
            attributions[i] = q - base_q
    return attributions


def generate_clinical_recommendation(explanation: Dict[str, Any]) -> str:
    """Translate algorithmic inference to operational recommendation."""
    action = explanation["action_name"]
    features = explanation["top_features"]

    lines = [f"Recomendación operativa: {action}"]
    lines.append("Justificación basada en covariables:")

    for f in features[:3]:
        name = f["feature"]
        val = f["value"]
        attr = f["attribution"]
        direction = "incrementa" if attr > 0 else "reduce"
        if "espera" in name and val > 0.3:
            lines.append(
                f"  - {name}: valor elevado ({val:.2f}), lo que {direction} la prioridad de despacho."
            )
        elif "cola" in name and val > 0.5:
            lines.append(
                f"  - {name}: congestión detectada ({val:.2f} normalizado)."
            )
        elif "SpO2" in name:
            lines.append(
                f"  - SpO2 promedio del sistema: {val*100:.1f}% — factor fisiológico considerado."
            )
        elif "ocupacion" in name:
            lines.append(
                f"  - {name}: {val*100:.0f}% — restricción logística activa."
            )

    lines.append(
        "Impacto proyectado: reducción estimada de espera en casos críticos "
        "mediante reasignación dinámica de recursos."
    )
    return "\n".join(lines)


def generate_manager_recommendation(explanation: Dict[str, Any]) -> str:
    """Operational panel for hospital managers."""
    action = explanation["action_name"]
    features = explanation["top_features"]

    lines = [f"Directiva de orquestación: {action}"]
    lines.append("Indicadores operativos determinantes:")
    for f in features[:4]:
        lines.append(f"  • {f['feature']}: {f['value']:.3f} (peso: {f['attribution']:+.3f})")
    lines.append(
        "Nota: Esta recomendación es de apoyo a la decisión clínica. "
        "Requiere validación del equipo de turno antes de implementación."
    )
    return "\n".join(lines)


def demo_explanation(seed: int = 42) -> Dict[str, str]:
    """Generate example XAI output for manuscript."""
    env = EDOrchestrationEnv(seed=seed, episode_hours=4.0)
    state = env.reset(seed=seed)
    env.step(0)
    state = env._encode_state(env.twin)

    agent = DQNAgent(seed=seed)
    agent.train(n_episodes=200, seed_offset=seed)

    explanation = explain_state(agent, state)
    return {
        "clinical": generate_clinical_recommendation(explanation),
        "manager": generate_manager_recommendation(explanation),
        "explanation": explanation,
    }
