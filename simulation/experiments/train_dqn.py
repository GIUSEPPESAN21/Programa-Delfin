"""Train DQN agent and evaluate against baselines."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.dqn import DQNAgent, AdaptiveDQNPolicy
from agents.mdp import EDOrchestrationEnv, get_action_policies
from ed_digital_twin.distributions import CONFIG
from ed_digital_twin.twin import EmergencyDepartmentTwin, POLICIES


def evaluate_dqn_policy(agent: DQNAgent, n_seeds: int = 30) -> pd.DataFrame:
    policies = get_action_policies()
    rows = []
    for i in range(n_seeds):
        seed = i

        def adaptive_policy(queue, now):
            if not hasattr(adaptive_policy, "_action"):
                adaptive_policy._action = 0
            return policies[adaptive_policy._action](queue, now)

        twin = EmergencyDepartmentTwin(seed=seed, policy=adaptive_policy)

        original_run = twin.run

        def run_with_dqn():
            twin.env = __import__("simpy").Environment()
            twin.env.process(twin._arrival_process())
            twin.env.process(twin._sync_process())

            def sync_and_update():
                interval = twin.sync.sync_interval / 60.0
                while True:
                    yield twin.env.timeout(interval)
                    twin._sync_cycle()
                    state = EDOrchestrationEnv()._encode_state(twin)
                    action = agent.get_best_action(state)
                    adaptive_policy._action = action

            twin.env.process(sync_and_update())
            twin.env.run(until=twin.duration)
            return twin.metrics

        metrics = run_with_dqn()
        summary = metrics.summary()
        summary["policy"] = "dqn_dt"
        summary["seed"] = seed
        rows.append(summary)
        if (i + 1) % 10 == 0:
            print(f"  dqn_dt: {i+1}/{n_seeds} completed")
    return pd.DataFrame(rows)


def main():
    output_dir = Path(__file__).parent.parent / "outputs"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    n_episodes = CONFIG["dqn"]["training_episodes"]
    print(f"Training DQN for {n_episodes} episodes...")
    agent = DQNAgent(seed=42)
    history = agent.train(n_episodes=n_episodes, seed_offset=0)

    agent.save(output_dir / "dqn_model.pt")

    history_path = tables_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(
            {
                "rewards": history["reward_history"][-500:],
                "losses": history["loss_history"][-500:],
            },
            f,
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rewards = history["reward_history"]
    window = 50
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(smoothed, color="#2E86AB", linewidth=1.5)
    ax.set_xlabel("Episodio (media móvil 50)")
    ax.set_ylabel("Recompensa acumulada")
    ax.set_title("Curva de convergencia del agente DQN")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "convergence_dqn.png", dpi=300)
    plt.close()

    n_seeds = CONFIG["experiments"]["monte_carlo_seeds"]
    print(f"\nEvaluating DQN+DT policy ({n_seeds} seeds)...")
    dqn_results = evaluate_dqn_policy(agent, n_seeds=n_seeds)
    dqn_results.to_csv(tables_dir / "dqn_results.csv", index=False)

    if (tables_dir / "baseline_results.csv").exists():
        baselines = pd.read_csv(tables_dir / "baseline_results.csv")
        combined = pd.concat([baselines, dqn_results], ignore_index=True)
        combined.to_csv(tables_dir / "all_results.csv", index=False)

    print("Training and evaluation complete.")


if __name__ == "__main__":
    main()
