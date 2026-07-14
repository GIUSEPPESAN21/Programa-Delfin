"""Robustness analysis under extreme stochastic conditions."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.dqn import DQNAgent
from agents.mdp import EDOrchestrationEnv, get_action_policies
from ed_digital_twin.distributions import CONFIG
from ed_digital_twin.twin import EmergencyDepartmentTwin, POLICIES


def evaluate_dqn_scenario(
    agent: DQNAgent,
    seed: int,
    demand_multiplier: float,
    resource_multiplier: float,
) -> dict:
    policies = get_action_policies()

    def adaptive_policy(queue, now):
        if not hasattr(adaptive_policy, "_action"):
            adaptive_policy._action = 0
        return policies[adaptive_policy._action](queue, now)

    twin = EmergencyDepartmentTwin(
        seed=seed,
        policy=adaptive_policy,
        demand_multiplier=demand_multiplier,
        resource_multiplier=resource_multiplier,
    )

    import simpy

    twin.env = simpy.Environment()
    twin.env.process(twin._arrival_process())
    twin.env.process(twin._sync_process())

    def sync_and_update():
        interval = twin.sync.sync_interval / 60.0
        while True:
            yield twin.env.timeout(interval)
            twin._sync_cycle()
            state = EDOrchestrationEnv()._encode_state(twin)
            adaptive_policy._action = agent.get_best_action(state)

    twin.env.process(sync_and_update())
    twin.env.run(until=twin.duration)
    summary = twin.metrics.summary()
    summary["policy"] = "dqn_dt"
    summary["seed"] = seed
    return summary


def run_robustness_scenarios(n_seeds: int = 10, agent: DQNAgent | None = None) -> pd.DataFrame:
    scenarios = [
        ("baseline", 1.0, 1.0),
        ("demand_+40%", 1.0 + CONFIG["experiments"]["robustness_demand_increase"], 1.0),
        ("resources_-25%", 1.0, 1.0 - CONFIG["experiments"]["robustness_resource_reduction"]),
        (
            "combined_stress",
            1.0 + CONFIG["experiments"]["robustness_demand_increase"],
            1.0 - CONFIG["experiments"]["robustness_resource_reduction"],
        ),
    ]

    policy_names = ["esi_fifo", "srpt", "heuristic"]
    if agent is not None:
        policy_names.append("dqn_dt")

    rows = []
    for scenario_name, demand, resource in scenarios:
        for policy_name in policy_names:
            for i in range(n_seeds):
                if policy_name == "dqn_dt":
                    summary = evaluate_dqn_scenario(agent, i, demand, resource)
                else:
                    twin = EmergencyDepartmentTwin(
                        seed=i,
                        policy=POLICIES[policy_name],
                        demand_multiplier=demand,
                        resource_multiplier=resource,
                    )
                    summary = twin.run().summary()
                summary["scenario"] = scenario_name
                summary["policy"] = policy_name
                summary["seed"] = i
                rows.append(summary)
        print(f"  Scenario {scenario_name} completed")

    return pd.DataFrame(rows)


def main():
    output_dir = Path(__file__).parent.parent / "outputs" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(__file__).parent.parent / "outputs" / "dqn_model.pt"

    agent = None
    if model_path.exists():
        print(f"Loading DQN model from {model_path}")
        agent = DQNAgent(seed=42)
        agent.load(model_path)
    else:
        print("Warning: DQN model not found; run train_dqn.py first for DQN robustness.")

    print("Running robustness analysis...")
    df = run_robustness_scenarios(n_seeds=10, agent=agent)
    df.to_csv(output_dir / "robustness_results.csv", index=False)

    summary = df.groupby(["scenario", "policy"]).agg({
        "mean_los": ["mean", "std"],
        "max_occupancy": "mean",
        "equity_index": "mean",
    }).round(2)
    summary.to_csv(output_dir / "robustness_summary.csv")
    print("\n=== Robustness Summary ===")
    print(summary)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
