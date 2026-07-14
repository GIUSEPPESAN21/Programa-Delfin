"""Generate all figures and tables for the manuscript."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))

OUTPUT = Path(__file__).parent / "outputs"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"
FIGURES.mkdir(parents=True, exist_ok=True)


def plot_los_comparison():
    path = TABLES / "all_results.csv"
    if not path.exists():
        path = TABLES / "baseline_results.csv"
    if not path.exists():
        print("No results found. Run experiments first.")
        return

    df = pd.read_csv(path)
    policy_labels = {
        "esi_fifo": "ESI+FIFO",
        "srpt": "SRPT",
        "heuristic": "Heurístico",
        "dqn_dt": "DQN+DT",
    }
    df["policy_label"] = df["policy"].map(policy_labels).fillna(df["policy"])

    fig, ax = plt.subplots(figsize=(9, 6))
    order = ["ESI+FIFO", "SRPT", "Heurístico", "DQN+DT"]
    available = [p for p in order if p in df["policy_label"].values]
    sns.boxplot(
        data=df, x="policy_label", y="mean_los",
        order=available, palette="Set2", ax=ax,
    )
    ax.set_xlabel("Política de despacho")
    ax.set_ylabel("LOS medio (minutos)")
    ax.set_title("Comparación de Length of Stay por política")
    fig.tight_layout()
    fig.savefig(FIGURES / "los_comparison.png", dpi=300)
    plt.close()
    print("  Saved los_comparison.png")


def plot_occupancy_trace():
    from agents.dqn import DQNAgent
    from agents.mdp import EDOrchestrationEnv, get_action_policies
    from ed_digital_twin.twin import EmergencyDepartmentTwin, POLICIES

    fig, ax = plt.subplots(figsize=(10, 5))

    twin = EmergencyDepartmentTwin(seed=42, policy=POLICIES["esi_fifo"])
    metrics = twin.run()
    if metrics.occupancy_trace:
        times, occs = zip(*metrics.occupancy_trace[::10])
        ax.plot(np.array(times) / 60, occs, label="ESI+FIFO", color="#E74C3C", alpha=0.8)

    twin = EmergencyDepartmentTwin(seed=42, policy=POLICIES["heuristic"])
    metrics = twin.run()
    if metrics.occupancy_trace:
        times, occs = zip(*metrics.occupancy_trace[::10])
        ax.plot(np.array(times) / 60, occs, label="Heurístico", color="#F39C12", alpha=0.8)

    model_path = Path(__file__).parent / "outputs" / "dqn_model.pt"
    if model_path.exists():
        agent = DQNAgent(seed=42)
        agent.load(model_path)
        policies = get_action_policies()

        def adaptive_policy(queue, now):
            if not hasattr(adaptive_policy, "_action"):
                adaptive_policy._action = 0
            return policies[adaptive_policy._action](queue, now)

        twin = EmergencyDepartmentTwin(seed=42, policy=adaptive_policy)
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
        if twin.metrics.occupancy_trace:
            times, occs = zip(*twin.metrics.occupancy_trace[::10])
            ax.plot(np.array(times) / 60, occs, label="DQN+DT", color="#2E86AB", alpha=0.8)

    ax.axhline(y=25, color="gray", linestyle="--", alpha=0.5, label="Capacidad camas")
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Pacientes en sistema")
    ax.set_title("Evolución temporal de ocupación")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "occupancy_trace.png", dpi=300)
    plt.close()
    print("  Saved occupancy_trace.png")


def plot_equity_by_esi():
    from ed_digital_twin.twin import EmergencyDepartmentTwin, POLICIES

    fig, ax = plt.subplots(figsize=(8, 5))
    policies = ["esi_fifo", "srpt", "heuristic"]
    labels = ["ESI+FIFO", "SRPT", "Heurístico"]
    x = np.arange(1, 6)
    width = 0.25

    for idx, (pname, label) in enumerate(zip(policies, labels)):
        waits = {i: [] for i in range(1, 6)}
        for seed in range(10):
            twin = EmergencyDepartmentTwin(seed=seed, policy=POLICIES[pname])
            m = twin.run()
            for esi in range(1, 6):
                if m.wait_by_esi[esi]:
                    waits[esi].append(np.mean(m.wait_by_esi[esi]))
        means = [np.mean(waits[i]) if waits[i] else 0 for i in range(1, 6)]
        ax.bar(x + idx * width, means, width, label=label, alpha=0.85)

    ax.set_xlabel("Nivel ESI")
    ax.set_ylabel("Tiempo de espera medio (min)")
    ax.set_title("Equidad de acceso: espera por grupo de riesgo")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"ESI-{i}" for i in range(1, 6)])
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "equity_by_esi.png", dpi=300)
    plt.close()
    print("  Saved equity_by_esi.png")


def plot_robustness():
    path = TABLES / "robustness_results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    policy_labels = {"esi_fifo": "ESI+FIFO", "srpt": "SRPT", "heuristic": "Heurístico", "dqn_dt": "DQN+DT"}
    df["policy_label"] = df["policy"].map(policy_labels).fillna(df["policy"])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=df, x="scenario", y="mean_los", hue="policy_label",
        palette="Set2", ax=ax, errorbar="sd",
    )
    ax.set_xlabel("Escenario")
    ax.set_ylabel("LOS medio (minutos)")
    ax.set_title("Análisis de robustez ante variabilidad extrema")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIGURES / "robustness.png", dpi=300)
    plt.close()
    print("  Saved robustness.png")


def generate_comparison_table():
    path = TABLES / "all_results.csv"
    if not path.exists():
        path = TABLES / "baseline_results.csv"
    if not path.exists():
        return

    df = pd.read_csv(path)
    policy_labels = {
        "esi_fifo": "ESI+FIFO",
        "srpt": "SRPT",
        "heuristic": "Heurístico",
        "dqn_dt": "DQN+DT",
    }

    rows = []
    for policy in df["policy"].unique():
        subset = df[df["policy"] == policy]
        rows.append({
            "Política": policy_labels.get(policy, policy),
            "LOS medio (min)": f"{subset['mean_los'].mean():.1f} ± {subset['mean_los'].std():.1f}",
            "LOS P95 (min)": f"{subset['p95_los'].mean():.1f}",
            "Ocupación máx.": f"{subset['max_occupancy'].mean():.1f}",
            "Espera crítica (min)": f"{subset['mean_critical_wait'].mean():.1f}",
            "Índice equidad": f"{subset['equity_index'].mean():.3f}",
            "Throughput": f"{subset['throughput'].mean():.0f}",
        })

    table_df = pd.DataFrame(rows)
    table_df.to_csv(TABLES / "comparison_table.csv", index=False)
    print("  Saved comparison_table.csv")
    return table_df


def main():
    print("Generating manuscript assets...")
    plot_los_comparison()
    plot_occupancy_trace()
    plot_equity_by_esi()
    plot_robustness()
    table = generate_comparison_table()
    if table is not None:
        print("\n=== Comparison Table ===")
        print(table.to_string(index=False))
    print(f"\nAll assets saved to {OUTPUT}")


if __name__ == "__main__":
    main()
