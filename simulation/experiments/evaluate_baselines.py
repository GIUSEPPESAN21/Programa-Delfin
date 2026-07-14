"""Evaluate baseline dispatch policies with Monte Carlo replication."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ed_digital_twin.distributions import CONFIG
from ed_digital_twin.twin import EmergencyDepartmentTwin, POLICIES


def run_monte_carlo(
    policy_name: str,
    n_seeds: int = 30,
    seed_start: int = 0,
) -> pd.DataFrame:
    policy = POLICIES[policy_name]
    rows = []
    for i in range(n_seeds):
        seed = seed_start + i
        twin = EmergencyDepartmentTwin(seed=seed, policy=policy)
        metrics = twin.run()
        summary = metrics.summary()
        summary["policy"] = policy_name
        summary["seed"] = seed
        rows.append(summary)
        if (i + 1) % 10 == 0:
            print(f"  {policy_name}: {i+1}/{n_seeds} completed")
    return pd.DataFrame(rows)


def bootstrap_ci(data: np.ndarray, confidence: float = 0.95) -> tuple:
    n = len(data)
    boot_means = []
    for _ in range(1000):
        sample = np.random.choice(data, size=n, replace=True)
        boot_means.append(np.mean(sample))
    alpha = (1 - confidence) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    return float(np.mean(data)), float(lo), float(hi)


def main():
    n_seeds = CONFIG["experiments"]["monte_carlo_seeds"]
    output_dir = Path(__file__).parent.parent / "outputs" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Validating Digital Twin fidelity...")
    fidelity = EmergencyDepartmentTwin.validate_fidelity()
    with open(output_dir / "fidelity_validation.json", "w") as f:
        json.dump(fidelity, f, indent=2)
    print(f"  MAPE hourly arrivals: {fidelity['mape_hourly_arrivals_pct']:.1f}%")

    all_results = []
    for policy_name in ["esi_fifo", "srpt", "heuristic"]:
        print(f"\nEvaluating {policy_name}...")
        df = run_monte_carlo(policy_name, n_seeds=n_seeds)
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(output_dir / "baseline_results.csv", index=False)

    summary_rows = []
    metrics = ["mean_los", "max_occupancy", "mean_critical_wait", "equity_index", "throughput"]
    for policy in combined["policy"].unique():
        subset = combined[combined["policy"] == policy]
        row = {"policy": policy}
        for m in metrics:
            mean, lo, hi = bootstrap_ci(subset[m].values)
            row[f"{m}_mean"] = mean
            row[f"{m}_ci_lo"] = lo
            row[f"{m}_ci_hi"] = hi
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "baseline_summary.csv", index=False)
    print("\n=== Baseline Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
