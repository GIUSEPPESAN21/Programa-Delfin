"""Prepare synthetic ED historical logs for offline training."""

from pathlib import Path

import pandas as pd
import numpy as np

out = Path("data/processed/ed_logs.parquet")
out.parent.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
n = 5000
df = pd.DataFrame({
    "arrival_hour": rng.integers(0, 24, n),
    "esi": rng.choice([1, 2, 3, 4, 5], n, p=[0.02, 0.08, 0.25, 0.40, 0.25]),
    "los_minutes": rng.lognormal(4.5, 0.4, n),
    "admitted": rng.random(n) < 0.2,
})
df.to_parquet(out, index=False)
print(f"Wrote {out}")
