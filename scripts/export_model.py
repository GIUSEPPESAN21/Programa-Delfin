"""Export trained model to production models directory."""

import shutil
from pathlib import Path

src = Path("simulation/outputs/dqn_model.pt")
dst = Path("models/dqn_model.pt")
dst.parent.mkdir(parents=True, exist_ok=True)
if src.exists():
    shutil.copy(src, dst)
    print(f"Exported {dst}")
else:
    raise FileNotFoundError(src)
