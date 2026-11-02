"""Where code, data and run outputs live.

Defaults (override with environment variables):
  MOEV2_DATA  datasets   E:/data/MoE_Structures_v2 on the Windows machine, else <project>/_data
  MOEV2_RUNS  run output <project>/runs
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _default_data() -> Path:
    win = Path("E:/data/MoE_Structures_v2")
    if os.name == "nt" and win.parent.exists():
        return win
    return PROJECT_ROOT / "_data"


DATA_ROOT = Path(os.environ.get("MOEV2_DATA", _default_data()))
RUNS_ROOT = Path(os.environ.get("MOEV2_RUNS", PROJECT_ROOT / "runs"))
FIG_ROOT = PROJECT_ROOT / "reports" / "figures"
