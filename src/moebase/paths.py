"""Project paths and pinned upstream sources.

Every path can be overridden with an environment variable so that the same code
runs on the workstation (Windows) and on a cluster.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Generated datasets, manifests and run records live here (override with MOE_BASE_DATA_DIR).
DATA_DIR = Path(os.environ.get("MOE_BASE_DATA_DIR", PROJECT_ROOT / "data"))
# Upstream repositories are cloned here at pinned commits by scripts/fetch_upstream.py.
THIRD_PARTY_DIR = Path(os.environ.get("MOE_BASE_THIRD_PARTY", PROJECT_ROOT / "third_party"))
RUNS_DIR = Path(os.environ.get("MOE_BASE_RUNS_DIR", PROJECT_ROOT / "runs"))

# Bump whenever data preparation behaviour changes (plan section 13).
DATA_VERSION = "logic-v1"

# Commits audited on 2026-10-17. Changing any of these invalidates parity results.
UPSTREAM = {
    "SUT": {
        "url": "https://github.com/shawntan/SUT",
        "commit": "192cbd3ed567e45e6ec6d33eb1ea9c3b331bc2b0",
        "license": "no LICENSE file; source files carry the fairseq (Facebook, MIT) header",
    },
    "Mixture-of-Modules": {
        "url": "https://github.com/gzhch/Mixture-of-Modules",
        "commit": "2ed1d28c320c0a8d114b203b2c68675ffa489d45",
        "license": "no LICENSE file; source files carry the HuggingFace Apache-2.0 header",
    },
    "Ordered-Memory": {
        "url": "https://github.com/yikangshen/Ordered-Memory",
        "commit": "2a0d3a22fb70216993a7faf25de3515f42e10431",
        "license": "MIT",
    },
}


def upstream_dir(name: str) -> Path:
    return THIRD_PARTY_DIR / name


def logic_raw_dir() -> Path:
    """Released propositional-logic files (train0..12, test0..12) from Ordered-Memory."""
    override = os.environ.get("MOE_BASE_LOGIC_RAW")
    if override:
        return Path(override)
    return upstream_dir("Ordered-Memory") / "data" / "propositionallogic"
