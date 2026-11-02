"""Import the released reference code (pinned clones) for parity tests.

Clones (see references/README.md for commits): SUT and Mixture-of-Modules in REF1, MoA / MoH / MiTA
in REF2. fairseq / timm are not installed; the reference files import them only for small utilities,
replaced by stand-ins with the same behaviour (borrowed from pilot v1's tests/upstream_loader.py).
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import re
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REF1 = Path(os.environ.get("MOEV2_REF1", "/home/claude/ref"))
REF2 = Path(os.environ.get("MOEV2_REF2", "/home/claude/ref2"))
PILOT = Path(os.environ.get("MOEV2_PILOT", "/home/claude/work/MoE_Base_Test"))


def _stub_fairseq():
    if "fairseq" in sys.modules:
        return
    sys.path.insert(0, str(PILOT / "tests"))
    sys.path.insert(0, str(PILOT / "src"))
    import upstream_loader  # pilot v1 helper with the fairseq stand-ins
    upstream_loader._stub_fairseq()


def load_package(name, pkg_dir: Path):
    spec = importlib.util.spec_from_file_location(name, pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sut_moe():
    """SUT ``sut_layer/parallel_linear/parallel_experts`` package (MoE, ParallelExperts, compute_gating)."""
    _stub_fairseq()
    return load_package("ref_sut_pe", REF1 / "SUT" / "sut_layer" / "parallel_linear" / "parallel_experts")


def moa():
    """MoA ``moa_layer`` package: MultiheadAttention + parallel_linear.MoE."""
    _stub_fairseq()
    load_package("ref_moa_layer", REF2 / "MoA" / "moa_layer")
    return importlib.import_module("ref_moa_layer.moa_attention")


def moh_attention_class():
    """MoHAttention class source from MoH-DiT/models.py, executed without timm."""
    src = (REF2 / "MoH" / "MoH-DiT" / "models.py").read_text()
    m = re.search(r"\nclass MoHAttention\(nn.Module\):.*?(?=\n(?:class |def )\w)", src, re.S)
    ns = {"torch": torch, "nn": nn, "F": F, "Final": object, "use_fused_attn": lambda: False}
    exec("from typing import Final\n" + m.group(0), ns)
    return ns["MoHAttention"]


def pilot_mom():
    """Pilot v1's MoM port (parity-tested against Mixture-of-Modules@2ed1d28 in pilot v1)."""
    sys.path.insert(0, str(PILOT / "src"))
    return importlib.import_module("moebase.models.mom.model")
