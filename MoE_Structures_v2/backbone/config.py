"""Run configuration: one YAML per run is assembled from three layers.

    configs/base.yaml                      shared backbone + training settings (every structure)
    models/<structure>/configs/<variant>.yaml   the structure's own parameters for that variant
    command-line / sweep overrides         dataset, balancing mode, seed, steps

`load_run_config(structure, variant, overrides)` merges them (later wins) and returns a plain dict.
`base:` keys inside a YAML file name another YAML to inherit from (relative to that file).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .paths import PROJECT_ROOT


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def load_yaml(path: Path) -> Dict[str, Any]:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = data.pop("base", None)
    if base:
        merged = load_yaml((path.parent / base).resolve())
        return deep_update(merged, data)
    return data


def variant_path(structure: str, variant: str) -> Path:
    return PROJECT_ROOT / "models" / structure / "configs" / f"{variant}.yaml"


def list_variants(structure: str):
    return sorted(p.stem for p in (PROJECT_ROOT / "models" / structure / "configs").glob("*.yaml"))


def set_dotted(cfg: Dict[str, Any], dotted: str, value: Any):
    keys = dotted.split(".")
    cur = cfg
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def load_run_config(structure: str, variant: str = "default", overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = load_yaml(PROJECT_ROOT / "configs" / "base.yaml")
    deep_update(cfg, load_yaml(variant_path(structure, variant)))
    cfg["structure"] = structure
    cfg["variant"] = variant
    for k, v in (overrides or {}).items():
        if "." in k:
            set_dotted(cfg, k, v)
        elif isinstance(v, dict) and isinstance(cfg.get(k), dict):
            deep_update(cfg[k], v)
        else:
            cfg[k] = v
    return cfg


def run_name(cfg: Dict[str, Any]) -> str:
    return f"{cfg['dataset']}__{cfg['structure']}__{cfg['variant']}__{cfg['routing']['balance']}__s{cfg['train']['seed']}"


def dumps(cfg: Dict[str, Any]) -> str:
    return json.dumps(cfg, indent=1, sort_keys=True)
