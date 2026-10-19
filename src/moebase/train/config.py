"""Load YAML/JSON run configs with optional ``base:`` inheritance and dotted overrides."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import List


def _load_file(path: Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml  # pyyaml
        return yaml.safe_load(text)
    return json.loads(text)


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: Path) -> dict:
    path = Path(path)
    cfg = _load_file(path)
    base = cfg.pop("base", None)
    if base:
        cfg = deep_merge(load_config(path.parent / base), cfg)
    return cfg


def _parse_value(s: str):
    try:
        import yaml
        return yaml.safe_load(s)
    except ImportError:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return s


def apply_overrides(cfg: dict, sets: List[str]) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in sets:
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if node.get(p) is None:
                node[p] = {}
            node = node[p]
        node[parts[-1]] = _parse_value(val)
    return cfg
