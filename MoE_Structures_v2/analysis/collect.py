"""Gather run outputs into tables.

* ``load_results(runs_root)``  one row per finished run (result.json): ids, accuracies, params, time,
  per-task accuracies (``task:<name>`` columns) and routing summaries (mean entropy / max load over
  the run's routed modules).
* ``load_history(runs_root)``  one row per evaluation step (history.jsonl) for learning curves.
* ``agg(df, by)``               mean, std and n of test accuracy per group (seeds pooled).
* ``separation(df)``            dataset screening score: how far apart the structures are,
  relative to seed noise (see docstring).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLS = ["dataset", "structure", "variant", "balance", "seed"]


def load_results(runs_root) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(runs_root).glob("*/result.json")):
        r = json.loads(p.read_text())
        if r.get("task") == "dynamics":          # analysis/dynamics.py collects these
            continue
        row = {k: r[k] for k in ID_COLS + ["test_acc", "valid_acc", "test_seq_acc", "test_loss", "params",
                                           "params_total", "train_time_s", "steps"]}
        row["run"] = r["run"]
        for t, a in r.get("test_per_task", {}).items():
            row[f"task:{t}"] = a
        rt = r.get("routing", {})
        if rt:
            row["route_entropy"] = float(np.mean([v["entropy"] for v in rt.values()]))
            row["route_max_load"] = float(np.mean([v["max_load"] for v in rt.values()]))
            row["route_min_entropy"] = float(np.min([v["entropy"] for v in rt.values()]))
            row["route_dead"] = int(np.sum([v["dead"] for v in rt.values()]))
        rows.append(row)
    return pd.DataFrame(rows)


def load_routing(runs_root) -> pd.DataFrame:
    """One row per (run, routed module): load vector, entropy, max_load, per-task counts."""
    rows = []
    for p in sorted(Path(runs_root).glob("*/result.json")):
        r = json.loads(p.read_text())
        if r.get("task") == "dynamics":
            continue
        cfg = json.loads((p.parent / "config.json").read_text())
        # MoH shared heads run for every token and are never routed; the usage vector still lists them (always 0).
        # Drop them so entropy / dead count describe the routed heads only.
        n_shared = cfg.get("structure_params", {}).get("shared_head", 0) if r["structure"] == "moh" else 0
        for mod, v in r.get("routing", {}).items():
            stats = {k: v[k] for k in ("entropy", "max_load", "dead")}
            load = v["load"]
            per_task = v["per_task"]
            if n_shared:
                load = load[n_shared:]
                per_task = {t: c[n_shared:] for t, c in per_task.items()}
                stats = _load_stats(load)
            rows.append({**{k: r[k] for k in ID_COLS}, "module": mod, **stats, "load": load, "per_task": per_task})
    return pd.DataFrame(rows)


def _load_stats(load):
    """Same definitions as backbone/metrics.py routing_stats."""
    q = np.asarray(load, dtype=float)
    q = q / q.sum() if q.sum() > 0 else q
    nz = q[q > 0]
    ent = float(-(nz * np.log(nz)).sum() / np.log(len(q))) if len(q) > 1 else 1.0
    return {"entropy": ent, "max_load": float(q.max()), "dead": int((q < 0.01 / len(q)).sum())}


def load_history(runs_root) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(runs_root).glob("*/history.jsonl")):
        cfg = json.loads((p.parent / "config.json").read_text())
        if cfg.get("task") == "dynamics":
            continue
        ids = {"dataset": cfg["dataset"], "structure": cfg["structure"], "variant": cfg["variant"],
               "balance": cfg["routing"]["balance"], "seed": cfg["train"]["seed"]}
        for line in p.read_text().splitlines():
            h = json.loads(line)
            rows.append({**ids, **{k: h[k] for k in ("step", "train_loss", "train_aux", "valid_acc", "valid_loss")}})
    return pd.DataFrame(rows)


def agg(df: pd.DataFrame, by, col="test_acc") -> pd.DataFrame:
    g = df.groupby(by)[col]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"])
    return out


def separation(df: pd.DataFrame) -> pd.DataFrame:
    """Per dataset (default variants, balance 'none'):

    * ``spread``     std across structures of their seed-mean test accuracy
    * ``range``      best minus worst structure
    * ``seed_sd``    pooled within-structure seed std (only where >= 2 seeds)
    * ``score``      spread / seed_sd (signal-to-noise; NaN without repeated seeds)
    * ``ceiling``    share of structures above 0.98 (a task everyone solves cannot rank them)
    * ``floor``      share of structures below 2x chance-level-ish 0.05
    """
    d = df[(df.variant == "default") & (df.balance == "none")]
    rows = []
    for ds, g in d.groupby("dataset"):
        m = g.groupby("structure").test_acc.mean()
        sd = g.groupby("structure").test_acc.std().dropna()
        pooled = float(np.sqrt((sd ** 2).mean())) if len(sd) else np.nan
        rows.append({"dataset": ds, "n_struct": len(m), "mean": m.mean(), "spread": m.std(), "range": m.max() - m.min(),
                     "seed_sd": pooled, "score": m.std() / pooled if pooled and pooled > 0 else np.nan,
                     "ceiling": float((m > 0.98).mean()), "floor": float((m < 0.05).mean()),
                     "best": m.idxmax(), "worst": m.idxmin()})
    return pd.DataFrame(rows).sort_values("spread", ascending=False)


def paired_delta(df: pd.DataFrame, a: str, b: str, by=("dataset", "structure", "variant")) -> pd.DataFrame:
    """Seed-paired difference of test accuracy between balance modes a and b (a - b)."""
    p = df.pivot_table(index=list(by) + ["seed"], columns="balance", values="test_acc").reset_index()
    if a not in p or b not in p:
        return pd.DataFrame()
    p["delta"] = p[a] - p[b]
    out = p.groupby(list(by)).delta.agg(["mean", "std", "count"]).reset_index()
    out["comparison"] = f"{a} - {b}"
    return out
