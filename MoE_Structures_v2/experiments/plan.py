"""The experiment plan: which trials make up each phase. Every trial is a dict
{structure, variant, dataset, balance, seed}; its run directory is named by backbone.config.run_name.

Phases
* calibrate  Dense on every dataset, 1 seed: checks each task is learnable but not trivial in budget.
* screen     every structure (default config) x every dataset, balance 'none', seeds 0-1:
             which dataset separates the structures most (spread across structures vs seed noise).
* main       on the chosen dataset(s): every routed structure x {none, lb, warmup} x seeds 0-2,
             plus Dense and Dense-wide x seeds 0-2.
* variants   on the chosen dataset(s): every non-default variant x {none, lb, warmup} x seeds 0-1
             (hash routing has no router: 'none' only).

Switching-dynamics study (datasets dyn / dyn_fast; MoE, MoH, MoA, MoMHA only; no balancing):
* dyn_main      Dense, Dense-wide, MoE, MoH, MoA, MoMHA (default configs) x {dyn, dyn_fast} x seeds 0-2.
* dyn_variants  on dyn: MoE top1; MoH k6, k8; MoA k6, k8, renorm_detached; MoMHA top1 x seeds 0-2.
                (MoE hash needs token ids and MoH shared2 uses shared heads: not run on dyn.)
"""
from __future__ import annotations

from backbone.config import list_variants

DATASETS = ["mqar", "fact", "mod", "copy", "rev", "logic", "mix"]
STRUCTURES = ["dense", "dense_wide", "moe", "moh", "moa", "momha", "mom", "mita"]
ROUTED = ["moe", "moh", "moa", "momha", "mom", "mita"]
BALANCES = ["none", "lb", "warmup"]


DYN_DATASETS = ["dyn", "dyn_fast"]
DYN_STRUCTURES = ["dense", "dense_wide", "moe", "moh", "moa", "momha"]
DYN_VARIANTS = [("moe", "top1"), ("moh", "k6"), ("moh", "k8"), ("moa", "k6"), ("moa", "k8"),
                ("moa", "renorm_detached"), ("momha", "top1")]


def trial(structure, dataset, balance="none", seed=0, variant="default"):
    return {"structure": structure, "variant": variant, "dataset": dataset, "balance": balance, "seed": seed}


def balances_for(structure, variant):
    if structure not in ROUTED or variant == "hash":
        return ["none"]
    return BALANCES


def phase(name, datasets=None, seeds=None):
    ds = datasets or DATASETS
    if name == "calibrate":
        return [trial("dense", d) for d in ds]
    if name == "screen":
        return [trial(s, d, seed=sd) for sd in (seeds or [0, 1]) for d in ds for s in STRUCTURES]
    if name == "main":
        out = []
        for sd in (seeds or [0, 1, 2]):
            for d in ds:
                for s in STRUCTURES:
                    for b in balances_for(s, "default"):
                        out.append(trial(s, d, b, sd))
        return out
    if name == "variants":
        out = []
        for sd in (seeds or [0, 1]):
            for d in ds:
                for s in ROUTED:
                    for v in list_variants(s):
                        if v == "default":
                            continue
                        for b in balances_for(s, v):
                            out.append(trial(s, d, b, sd, v))
        return out
    if name == "dyn_main":
        return [trial(s, d, "none", sd) for sd in (seeds or [0, 1, 2]) for d in (datasets or DYN_DATASETS)
                for s in DYN_STRUCTURES]
    if name == "dyn_variants":
        return [trial(s, d, "none", sd, v) for sd in (seeds or [0, 1, 2]) for d in (datasets or ["dyn"])
                for s, v in DYN_VARIANTS]
    raise ValueError(name)
