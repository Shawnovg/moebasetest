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
"""
from __future__ import annotations

from backbone.config import list_variants

DATASETS = ["mqar", "fact", "mod", "copy", "rev", "logic", "mix"]
STRUCTURES = ["dense", "dense_wide", "moe", "moh", "moa", "momha", "mom", "mita"]
ROUTED = ["moe", "moh", "moa", "momha", "mom", "mita"]
BALANCES = ["none", "lb", "warmup"]


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
    raise ValueError(name)
