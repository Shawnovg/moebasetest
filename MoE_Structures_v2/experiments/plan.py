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
* matched_k  fact: MoH k8, MoA k6, MoMHA k3/k4 x {lb, none, warmup} x seeds 0-1; dyn: MoMHA k3/k4 x {lb, none} x seeds 0-2.
* unified   MoH-SUT and MoA-SUT: MoMHA's router (MLP, random init), gates summing to 1, switch loss 0.01
            only -- same router setup as MoMHA and MoH-block2-SUT; 5 datasets x lb x seeds 0-2.
* moh_block  MoH block2 / block4 / block2_sut (MoMHA's router + gates + switch loss) (routes groups of heads, like MoMHA) x {dyn, dyn_fast, dyn_p50, fact, mod} x lb x seeds 0-2.
* attn_baselines  MHA-8 / MQA-8 / GQA-8 (router-free attention controls) x {dyn, dyn_fast, fact, mod} x seeds 0-2.
* attn_baselines4 MQA-4 / GQA-4 (4 query heads = the routed models' active width) x same datasets/seeds.
* dyn_main_lb, dyn_variants_lb  the same routed runs with the load-balance loss (balance 'lb').
"""
from __future__ import annotations

from backbone.config import list_variants

DATASETS = ["mqar", "fact", "mod", "copy", "rev", "logic", "mix"]
STRUCTURES = ["dense", "dense_wide", "moe", "moh", "moa", "momha", "mom", "mita"]
ROUTED = ["moe", "moh", "moa", "momha", "mom", "mita"]
BALANCES = ["none", "lb", "warmup"]


DYN_DATASETS = ["dyn", "dyn_fast"]
ATTN_BASELINES = ["mha8", "mqa8", "gqa8"]   # dense counterparts of MoH / MoA / MoMHA (all 8 heads on)
ATTN_BASELINES4 = ["mqa4", "gqa4"]          # active-width controls (4 heads; MHA-4 is Dense itself)
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
    if name == "matched_k":   # same active-head counts across MoH / MoA / MoMHA (4, 6, 8 of 8 heads)
        missing = [("moh", "k8"), ("moa", "k6"), ("momha", "k3"), ("momha", "k4")]
        fact = [trial(s, "fact", b, sd, v) for b in ("lb", "none", "warmup") for sd in (seeds or [0, 1])
                for s, v in missing]
        dyn = [trial("momha", "dyn", b, sd, v) for b in ("lb", "none") for sd in (seeds or [0, 1, 2])
               for v in ("k3", "k4")]
        return fact + dyn
    if name == "moh_block":   # MoMHA-style MoH: route groups of heads (block2: 2 of 4 groups x 2, block4: 1 of 2 x 4)
        return [trial("moh", d, "lb", sd, v) for v in ("block2", "block4", "block2_sut") for sd in (seeds or [0, 1, 2])
                for d in (datasets or ["dyn", "dyn_fast", "dyn_p50", "fact", "mod"])]
    if name == "unified":     # same router for all (10 Nov): SUT MLP router, random init, gates sum to 1, switch 0.01
        return [trial(s, d, "lb", sd, "sut") for sd in (seeds or [0, 1, 2])
                for d in (datasets or ["fact", "dyn", "dyn_fast", "dyn_p50", "mod"]) for s in ("moh", "moa")]
    if name == "heads16":  # 16 heads x 32: MoA and MoH-block2-SUT at k = 4..16 active, MHA/MQA controls at 10..16
        cfgs = [(s, f"{v}{k}") for k in (4, 8, 12, 16, 10, 14) for s, v in (("moa", "e16_k"), ("moh", "b2sut_e16_k"))]
        ctrl = [(s, f"h{k}") for k in (12, 16, 10, 14) for s in ("mha8", "mqa8")]
        return [trial(s, d, "lb" if s in ("moa", "moh") else "none", sd, v) for sd in (seeds or [0, 1, 2])
                for d in (datasets or ["fact"]) for s, v in cfgs + ctrl]
    if name == "scaled_ctrl":  # router-free controls with head (or 2-head block) weights summing to 1, like the
        # unified-router models; mha8 scale025 = grouped MHA-8 (4 blocks x 1/4), already run on fact
        cfgs = [("dense", "sum1"), ("dense", "bsum1"), ("mha8", "sum1"), ("mha8", "scale025"), ("mqa4", "sum1"),
                ("mqa8", "sum1"), ("gqa4", "sum1"), ("gqa8", "sum1")]
        return [trial(s, d, "none", sd, v) for d in (datasets or ["fact", "dyn", "dyn_fast", "mod", "dyn_p50"])
                for sd in (seeds or [0, 1, 2]) for s, v in cfgs]
    if name == "dyn_main":
        return [trial(s, d, "none", sd) for sd in (seeds or [0, 1, 2]) for d in (datasets or DYN_DATASETS)
                for s in DYN_STRUCTURES]
    if name == "dyn_variants":
        return [trial(s, d, "none", sd, v) for sd in (seeds or [0, 1, 2]) for d in (datasets or ["dyn"])
                for s, v in DYN_VARIANTS]
    if name == "attn_baselines":     # router-free attention controls (no balancing needed)
        return [trial(s, d, "none", sd) for sd in (seeds or [0, 1, 2])
                for d in (datasets or ["dyn", "dyn_fast", "fact", "mod"]) for s in ATTN_BASELINES]
    if name == "attn_baselines4":    # 4-head controls matched to the routed models' active width
        return [trial(s, d, "none", sd) for sd in (seeds or [0, 1, 2])
                for d in (datasets or ["dyn", "dyn_fast", "fact", "mod"]) for s in ATTN_BASELINES4]
    if name == "followup_p50":       # 5 Nov feedback: p_stay 0.5; attention structures only (no MoE)
        ctrl = ["dense"] + ATTN_BASELINES + ATTN_BASELINES4
        out = [trial(s, "dyn_p50", "none", sd) for sd in (seeds or [0, 1, 2]) for s in ctrl]
        out += [trial(s, "dyn_p50", "lb", sd, v) for sd in (seeds or [0, 1, 2])
                for s, v in (("moh", "default"), ("moh", "shared2"), ("moa", "default"), ("momha", "default"))]
        return out
    if name == "followup_shared":    # 5 Nov feedback: MoH with always-on (shared) heads on the other datasets
        return [trial("moh", d, "lb", sd, "shared2") for sd in (seeds or [0, 1, 2])
                for d in (datasets or ["dyn", "dyn_fast", "fact", "mod"])]
    if name == "followup_modlong":   # 5 Nov feedback: mod with 10x training and weight decay 1.0
        return [trial("dense", "mod", "none", 0, "long"), trial("moh", "mod", "lb", 0, "long")]
    if name == "dyn_main_lb":        # routed structures only: Dense / Dense-wide have no router (reuse the 'none' runs)
        return [trial(s, d, "lb", sd) for sd in (seeds or [0, 1, 2]) for d in (datasets or DYN_DATASETS)
                for s in DYN_STRUCTURES if s not in ("dense", "dense_wide")]
    if name == "dyn_variants_lb":
        return [trial(s, d, "lb", sd, v) for sd in (seeds or [0, 1, 2]) for d in (datasets or ["dyn"])
                for s, v in DYN_VARIANTS]
    raise ValueError(name)
