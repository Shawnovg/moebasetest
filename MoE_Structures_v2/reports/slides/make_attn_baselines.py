"""Routed attention vs its router-free counterpart (MoH/MHA-8, MoA/MQA-8, MoMHA/GQA-8).

Reads runs/*/result.json for fact / mod and reports/tables/dyn_main{,_lb}.csv for the dynamics.
Writes reports/tables/attn_baselines.csv and reports/slides/figures/attn_baselines.png.

    python reports/slides/make_attn_baselines.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis import figures as F  # noqa: E402

OUT = ROOT / "reports" / "slides" / "figures"
TAB = ROOT / "reports" / "tables"
PAIRS = [("moh", "mha8"), ("moa", "mqa8"), ("momha", "gqa8")]
C, L = F.STRUCT_COLOR, F.STRUCT_LABEL
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                     "savefig.dpi": 200, "font.size": 11})


def token_rows():
    rows = []
    for p in (ROOT / "runs").glob("*/result.json"):
        r = json.loads(p.read_text())
        if r.get("dataset") in ("fact", "mod") and r.get("variant") == "default" and "test_acc" in r:
            rows.append({k: r[k] for k in ("dataset", "structure", "balance", "seed", "test_acc")})
    d = pd.DataFrame(rows)
    return d.groupby(["dataset", "structure", "balance"]).test_acc.agg(["mean", "std", "count"]).reset_index()


def dyn_rows():
    out = []
    for bal, f in (("none", "dyn_main.csv"), ("lb", "dyn_main_lb.csv")):
        d = pd.read_csv(TAB / f)
        d = d[d.variant == "default"]
        for _, r in d.iterrows():
            out.append({"dataset": r.dataset, "structure": r.structure, "balance": bal,
                        "mean": r.excess * 1e3, "std": r.excess_sd * 1e3, "count": r.n_x})
    return pd.DataFrame(out)


def main():
    t = pd.concat([token_rows(), dyn_rows()], ignore_index=True)
    t.to_csv(TAB / "attn_baselines.csv", index=False)
    get = lambda ds, s, b: t[(t.dataset == ds) & (t.structure == s) & (t.balance == b)].iloc[0]
    panels = [("dyn", "dyn: MSE $-$ Bayes floor ($\\times10^{-3}$, lower is better)"),
              ("dyn_fast", "dyn_fast: MSE $-$ Bayes floor ($\\times10^{-3}$, lower is better)"),
              ("fact", "fact: test accuracy (higher is better)"),
              ("mod", "mod: test accuracy (higher is better)")]
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.6))
    for ax, (ds, title) in zip(axes.flat, panels):
        vals = []
        for i, (rs, fs) in enumerate(PAIRS):
            bars = [(get(ds, rs, "none"), C[rs], 0.45, f"{L[rs]}\nno bal."),
                    (get(ds, rs, "lb"), C[rs], 1.0, f"{L[rs]}\nLB loss"),
                    (get(ds, fs, "none"), C[fs], 1.0, L[fs])]
            for j, (r, col, a, _) in enumerate(bars):
                x = i * 3.6 + j
                ax.bar(x, r["mean"], width=0.85, color=col, alpha=a, yerr=r["std"], capsize=3,
                       error_kw={"lw": 1, "ecolor": F.INK2})
                ax.text(x, r["mean"] + (r["std"] if r["std"] == r["std"] else 0), f"{r['mean']:.3f}" if ds.startswith("dyn") else f"{r['mean']:.2f}",
                        ha="center", va="bottom", fontsize=9.5)
                vals.append(r["mean"] + (r["std"] if r["std"] == r["std"] else 0))
        ticks = [i * 3.6 + j for i in range(3) for j in range(3)]
        labs = [b for rs, fs in PAIRS for b in (f"{L[rs]}\nno bal.", f"{L[rs]}\nLB loss", L[fs])]
        ax.set_xticks(ticks, labs, fontsize=10)
        dense = get(ds, "dense", "none")["mean"]
        ax.axhline(dense, color=C["dense"], ls="--", lw=1.6)
        dtxt = f"{dense:.3f}" if ds.startswith("dyn") else f"{dense:.2f}"
        lo, hi = min(vals + [dense]), max(vals + [dense])
        if ds.startswith("dyn"):
            ax.set_ylim(lo * 0.8, hi * 1.07)
        else:
            ax.set_ylim(0, hi * 1.1)
        ax.set_title(title + f"\n- - -  Dense = {dtxt}", fontsize=12)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    p = OUT / "attn_baselines.png"
    fig.savefig(p, bbox_inches="tight", pad_inches=0.08)
    print(p)
    print(t[t.structure.isin(["dense"] + [s for pr in PAIRS for s in pr])].sort_values(["dataset", "structure", "balance"]).to_string())


if __name__ == "__main__":
    main()
