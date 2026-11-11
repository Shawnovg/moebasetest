"""Unified router (10 Nov): every routed model uses MoMHA's router setup (MLP router, random init, gates sum to 1,
switch loss 0.01 only), at 4 active heads, next to a router-free control whose head weights also sum to 1.

Pairs (control | routed):
  MHA-4 x 1/4 (dense/sum1)    | MoH-SUT          4 of 8 single heads, own K/V
  MHA-4 x 1/2 (dense/bsum1)   | MoH-block2-SUT   2 of 4 blocks x 2 heads, own K/V
  MQA-4 x 1/4 (mqa4/sum1)     | MoA-SUT          4 of 8 heads, 1 shared K/V
  GQA-4 x 1/2 (gqa4/sum1)     | MoMHA            2 of 4 blocks x 2 heads, 2 shared K/V
The original (unscaled) control is drawn as a tick on each control bar.

    python -m analysis.unified_compare -> reports/tables/unified_compare.csv + figure
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
DS = ["fact", "mod", "dyn", "dyn_fast", "dyn_p50"]
PAIRS = [  # (family colour key, control, unscaled control, routed)
    ("moh", ("dense", "sum1", "none", "MHA-4 ×¼"), ("dense", "default", "none"), ("moh", "sut", "lb", "MoH-SUT")),
    ("moh", ("dense", "bsum1", "none", "MHA-4 ×½"), ("dense", "default", "none"),
     ("moh", "block2_sut", "lb", "MoH-block2-SUT")),
    ("moa", ("mqa4", "sum1", "none", "MQA-4 ×¼"), ("mqa4", "default", "none"), ("moa", "sut", "lb", "MoA-SUT")),
    ("momha", ("gqa4", "sum1", "none", "GQA-4 ×½"), ("gqa4", "default", "none"), ("momha", "default", "lb", "MoMHA")),
]
OLD = [("moh", "default", "lb", "MoH (old router)"), ("moa", "default", "lb", "MoA (old router)")]


def floors():
    r = pd.read_csv(ROOT / "reports" / "tables" / "dyn_reference.csv")
    return {d: float(r[(r.dataset == d) & (r.reference == "bayes_lastobs")].mse_ctx8.iloc[0])
            for d in ("dyn", "dyn_fast", "dyn_p50")}


def vals(s, v, b, ds, fl):
    out = []
    for sd in range(5):
        p = RUNS / f"{ds}__{s}__{v}__{b}__s{sd}" / "result.json"
        if p.exists():
            j = json.loads(p.read_text())
            out.append((j["test_mse_ctx8"] - fl[ds]) * 1e3 if ds in fl else j["test_acc"])
    return out


def collect():
    fl = floors()
    rows = []
    specs = [(c[:3], c[3], "control") for _, c, _, _ in PAIRS] + [(r[:3], r[3], "routed") for _, _, _, r in PAIRS]
    specs += [(u, f"{c[3].split(' ')[0]} (unscaled)", "unscaled") for _, c, u, _ in PAIRS]
    specs += [(o[:3], o[3], "old") for o in OLD]
    for (s, v, b), lab, kind in specs:
        for ds in DS:
            for i, x in enumerate(vals(s, v, b, ds, fl)):
                rows.append({"label": lab, "kind": kind, "structure": s, "variant": v, "dataset": ds, "seed": i,
                             "value": x})
    return pd.DataFrame(rows).drop_duplicates(["label", "dataset", "seed"])


def main():
    from analysis.figures import INK, INK2, STRUCT_COLOR, _save, plt
    df = collect()
    t = df.groupby(["label", "dataset"]).value.agg(["mean", "std", "count"]).reset_index()
    t.to_csv(ROOT / "reports" / "tables" / "unified_compare.csv", index=False)
    piv = t.pivot(index="label", columns="dataset", values="mean")[DS]
    print(piv.round(3).to_string())
    print(t.pivot(index="label", columns="dataset", values="count")[DS].to_string())
    titles = {"fact": "fact · accuracy ↑", "mod": "mod · accuracy ↑", "dyn": "dyn (0.95) · excess MSE ×1e-3 ↓",
              "dyn_fast": "dyn_fast (0.80) · excess ×1e-3 ↓", "dyn_p50": "dyn_p50 (0.50) · excess ×1e-3 ↓"}
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.6))
    for ax, ds in zip(axes, DS):
        xt, xl = [], []
        for i, (fam, c, u, r) in enumerate(PAIRS):
            col = STRUCT_COLOR[fam]
            for j, (spec, hatch) in enumerate(((c, "///"), (r, None))):
                x = np.array(vals(*spec[:3], ds, floors()))
                pos = i * 2.6 + j
                if len(x):
                    ax.bar(pos, x.mean(), 0.86, yerr=x.std() if len(x) > 1 else 0, color=col,
                           alpha=0.45 if hatch else 1, hatch=hatch, edgecolor=col, capsize=2,
                           error_kw={"lw": .8, "ecolor": INK2})
                    ax.scatter([pos] * len(x), x, s=7, color=INK, zorder=3)
                    ax.text(pos, 1.0, f"{x.mean():.3f}" if ds.startswith("dyn") else f"{x.mean():.2f}",
                            transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color=INK)
                else:
                    ax.text(pos, 0.02, "running", transform=ax.get_xaxis_transform(), rotation=90, ha="center", va="bottom", fontsize=7, color=INK2)
                if j == 0:
                    xu = vals(*u, ds, floors())
                    if xu:
                        ax.hlines(np.mean(xu), pos - .43, pos + .43, color=INK, lw=1.6, ls=":")
                xt.append(pos); xl.append(spec[3])
        ax.set_xticks(xt, xl, rotation=55, ha="right", fontsize=8)
        ax.set_title(titles[ds], fontsize=9.5, loc="left", pad=14)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if not ds.startswith("dyn"):
            ax.set_ylim(0, 1)
        else:
            v = df[(df.dataset == ds) & (df.kind != "old")].value
            ax.set_ylim(v.min() * 0.85, v.max() * 1.05)
    fig.suptitle("Same router for every routed model (MLP, random init, gates sum to 1, switch loss) vs router-free "
                 "controls with head weights summing to 1 · 4 active heads · hatched = no router, dotted tick = "
                 "control without scaling", fontsize=10.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, "unified_compare", ROOT / "reports" / "slides" / "figures" / "att")


if __name__ == "__main__":
    print(main())


def fig_router_ab():
    """Original router vs SUT router, same structure (11 Nov): MoH and MoA, 5 tasks, lb, 3 seeds."""
    from analysis.figures import INK, INK2, STRUCT_COLOR, _save, plt
    fl = floors()
    rows = [("moh", "default", "MoH\nlinear"), ("moh", "sut", "MoH\nSUT"),
            ("moa", "default", "MoA\nlinear 0-init"), ("moa", "sut", "MoA\nSUT")]
    titles = {"fact": "fact · accuracy ↑", "mod": "mod · accuracy ↑", "dyn": "dyn (0.95) · excess ×1e-3 ↓",
              "dyn_fast": "dyn_fast (0.80) · excess ×1e-3 ↓", "dyn_p50": "dyn_p50 (0.50) · excess ×1e-3 ↓"}
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.4))
    out = []
    for ax, ds in zip(axes, DS):
        for i, (s, v, lab) in enumerate(rows):
            x = np.array(vals(s, v, "lb", ds, fl))
            pos = i + (i // 2) * 0.5
            ax.bar(pos, x.mean(), 0.85, yerr=x.std(), color=STRUCT_COLOR[s], alpha=0.45 if v == "default" else 1,
                   hatch="..." if v == "default" else None, edgecolor=STRUCT_COLOR[s], capsize=2.5,
                   error_kw={"lw": .8, "ecolor": INK2})
            ax.scatter([pos] * len(x), x, s=8, color=INK, zorder=3)
            ax.text(pos, 1.0, f"{x.mean():.3f}" if ds.startswith("dyn") else f"{x.mean():.2f}",
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5)
            out.append(dict(dataset=ds, model=lab.replace("\n", " "), mean=x.mean(), sd=x.std()))
        ax.set_xticks([i + (i // 2) * 0.5 for i in range(4)], [r[2] for r in rows], fontsize=8, rotation=0)
        ax.set_title(titles[ds], fontsize=10, loc="left", pad=16)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", visible=False)
        if ds.startswith("dyn"):
            ax.set_ylim(0.07 if ds == "dyn_p50" else 0.14, None)
        else:
            ax.set_ylim(0, 0.7)
    fig.suptitle("Same structure, different router (load-balance loss, 4 active heads, 3 seeds; dotted = original "
                 "router, solid = SUT router: MLP, random init, gates sum to 1, switch loss only)", fontsize=10.5,
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    print(pd.DataFrame(out).round(3).to_string())
    return _save(fig, "router_ab", ROOT / "reports" / "slides" / "figures" / "att")
