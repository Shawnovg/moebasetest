"""MoH head-block variants vs MoMHA (question of 9 Nov): is MoMHA's gap to MoH due to shared K/V or to the router?

Rows (all with the load-balance loss; Dense and MHA-8 without a router):
  MoH            4 of 8 single heads, linear router, gates x4, MoH balance loss
  MoH-block2     2 of 4 blocks x 2 heads, linear router, gates x2, MoH loss
  MoH-block2-SUT 2 of 4 blocks x 2 heads, MLP router, gates sum to 1, switch loss  (= MoMHA but own K/V per head)
  MoMHA          2 of 4 blocks x 2 heads, MLP router, gates sum to 1, switch loss, 2 K/V heads shared by all
  MoH-block4     1 of 2 blocks x 4 heads, linear router
  MoMHA top-1    1 of 4 blocks x 4 heads (16 heads in total)

    python -m analysis.moh_block_compare -> reports/tables/moh_block_compare.csv, figure
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
DS = ["dyn", "dyn_fast", "dyn_p50", "fact", "mod"]
ROWS = [("moh", "default", "lb", "MoH · 4 of 8 heads"),
        ("moh", "block2", "lb", "MoH-block2 · 2 of 4 blocks × 2"),
        ("moh", "block2_sut", "lb", "MoH-block2-SUT · own K/V"),
        ("momha", "default", "lb", "MoMHA · shared K/V"),
        ("moh", "block4", "lb", "MoH-block4 · 1 of 2 blocks × 4"),
        ("momha", "top1", "lb", "MoMHA top-1 · 1 of 4 blocks × 4")]
REFS = [("dense", "default", "none", "Dense (MHA-4)"), ("mha8", "default", "none", "MHA-8")]


def floors():
    r = pd.read_csv(ROOT / "reports" / "tables" / "dyn_reference.csv")
    return {d: float(r[(r.dataset == d) & (r.reference == "bayes_lastobs")].mse_ctx8.iloc[0]) for d in DS[:3]}


def collect():
    fl = floors()
    out = []
    for s, v, b, lab in ROWS + REFS:
        for ds in DS:
            for sd in range(5):
                p = RUNS / f"{ds}__{s}__{v}__{b}__s{sd}" / "result.json"
                if not p.exists():
                    continue
                j = json.loads(p.read_text())
                val = (j["test_mse_ctx8"] - fl[ds]) * 1e3 if ds in fl else j["test_acc"]
                out.append({"label": lab, "structure": s, "variant": v, "dataset": ds, "seed": sd, "value": val})
    return pd.DataFrame(out)


def main():
    from analysis.figures import INK, INK2, STRUCT_COLOR, _save, plt
    df = collect()
    t = df.groupby(["label", "dataset"]).value.agg(["mean", "std", "count"]).reset_index()
    t.to_csv(ROOT / "reports" / "tables" / "moh_block_compare.csv", index=False)
    print(t.pivot(index="label", columns="dataset", values="mean").round(3).to_string())
    print(df[df.dataset == "fact"].pivot_table(index="label", columns="seed", values="value").round(3))
    labs = [r[3] for r in ROWS]
    col = {r[3]: STRUCT_COLOR[r[0]] for r in ROWS}
    alpha = {r[3]: 1.0 if r[1] == "default" else 0.6 for r in ROWS}
    titles = {"dyn": "dyn (stay 0.95)\nMSE − floor ×1e-3 · lower better", "dyn_fast": "dyn_fast (0.80)\nMSE − floor ×1e-3",
              "dyn_p50": "dyn_p50 (0.50)\nMSE − floor ×1e-3", "fact": "fact\ntest accuracy · higher better",
              "mod": "mod\ntest accuracy"}
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.9), sharey=True)
    y = np.array([0, 1, 2, 3, 4.6, 5.6])
    for ax, ds in zip(axes, DS):
        for yy, lab in zip(y, labs):
            x = df[(df.label == lab) & (df.dataset == ds)].value
            if not len(x):
                continue
            ax.barh(yy, x.mean(), xerr=x.std() if len(x) > 1 else 0, height=0.72, color=col[lab], alpha=alpha[lab],
                    capsize=2.5, error_kw={"lw": .9, "ecolor": INK2})
            ax.scatter(x, [yy] * len(x), s=9, color=INK, zorder=3)
            ax.text(1.0, yy, f"{x.mean():.3f}" if ds.startswith("dyn") else f"{x.mean():.2f}",
                    transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9, color=INK)
        for (s, v, b, lab), ls in zip(REFS, ("--", ":")):
            x = df[(df.label == lab) & (df.dataset == ds)].value
            if len(x):
                ax.axvline(x.mean(), color=STRUCT_COLOR[s], ls=ls, lw=1.6, label=lab)
        ax.set_title(titles[ds], fontsize=10, loc="left")
        ax.grid(axis="y", visible=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if ds.startswith("dyn"):
            lo = df[df.dataset == ds].value
            ax.set_xlim(max(0, lo.min() - 0.03), lo.max() * 1.12)
        else:
            ax.set_xlim(0, 1.0)
    axes[0].set_yticks(y, labs, fontsize=10)
    axes[0].invert_yaxis()
    axes[-1].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("MoH with head blocks vs MoMHA (load-balance loss; bar = mean ± sd, dots = seeds; "
                 "dashed = Dense, dotted = MHA-8)", fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, "moh_block_compare", ROOT / "reports" / "slides" / "figures" / "att")


if __name__ == "__main__":
    print(main())
