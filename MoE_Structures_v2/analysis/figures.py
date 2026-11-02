"""Result figures (PNG, reports/figures/). Each function takes the tables from analysis.collect and
returns the saved path. Colours: one fixed hue per structure (same in every figure), one per
balancing mode; structures are always named on the axis, so identity never relies on colour.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backbone.paths import FIG_ROOT  # noqa: E402

STRUCT_ORDER = ["dense", "dense_wide", "moe", "moh", "moa", "momha", "mom", "mita"]
STRUCT_LABEL = {"dense": "Dense", "dense_wide": "Dense-wide", "moe": "MoE", "moh": "MoH", "moa": "MoA",
                "momha": "MoMHA", "mom": "MoM", "mita": "MiTA"}
STRUCT_COLOR = dict(zip(STRUCT_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                                       "#6250d6", "#e34948"]))
BAL_ORDER = ["none", "lb", "warmup"]
BAL_LABEL = {"none": "no balancing", "lb": "load-balance loss", "warmup": "random warm-up"}
BAL_COLOR = {"none": "#8a8984", "lb": "#184f95", "warmup": "#86b6ef"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DS_ORDER = ["mqar", "fact", "mod", "copy", "rev", "logic", "mix"]


def style():
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
                         "axes.titleweight": "bold", "legend.frameon": False, "savefig.dpi": 160})


style()


def _save(fig, name, out=FIG_ROOT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def _structs(df):
    return [s for s in STRUCT_ORDER if s in set(df.structure)]


def screen_heatmap(df, name="screen_heatmap", out=FIG_ROOT):
    """Structures x datasets, mean test accuracy (balance none, default variants)."""
    d = df[(df.variant == "default") & (df.balance == "none")]
    ds = [x for x in DS_ORDER if x in set(d.dataset)]
    ss = _structs(d)
    M = d.groupby(["structure", "dataset"]).test_acc.mean().unstack().reindex(index=ss, columns=ds)
    fig, ax = plt.subplots(figsize=(1.1 * len(ds) + 2, 0.45 * len(ss) + 1.2))
    ax.grid(False)
    im = ax.imshow(M.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(ss)):
        for j in range(len(ds)):
            v = M.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.6 else INK)
    ax.set_xticks(range(len(ds)), ds)
    ax.set_yticks(range(len(ss)), [STRUCT_LABEL[s] for s in ss])
    ax.set_title("Dataset screening: test accuracy (default configs, no balancing)")
    fig.colorbar(im, ax=ax, fraction=0.03, label="test accuracy")
    return _save(fig, name, out)


def screen_separation(sep: pd.DataFrame, name="screen_separation", out=FIG_ROOT):
    sep = sep.sort_values("spread", ascending=True)
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(sep) + 1))
    ax.barh(sep.dataset, sep.spread, color="#2a78d6", height=0.6)
    for y, (s, sd) in enumerate(zip(sep.spread, sep.seed_sd)):
        lab = f"{s:.3f}" + (f"   (seed sd {sd:.3f})" if np.isfinite(sd) else "")
        ax.text(s, y, " " + lab, va="center", fontsize=8, color=INK2)
    ax.set_xlabel("spread of structure means (std of test accuracy across structures)")
    ax.set_title("Which dataset separates the structures?")
    ax.set_xlim(0, sep.spread.max() * 1.6)
    return _save(fig, name, out)


def main_bars(df, dataset, variant="default", name=None, out=FIG_ROOT):
    """Test accuracy per structure, one bar per balancing mode, dots = seeds."""
    d = df[(df.dataset == dataset) & (df.variant == variant)]
    ss = _structs(d)
    fig, ax = plt.subplots(figsize=(1.0 * len(ss) + 2, 3.6))
    w = 0.26
    for bi, b in enumerate(BAL_ORDER):
        xs, ms = [], []
        for si, s in enumerate(ss):
            g = d[(d.structure == s) & (d.balance == b)]
            if len(g) == 0:
                continue
            x = si + (bi - 1) * w
            xs.append(x); ms.append(g.test_acc.mean())
            ax.scatter(np.full(len(g), x), g.test_acc, s=10, color=INK, zorder=3, linewidths=0)
        ax.bar(xs, ms, width=w - 0.03, color=BAL_COLOR[b], label=BAL_LABEL[b])
    ax.set_xticks(range(len(ss)), [STRUCT_LABEL[s] for s in ss])
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.13))
    ax.set_title(f"{dataset}: structures x balancing (bars = seed mean, dots = seeds)", pad=24)
    return _save(fig, name or f"main_{dataset}", out)


def balance_delta(deltas: pd.DataFrame, dataset, name=None, out=FIG_ROOT):
    """Seed-paired accuracy change of lb and warmup against no balancing, per structure (+ variant)."""
    d = deltas[deltas.dataset == dataset].copy()
    d["label"] = d.apply(lambda r: STRUCT_LABEL.get(r.structure, r.structure) +
                         ("" if r.variant == "default" else f" / {r.variant}"), axis=1)
    labels = list(dict.fromkeys(d.sort_values(["structure", "variant"]).label))
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(labels) + 1.2))
    for ci, (comp, col) in enumerate([("lb - none", BAL_COLOR["lb"]), ("warmup - none", BAL_COLOR["warmup"])]):
        g = d[d.comparison == comp].set_index("label").reindex(labels)
        y = np.arange(len(labels)) + (ci - 0.5) * 0.36
        ax.barh(y, g["mean"], height=0.34, color=col, label=comp.replace("lb", "load-balance loss").replace("warmup", "random warm-up").replace("none", "none"))
        ax.errorbar(g["mean"], y, xerr=g["std"].fillna(0), fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("change in test accuracy vs no balancing (seed-paired mean ± sd)")
    ax.legend(loc="lower right")
    ax.set_title(f"{dataset}: effect of the two balancing methods")
    return _save(fig, name or f"balance_delta_{dataset}", out)


def variants_bars(df, dataset, structure, name=None, out=FIG_ROOT):
    d = df[(df.dataset == dataset) & (df.structure == structure)]
    vs = ["default"] + sorted(v for v in set(d.variant) if v != "default")
    fig, ax = plt.subplots(figsize=(1.1 * len(vs) + 2, 3.2))
    w = 0.26
    for bi, b in enumerate(BAL_ORDER):
        for vi, v in enumerate(vs):
            g = d[(d.variant == v) & (d.balance == b)]
            if len(g) == 0:
                continue
            x = vi + (bi - 1) * w
            ax.bar(x, g.test_acc.mean(), width=w - 0.03, color=BAL_COLOR[b], label=BAL_LABEL[b] if vi == 0 else None)
            ax.scatter(np.full(len(g), x), g.test_acc, s=10, color=INK, zorder=3, linewidths=0)
    ax.set_xticks(range(len(vs)), vs)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("test accuracy")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.15))
    ax.set_title(f"{STRUCT_LABEL[structure]} variants on {dataset}", pad=24)
    return _save(fig, name or f"variants_{dataset}_{structure}", out)


def curves(hist, dataset, balance="none", variant="default", name=None, out=FIG_ROOT):
    d = hist[(hist.dataset == dataset) & (hist.balance == balance) & (hist.variant == variant)]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for s in _structs(d):
        g = d[d.structure == s].groupby("step").valid_acc.agg(["mean", "min", "max"]).reset_index()
        ax.plot(g.step, g["mean"], color=STRUCT_COLOR[s], lw=2, label=STRUCT_LABEL[s])
        ax.fill_between(g.step, g["min"], g["max"], color=STRUCT_COLOR[s], alpha=0.12, lw=0)
    ax.set_xlabel("training step"); ax.set_ylabel("validation accuracy"); ax.set_ylim(0, 1.02)
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, 1.2))
    ax.set_title(f"{dataset}, {BAL_LABEL[balance]}: learning curves (band = seed range)", pad=30)
    return _save(fig, name or f"curves_{dataset}_{balance}", out)


def routing_load(rdf, dataset, structure, variant="default", seed=0, name=None, out=FIG_ROOT):
    """Unit load per routed module, one panel row per balancing mode: does routing collapse?"""
    d = rdf[(rdf.dataset == dataset) & (rdf.structure == structure) & (rdf.variant == variant) & (rdf.seed == seed)]
    mods = list(dict.fromkeys(d.module))
    bals = [b for b in BAL_ORDER if b in set(d.balance)]
    if not mods or not bals:
        return None
    fig, axes = plt.subplots(len(bals), len(mods), figsize=(2.3 * len(mods) + 0.5, 1.6 * len(bals) + 0.6),
                             squeeze=False, sharey=True)
    for i, b in enumerate(bals):
        for j, m in enumerate(mods):
            ax = axes[i][j]
            r = d[(d.balance == b) & (d.module == m)]
            if len(r):
                load = r.iloc[0]["load"]
                ax.bar(range(len(load)), load, color=STRUCT_COLOR.get(structure, "#2a78d6"), width=0.8)
                ax.axhline(1 / len(load), color=INK2, lw=0.8, ls="--")
                ax.text(0.98, 1.0, f"H={max(0.0, r.iloc[0]['entropy']):.2f}", transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=7, color=INK2)
                labels = [str(u) for u in range(len(load))]
                if structure == "mom" and variant != "noskip":
                    labels[-1] = "skip"          # MoM: the last router class is SKIP
                ax.set_xticks(range(len(load)), labels, fontsize=6)
            if i == 0:
                ax.set_title(m.replace("blocks.", "L").replace("body.", ""), fontsize=8, fontweight="normal", pad=12)
            if j == 0:
                ax.set_ylabel(BAL_LABEL[b], fontsize=8)
    fig.suptitle(f"{STRUCT_LABEL[structure]} ({variant}) on {dataset}: share of picks per unit "
                 f"(dashed = even share, H = normalised entropy)", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return _save(fig, name or f"routing_{dataset}_{structure}_{variant}", out)


def collapse_overview(ct, dataset, name=None, out=FIG_ROOT):
    d = ct[(ct.dataset == dataset) & (ct.variant == "default")]
    ss = _structs(d)
    fig, ax = plt.subplots(figsize=(1.0 * len(ss) + 2, 3.0))
    w = 0.26
    for bi, b in enumerate(BAL_ORDER):
        g = d[d.balance == b].set_index("structure").reindex(ss)
        ax.bar(np.arange(len(ss)) + (bi - 1) * w, g.min_entropy, width=w - 0.03, color=BAL_COLOR[b], label=BAL_LABEL[b])
    ax.set_xticks(range(len(ss)), [STRUCT_LABEL[s] for s in ss])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("lowest routing entropy\n(over modules; 0 = collapsed)")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.18))
    ax.set_title(f"{dataset}: routing collapse by balancing mode", pad=26)
    return _save(fig, name or f"collapse_{dataset}", out)


def task_unit_heatmap(row, name, out=FIG_ROOT):
    from .routing import mutual_information, task_unit_matrix
    tasks, M = task_unit_matrix(row)
    P = M / M.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(0.4 * M.shape[1] + 2, 0.4 * len(tasks) + 1.2))
    ax.grid(False)
    ax.imshow(P, cmap="Blues", vmin=0, vmax=max(P.max(), 1e-9), aspect="auto")
    ax.set_yticks(range(len(tasks)), tasks); ax.set_xticks(range(M.shape[1]))
    ax.set_xlabel("unit")
    ax.set_title(f"{STRUCT_LABEL.get(row['structure'])} {row['module']}: unit share per task "
                 f"(I(task;unit)/H(task) = {mutual_information(M):.2f})", fontsize=9)
    return _save(fig, name, out)
