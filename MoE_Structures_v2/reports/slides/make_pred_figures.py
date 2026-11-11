"""Prediction-vs-truth figures for dyn_fast (p_stay 0.80) and dyn_p50 (0.50) (question of 11 Nov):
why the curves look identical although the excess MSE differs by a factor ~2 across models.

* error_parts       total test MSE split into noise / unknown regime / model excess, per model
* pred_seq_<ds>     one test sequence: truth + all predictions (they overlap), then prediction − Bayes per model
* pred_cum          extra squared error vs Bayes, summed over the whole test set: small gaps add up

Uses the cached test-set predictions of analysis.dyn_error_decomp (3 seeds; routed models with 'lb').

    python reports/slides/make_pred_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_followup_figures import C, INK, INK2, L, OUT, REG, ROOT, TAB, plt, save  # noqa: E402

CTX = 8
# 4 active heads everywhere: each routed model next to its router-free control with 4 query heads
MODELS = ["dense", "moh", "moh_b2sut", "mqa4", "moa", "gqa4", "momha"]
LAB = {**L, "dense": "MHA-4 (Dense)", "moh_b2sut": "MoH-block2-SUT", "mqa4": "MQA-4", "gqa4": "GQA-4"}
C = {**C, "dense": "#2a78d6", "moh_b2sut": "#9c5b12", "mqa4": "#7a5195", "gqa4": "#5b6770"}
PST = {"dyn": 0.95, "dyn_fast": 0.80, "dyn_p50": 0.50}
OUT = OUT / "att"


def load(ds):
    return dict(np.load(ROOT / "reports" / "cache" / f"dyn_preds_{ds}.npz"))


def sq(z, key):
    return ((z[key][:, CTX:].astype(float) - z["x_next"][:, CTX:]) ** 2).mean(-1)      # N x (T-CTX)


def fig_error_parts():
    ref = pd.read_csv(TAB / "dyn_reference.csv").set_index(["dataset", "reference"]).mse_ctx8
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), sharex=True)
    for ax, ds in zip(axes, ["dyn", "dyn_fast", "dyn_p50"]):
        z = load(ds)
        noise = ref[(ds, "oracle_true")] * 1e3
        bay = sq(z, "bayes").mean() * 1e3
        for i, m in enumerate(MODELS):
            ex = np.mean([sq(z, f"{m}_s{s}").mean() for s in range(3)]) * 1e3 - bay
            ax.barh(i, noise, color="#c9ced8", height=0.7)
            ax.barh(i, bay - noise, left=noise, color="#8fa3c4", height=0.7)
            ax.barh(i, ex, left=bay, color=C[m], height=0.7)
            ax.text(bay + ex + 0.4, i, f"+{ex:.3f}  ({ex / (bay + ex) * 100:.1f} %)", va="center",
                    fontsize=10.5, color=INK)
        ax.text(noise / 2, -0.55, f"noise {noise:.1f}", ha="center", va="bottom", fontsize=10, color=INK)
        ax.text(noise, -1.05, f"regime not known yet {bay - noise:.1f}", ha="left",
                va="bottom", fontsize=10, color="#3c5a8a")
        ax.set_yticks(range(len(MODELS)), [LAB[m] for m in MODELS] if ds == "dyn" else [""] * len(MODELS))
        ax.set_ylim(len(MODELS) - 0.5, -1.6)
        ax.set_title(f"{ds} (p_stay {PST[ds]}): total MSE ×10⁻³", fontsize=13, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 40)
    axes[1].set_xlabel("grey = noise (no model can predict it) · blue = cost of not knowing the regime "
                       "(the best possible predictor pays it too) · colour = model's own excess", fontsize=11.5)
    fig.tight_layout()
    save(fig, "error_parts")


def fig_seq(ds, idx=0, seed=0, t_max=64, zoom=(30, 42)):
    z = load(ds)
    R, y, b = z["regimes"][idx], z["x_next"][idx], z["bayes"][idx]
    t = np.arange(1, t_max + 1)                    # prediction made at step t-1 is for x[t]
    fig = plt.figure(figsize=(18, 12.6))
    gs = fig.add_gridspec(2 + len(MODELS), 2, height_ratios=[2.6, 0.45] + [1] * len(MODELS), width_ratios=[3, 1],
                          hspace=0.25, wspace=0.08)
    ax0 = fig.add_subplot(gs[0, 0]); axz = fig.add_subplot(gs[0, 1])
    sw = np.nonzero(R[1:] != R[:-1])[0] + 1
    for ax, (lo, hi) in ((ax0, (0, t_max)), (axz, zoom)):
        for k in range(lo, hi):
            ax.axvspan(k + 0.5, k + 1.5, color=REG[R[k]], alpha=0.10, lw=0)
        ax.plot(t[lo:hi], y[lo:hi, 0], color=INK, lw=2.6, label="truth")
        for m in MODELS:
            ax.plot(t[lo:hi], z[f"{m}_s{seed}"][idx, lo:hi, 0], lw=1.3, color=C[m], label=LAB[m])
        ax.plot(t[lo:hi], b[lo:hi, 0], color=INK2, lw=1.6, ls="--", label="Bayes (best possible)")
        ax.set_xlim(lo + 0.5, hi + 0.5)
        ax.grid(axis="x", visible=False)
    ax0.axvspan(zoom[0] + 0.5, zoom[1] + 0.5, fill=False, ec=INK, lw=1.2)
    ax0.set_ylabel("$x_1$", fontsize=14, rotation=0, labelpad=12)
    ax0.legend(ncol=9, fontsize=10.5, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False)
    ax0.set_title("truth and all predictions: the model lines lie on top of each other (big misses after a "
                  "switch: Bayes makes them too)", fontsize=12, loc="left", pad=28)
    axz.set_title(f"zoom, steps {zoom[0] + 1}–{zoom[1]}", fontsize=11, loc="left", pad=28)
    axz.set_yticks([])
    dev = {m: z[f"{m}_s{seed}"][idx].astype(float) - b for m in MODELS}
    lim = max(np.abs(d[CTX:t_max, 0]).max() for d in dev.values()) * 1.1
    full = {m: np.mean([((z[f"{m}_s{s}"][:, CTX:].astype(float) - z["bayes"][:, CTX:]) ** 2).mean()
                        for s in range(3)]) * 1e3 for m in MODELS}
    for i, m in enumerate(MODELS):
        ax = fig.add_subplot(gs[2 + i, 0])
        for k in range(t_max):
            ax.axvspan(k + 0.5, k + 1.5, color=REG[R[k]], alpha=0.10, lw=0)
        for s_ in sw:
            ax.axvline(s_ + 0.5, color=INK2, lw=0.7, alpha=0.6)
        ax.axhline(0, color=INK2, lw=0.9, ls="--")
        ax.axvspan(0.5, CTX + 0.5, color="white", alpha=0.7, lw=0)
        ax.fill_between(t, 0, dev[m][:t_max, 0], color=C[m], alpha=0.35, lw=0)
        ax.plot(t, dev[m][:t_max, 0], color=C[m], lw=1.4)
        ax.set_ylim(-lim, lim); ax.set_xlim(0.5, t_max + 0.5)
        ax.set_yticks([-0.05, 0, 0.05] if lim > 0.06 else [-0.02, 0, 0.02])
        ax.tick_params(labelsize=9)
        ax.text(1.002, 0.5, f"{LAB[m]}\ndistance² {full[m]:.3f}", transform=ax.transAxes, fontsize=10.5, va="center",
                color=INK)
        ax.grid(axis="x", visible=False)
        if i < len(MODELS) - 1:
            ax.set_xticklabels([])
        if i == 0:
            ax.set_title("each model's prediction − Bayes prediction ($x_1$; dashed line = Bayes; "
                         "distance² = this gap squared, averaged over the whole test set and 3 seeds, ×10⁻³)",
                         fontsize=12, loc="left")
    ax.set_xlabel(f"time step (background = hidden regime, thin lines = switches, first {CTX} steps not scored; "
                  f"test sequence {idx}, seed {seed})", fontsize=11.5)
    fig.suptitle(f"{ds} (p_stay {PST[ds]})", fontsize=15, fontweight="bold", x=0.01, ha="left", y=0.995)
    save(fig, f"pred_seq_{ds}")


def fig_cum():
    fig, axes = plt.subplots(1, 2, figsize=(18, 5.4))
    for ax, ds in zip(axes, ["dyn_fast", "dyn_p50"]):
        z = load(ds)
        bq = sq(z, "bayes").reshape(-1)
        n = np.arange(1, bq.size + 1)
        for m in MODELS:
            d = np.mean([sq(z, f"{m}_s{s}").reshape(-1) - bq for s in range(3)], 0)
            c = np.cumsum(d)
            ax.plot(n / 1e3, c, color=C[m], lw=2, ls="--" if m in ("dense", "mqa4", "gqa4") else "-", label=f"{LAB[m]}  ({c[-1] / n[-1] * 1e3:.3f})")
        ax.set_title(f"{ds} (p_stay {PST[ds]}): extra squared error vs Bayes, added up over test steps",
                     fontsize=12.5, loc="left")
        ax.set_xlabel("number of test steps added up (thousands; 2000 sequences × 56 scored steps)", fontsize=11)
        ax.legend(fontsize=10.5, loc="upper left", title="dashed = no router  (final slope = excess ×10⁻³)",
                  title_fontsize=10.5)
        ax.set_xlim(0, n[-1] / 1e3)
        ax.set_ylim(0, None)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("summed extra squared error", fontsize=12)
    fig.tight_layout()
    save(fig, "pred_cum")


if __name__ == "__main__":
    fig_error_parts()
    for d in ("dyn_fast", "dyn_p50"):
        fig_seq(d)
    fig_cum()


def fig_dyn_final():
    """Slide 26 (11 Nov): excess over Bayes at matched 4 active heads, one shared y-axis for all three datasets;
    bottom row: change relative to the model's own 4-head control, also on one shared axis."""
    import json
    ref = pd.read_csv(TAB / "dyn_reference.csv").set_index(["dataset", "reference"]).mse_ctx8
    groups = [("dense", [("moh", "default"), ("moh", "block2_sut")]), ("mqa4", [("moa", "default")]),
              ("gqa4", [("momha", "default")])]
    key = {("moh", "default"): "moh", ("moh", "block2_sut"): "moh_b2sut", ("moa", "default"): "moa",
           ("momha", "default"): "momha"}

    def ex(s, v, b, ds):
        out = []
        for sd in range(3):
            p = ROOT / "runs" / f"{ds}__{s}__{v}__{b}__s{sd}" / "result.json"
            if p.exists():
                out.append((json.loads(p.read_text())["test_mse_ctx8"] - ref[(ds, "bayes_lastobs")]) * 1e3)
        return np.array(out)

    fig, axes = plt.subplots(2, 3, figsize=(19, 8.6), sharey="row", gridspec_kw={"height_ratios": [1.5, 1]})
    rows = []
    for j, ds in enumerate(["dyn", "dyn_fast", "dyn_p50"]):
        ax, axr = axes[0, j], axes[1, j]
        x, ticks, labs = 0, [], []
        for ctrl, routed in groups:
            c = ex(ctrl, "default", "none", ds)
            items = [(ctrl, c, True)] + [(key[r], ex(r[0], r[1], "lb", ds), False) for r in routed]
            for m, e, is_c in items:
                ax.bar(x, e.mean(), 0.8, yerr=e.std(), color=C[m], alpha=0.5 if is_c else 1, hatch="//" if is_c else None,
                       edgecolor=C[m], capsize=2.5, error_kw={"lw": .9, "ecolor": INK2})
                ax.scatter([x] * len(e), e, s=9, color=INK, zorder=3)
                ax.text(x, e.mean() + e.std() + 0.004, f"{e.mean():.3f}", ha="center", fontsize=9.5, color=INK)
                if not is_c:
                    d = (e - c[:len(e)]) / c.mean() * 100          # paired by seed
                    axr.bar(x, d.mean(), 0.8, yerr=d.std(), color=C[m], capsize=2.5, error_kw={"lw": .9, "ecolor": INK2})
                    axr.scatter([x] * len(d), d, s=9, color=INK, zorder=3)
                    axr.text(x, 1.2 if d.mean() < 0 else -1.2, f"{d.mean():+.0f} %", ha="center",
                             va="bottom" if d.mean() < 0 else "top", fontsize=10.5, color=INK, fontweight="bold")
                    rows.append(dict(dataset=ds, model=m, control=ctrl, excess=e.mean(), ctrl=c.mean(), rel=d.mean()))
                ticks.append(x); labs.append(LAB[m])
                x += 1
            x += 0.6
        ax.set_title(f"{ds} (p_stay {PST[ds]})\nBayes floor MSE = {ref[(ds, 'bayes_lastobs')] * 1e3:.1f}×10⁻³",
                     fontsize=12.5,
                     loc="left")
        for a in (ax, axr):
            a.set_xticks(ticks, labs if a is axr else [""] * len(labs), rotation=40, ha="right", fontsize=10)
            a.grid(axis="x", visible=False)
        axr.axhline(0, color=INK, lw=1)
    axes[0, 0].set_ylim(0, 0.25)
    axes[0, 0].set_ylabel("MSE − Bayes floor (×10⁻³)\nlower is better", fontsize=12)
    axes[1, 0].set_ylabel("vs own 4-head control\n(below 0 = routing better)", fontsize=12)
    fig.tight_layout()
    save(fig, "dyn_final")
    t = pd.DataFrame(rows); t.to_csv(TAB / "dyn_final.csv", index=False); print(t.round(3).to_string())


def fig_decomp_matched():
    """Slide 22 (11 Nov): routed model minus its own 4-head control, split by steps since the switch; the bars
    of one model add up to its total. Paired by seed, whole test set, one shared y-axis for all datasets."""
    import torch
    from backbone.metrics_dyn import AGE_BINS, regime_age
    pairs = [("moh", "dense", "MoH − MHA-4"), ("moh_b2sut", "dense", "MoH-block2-SUT − MHA-4"),
             ("moa", "mqa4", "MoA − MQA-4"), ("momha", "gqa4", "MoMHA − GQA-4")]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.6), sharey=True)
    rows = []
    for ax, ds in zip(axes, ["dyn", "dyn_fast", "dyn_p50"]):
        z = load(ds)
        age = regime_age(torch.from_numpy(z["regimes"])).numpy()[:, CTX:]
        share = [((age >= lo) & (age <= hi)).mean() for _, lo, hi in AGE_BINS]
        w = 0.2
        for j, (r, c, lab) in enumerate(pairs):
            con = []
            for s in range(3):
                d = sq(z, f"{r}_s{s}") - sq(z, f"{c}_s{s}")
                con.append([d[(age >= lo) & (age <= hi)].sum() / d.size * 1e3 for _, lo, hi in AGE_BINS])
            con = np.array(con)
            tot = con.sum(1).mean()
            ax.bar(np.arange(len(AGE_BINS)) + (j - 1.5) * w, con.mean(0), w * 0.95, yerr=con.std(0), capsize=2,
                   color=C[r], error_kw={"lw": .8, "ecolor": INK2}, label=f"{lab}  (total {tot:+.3f})")
            rows += [dict(dataset=ds, pair=lab, age=a[0], contrib_e3=v) for a, v in zip(AGE_BINS, con.mean(0))]
            rows.append(dict(dataset=ds, pair=lab, age="total", contrib_e3=tot))
        ax.axhline(0, color=INK, lw=1)
        ax.set_xticks(range(len(AGE_BINS)), [f"{a[0]}\n({p * 100:.0f}%)" for a, p in zip(AGE_BINS, share)],
                      fontsize=11)
        ax.set_xlabel("steps since the last regime switch (0 = switch step; % of scored steps)", fontsize=11.5)
        ax.set_title(f"{ds}  (p_stay {PST[ds]})", fontsize=14, loc="left")
        ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("contribution to MSE(routed) − MSE(control)\n(×10⁻³, below 0 = routing better)", fontsize=12)
    fig.tight_layout()
    save(fig, "mse_decomp_matched")
    t = pd.DataFrame(rows); t.to_csv(TAB / "dyn_error_decomp_matched.csv", index=False)
    print(t.pivot_table(index=["dataset", "pair"], columns="age", values="contrib_e3").round(3).to_string())


def fig_dyn_k(ds="dyn"):
    """Slide 32 (11 Nov): excess vs number of active heads (4, 6, 8 of 8) for the four routed models, all with the
    load-balance loss; MHA-4 (= Dense) and MHA-8 as router-free reference lines."""
    import json
    fl = pd.read_csv(TAB / "dyn_reference.csv").set_index(["dataset", "reference"]).mse_ctx8[(ds, "bayes_lastobs")]

    def ex(s, v, b):
        out = []
        for sd in range(3):
            p = ROOT / "runs" / f"{ds}__{s}__{v}__{b}__s{sd}" / "result.json"
            if p.exists():
                out.append((json.loads(p.read_text())["test_mse_ctx8"] - fl) * 1e3)
        return np.array(out)

    lines = [("moh", "moh", ["default", "k6", "k8"]), ("moh_b2sut", "moh", ["block2_sut", "block2_sut_k3", "block2_sut_k4"]),
             ("moa", "moa", ["default", "k6", "k8"]), ("momha", "momha", ["default", "k3", "k4"])]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    off = {"moh": -0.12, "moh_b2sut": -0.04, "moa": 0.04, "momha": 0.12}
    for key, s, vs in lines:
        xs, ms, sds = [], [], []
        for k, v in zip([4, 6, 8], vs):
            e = ex(s, v, "lb")
            if len(e):
                xs.append(k + off[key]); ms.append(e.mean()); sds.append(e.std() if len(e) > 1 else 0)
        ax.errorbar(xs, ms, yerr=sds, fmt="o-", color=C[key], lw=2.2, ms=8, capsize=4, label=LAB[key])
    for s, ls, lab in (("dense", "--", "MHA-4 (Dense), no router"), ("mha8", ":", "MHA-8, no router")):
        ax.axhline(ex(s, "default", "none").mean(), color=INK2 if s == "mha8" else C["dense"], ls=ls, lw=1.8,
                   label=lab)
    ax.set_xticks([4, 6, 8], ["4 of 8\n(default)", "6 of 8", "8 of 8\n(all heads)"], fontsize=12)
    ax.set_ylabel("MSE − Bayes floor (×10⁻³)", fontsize=12)
    ax.set_xlabel("active heads per token (MoMHA, MoH-block2-SUT: 2, 3, 4 of 4 blocks × 2 heads)", fontsize=12)
    ax.legend(fontsize=10.5, ncol=2, loc="upper left")
    ax.set_xlim(3.5, 8.5)
    ax.grid(axis="x", visible=False)
    save(fig, f"dyn_k_{ds}")
