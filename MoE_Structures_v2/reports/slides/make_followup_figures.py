"""Figures for the follow-up after the 5 Nov supervisor meeting (white slide style).

* pred_compare_<ds>.png   one test sequence: true state vs 3-seed mean prediction of a normal Transformer
                          (Dense = MHA-4), MHA-8 and MoH, plus per-step squared error incl. the Bayes predictor
* pred_<ds>_<model>.png   the full 4-coordinate prediction figure (3 seeds) for each model
* mod_diag.png            mod: accuracy on held-out (a, b) pairs split by operation and by whether (b, a) was trained
* attn_dyn3.png           excess over the Bayes floor on dyn / dyn_fast / dyn_p50, attention structures only
* shared_heads.png        MoH with 2 always-on heads + 2-of-6 routed vs plain MoH and the controls

    python reports/slides/make_followup_figures.py [pred|mod|dyn3|shared ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis import figures as F  # noqa: E402
from analysis import dyn_predictions as P  # noqa: E402

OUT = ROOT / "reports" / "slides" / "figures"
TAB = ROOT / "reports" / "tables"
SCR = Path("/tmp/claude-0/-home-claude/e857238f-5029-57e7-91a7-11f63f106500/scratchpad")
C, L = dict(F.STRUCT_COLOR), dict(F.STRUCT_LABEL)
C["moh_shared2"], L["moh_shared2"] = "#7a4a00", "MoH + 2 shared"
L["dense"] = "Dense (MHA-4)"
INK, INK2 = F.INK, F.INK2
REG = ["#2a78d6", "#e8743b", "#19a979"]
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                     "savefig.dpi": 200, "font.size": 12})


def save(fig, name):
    p = OUT / f"{name}.png"
    fig.savefig(p, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(p)


# ------------------------------------------------------------------ predictions
MODELS = [("dense", "default", "none"), ("mha8", "default", "none"), ("moh", "default", "lb")]


FAMILIES = {"moh": [("dense", "default", "none"), ("mha8", "default", "none"), ("moh", "default", "lb")],
            "moa": [("mqa4", "default", "none"), ("mqa8", "default", "none"), ("moa", "default", "lb")],
            "momha": [("gqa4", "default", "none"), ("gqa8", "default", "none"), ("momha", "default", "lb")]}


def fam_colors(fam):
    """Distinct colours inside one family: 4-head control blue, 8-head control grey, routed model its own colour."""
    m = [x[0] for x in FAMILIES[fam]]
    return dict(C) if fam == "moh" else {**C, m[0]: "#2a78d6", m[1]: "#6b6b6b"}


def fig_pred_compare(ds="dyn", idx=0, t_max=64, fam="moh"):
    MODELS = FAMILIES[fam]
    C = fam_colors(fam)
    data = P.DynamicsDataset(ds, "test")
    X, R = data.raw[idx].numpy(), data.regimes[idx].numpy()
    bay = P.bayes(data, idx)
    preds = {s: np.mean([P.predict(P.load_model(s, ds, sd, v, b), data, idx) for sd in (0, 1, 2)], 0)
             for s, v, b in MODELS}
    fig, axes = plt.subplots(3, 1, figsize=(17, 9.4), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.3, 1.5]})
    tn = np.arange(1, t_max + 1)
    sw = np.nonzero(R[1:t_max] != R[:t_max - 1])[0] + 1
    for ax in axes:
        for t in range(t_max):
            ax.axvspan(t, t + 1, color=REG[R[t]], alpha=0.09, lw=0)
        for t in sw:
            ax.axvline(t + 1, color=INK2, lw=0.8, alpha=0.6)
    for ax, c in zip(axes[:2], (0, 2)):
        ax.plot(np.arange(t_max + 1), X[:t_max + 1, c], color=INK, lw=2.4, label="true state")
        for s, _, _ in MODELS:
            ax.plot(tn, preds[s][:t_max, c], color=C[s], lw=1.7, ls="--", label=f"{L[s]} prediction")
        ax.set_ylabel(f"$x_{c + 1}$", fontsize=15, rotation=0, labelpad=16)
        ax.grid(axis="x", visible=False)
    axes[0].legend(ncol=4, fontsize=12, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False)
    ax = axes[2]
    for s, _, _ in MODELS:
        e = ((preds[s][:t_max] - X[1:t_max + 1]) ** 2).mean(-1)
        ax.plot(tn, e, color=C[s], lw=1.8, label=f"{L[s]}  (mean {e[7:].mean():.4f})")
    e = ((bay[:t_max] - X[1:t_max + 1]) ** 2).mean(-1)
    ax.plot(tn, e, color=INK, lw=1.4, ls=":", label=f"Bayes predictor  (mean {e[7:].mean():.4f})")
    ax.set_yscale("log"); ax.set_ylabel("squared error", fontsize=13)
    ax.legend(ncol=2, fontsize=12, loc="lower right")
    ax.set_xlim(0, t_max)
    ax.set_xlabel("time step  (background = hidden regime;  vertical line = first state after a switch;  "
                  "predictions = mean of 3 seeds)", fontsize=12)
    fig.tight_layout()
    save(fig, f"pred_compare_{ds}" + ("" if fam == "moh" else f"_{fam}"))


def fig_pred_each(ds="dyn"):
    for s, v, b in [("dense", "default", "none"), ("mha8", "default", "none"), ("moh", "default", "lb"),
                    ("moa", "default", "lb"), ("momha", "default", "lb")]:
        p = P.fig_prediction(s, ds=ds, balance=b, variant=v, out=OUT)
        print(p)


# ------------------------------------------------------------------ mod diagnosis
def fig_mod():
    """Held-out mod accuracy split by whether the swapped pair was trained: 3k steps vs 30k steps + wd 1.0."""
    d3 = json.loads((TAB / "mod_swap_3k.json").read_text())
    dl = json.loads((TAB / "mod_swap_long.json").read_text())
    groups = [("+", "swap in train"), ("+", "swap not in train"), ("*", "swap in train"),
              ("*", "swap not in train"), ("-", "swap in train"), ("-", "swap not in train")]
    lab = {"+": "a + b", "*": "a × b", "-": "a − b"}
    panels = [(d3, [("mod__dense__default__none__s0", "dense"), ("mod__moh__default__lb__s0", "moh")],
               "3,000 steps, weight decay 0.01 (main setting)"),
              (dl, [("mod__dense__long__none__s0", "dense"), ("mod__moh__long__lb__s0", "moh")],
               "30,000 steps, weight decay 1.0")]
    fig, axes = plt.subplots(1, 2, figsize=(20, 5.8), sharey=True)
    rows = []
    w = 0.38
    for ax, (d, runs, title) in zip(axes, panels):
        for j, (run, s) in enumerate(runs):
            vals = [d[run][f"{o} | {k}"][0] for o, k in groups]
            rows += [dict(setting=title, run=run, op=o, swap=k, acc=v) for (o, k), v in zip(groups, vals)]
            xs = np.arange(len(groups)) + (j - 0.5) * w
            ax.bar(xs, vals, width=w * 0.95, color=C[s], label="Dense (MHA-4)" if s == "dense" else "MoH (LB loss)")
            for x, v in zip(xs, vals):
                ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=10.5)
        ax.axhline(1 / 97, color=INK2, ls=":", lw=1.3)
        ax.text(-0.45, 1 / 97 + 0.025, "chance 1/97", ha="left", fontsize=10.5, color=INK2)
        ax.set_xticks(np.arange(len(groups)),
                      [f"{lab[o]}\n(b, a) {'trained' if k == 'swap in train' else 'not trained'}" for o, k in groups],
                      fontsize=10.5)
        ax.set_title(title, fontsize=14)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("accuracy on unseen (a, b) pairs", fontsize=13)
    axes[0].legend(fontsize=12, loc="upper right")
    fig.tight_layout()
    save(fig, "mod_diag")
    pd.DataFrame(rows).to_csv(TAB / "followup_mod_swap.csv", index=False)


# ------------------------------------------------------------------ dynamics: three switching rates
DS3 = [("dyn", "p_stay 0.95"), ("dyn_fast", "p_stay 0.80"), ("dyn_p50", "p_stay 0.50")]
ORDER = [("dense", "none"), ("mha8", "none"), ("moh", "lb"), ("mqa4", "none"), ("mqa8", "none"), ("moa", "lb"),
         ("gqa4", "none"), ("gqa8", "none"), ("momha", "lb")]


def dyn_table():
    rows = []
    for p in (ROOT / "runs").glob("dyn*__*/result.json"):
        r = json.loads(p.read_text())
        rows.append(dict(dataset=r["dataset"], structure=r["structure"], variant=r["variant"], balance=r["balance"],
                         seed=r["seed"], mse=r["test_mse_ctx8"]))
    df = pd.DataFrame(rows)
    ref = pd.read_csv(TAB / "dyn_reference.csv")
    return df, ref


def floor(ref, ds):
    r = ref[(ref.dataset == ds) & (ref.reference == "bayes_lastobs")]
    return float(r.mse_ctx8.iloc[0])


def fig_dyn3():
    df, ref = dyn_table()
    out = []
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
    for ax, (ds, lab) in zip(axes, DS3):
        fl = floor(ref, ds)
        for i, (s, b) in enumerate(ORDER):
            sub = df[(df.dataset == ds) & (df.structure == s) & (df.variant == "default") & (df.balance == b)]
            ex = (sub.mse - fl) * 1e3
            m, sd = ex.mean(), ex.std()
            out.append(dict(dataset=ds, structure=s, balance=b, n=len(sub), floor=fl, excess_e3=m, sd_e3=sd))
            x = i + (i // 3) * 0.6
            ax.bar(x, m, color=C[s], width=0.82, yerr=sd, capsize=3, error_kw={"lw": 1, "ecolor": INK2})
            ax.text(x, m + (sd if sd == sd else 0), f"{m:.3f}", ha="center", va="bottom", fontsize=10)
        xt = [i + (i // 3) * 0.6 for i in range(len(ORDER))]
        ax.set_xticks(xt, [("MHA-4\n(Dense)" if s == "dense" else L[s]) for s, _ in ORDER], fontsize=9.5)
        ax.set_title(f"{ds}  ({lab}),  Bayes floor {fl:.4f}", fontsize=13)
        ax.grid(axis="x", visible=False)
        lo = min(o["excess_e3"] for o in out if o["dataset"] == ds)
        hi = max(o["excess_e3"] + (o["sd_e3"] if o["sd_e3"] == o["sd_e3"] else 0) for o in out if o["dataset"] == ds)
        ax.set_ylim(lo * 0.75, hi * 1.06)
    axes[0].set_ylabel(r"MSE $-$ Bayes floor ($\times10^{-3}$, lower is better)", fontsize=12)
    fig.tight_layout()
    save(fig, "attn_dyn3")
    t = pd.DataFrame(out); t.to_csv(TAB / "followup_dyn3.csv", index=False)
    print(t.to_string())


def fig_shared():
    df, ref = dyn_table()
    tok = []
    for p in (ROOT / "runs").glob("*__moh__*/result.json"):
        r = json.loads(p.read_text())
        if r["dataset"] in ("fact", "mod") and r["variant"] in ("default", "shared2") and r["balance"] == "lb":
            tok.append(dict(dataset=r["dataset"], variant=r["variant"], seed=r["seed"], acc=r["test_acc"]))
    for s in ("dense", "mha8"):
        for p in (ROOT / "runs").glob(f"*__{s}__default__none__*/result.json"):
            r = json.loads(p.read_text())
            if r["dataset"] in ("fact", "mod"):
                tok.append(dict(dataset=r["dataset"], variant=s, seed=r["seed"], acc=r["test_acc"]))
    tok = pd.DataFrame(tok)
    bars = [("dense", "Dense\n(MHA-4)", C["dense"]), ("mha8", "MHA-8", C["mha8"]),
            ("default", "MoH\n(4 of 8)", C["moh"]), ("shared2", "MoH\n(2 shared\n+ 2 of 6)", C["moh"])]
    fig, axes = plt.subplots(1, 5, figsize=(21, 5.6))
    out = []
    for ax, ds in zip(axes, ["dyn", "dyn_fast", "dyn_p50", "fact", "mod"]):
        for i, (key, lab, col) in enumerate(bars):
            if ds.startswith("dyn"):
                if key in ("dense", "mha8"):
                    sub = df[(df.dataset == ds) & (df.structure == key) & (df.variant == "default")]
                else:
                    sub = df[(df.dataset == ds) & (df.structure == "moh") & (df.variant == key) & (df.balance == "lb")]
                v = (sub.mse - floor(ref, ds)) * 1e3
            else:
                v = tok[(tok.dataset == ds) & (tok.variant == key)].acc
            m, sd = v.mean(), v.std()
            out.append(dict(dataset=ds, model=lab.replace("\n", " "), n=len(v), mean=m, sd=sd))
            ax.bar(i, m, color=col, width=0.8, yerr=sd, capsize=3, error_kw={"lw": 1, "ecolor": INK2},
                   hatch="//" if key == "shared2" else None, edgecolor="white" if key == "shared2" else None)
            ax.text(i, m + (sd if sd == sd else 0), f"{m:.3f}" if ds.startswith("dyn") else f"{m:.2f}",
                    ha="center", va="bottom", fontsize=10)
        ax.set_xticks(range(len(bars)), [b[1] for b in bars], fontsize=9.5, rotation=0)
        ax.set_title(ds + ("  (excess ×1e-3, lower better)" if ds.startswith("dyn") else "  (accuracy, higher better)"),
                     fontsize=12)
        vals = [o["mean"] for o in out if o["dataset"] == ds]
        if ds.startswith("dyn"):
            ax.set_ylim(min(vals) * 0.75, max(vals) * 1.15)
        else:
            ax.set_ylim(0, min(1.05, max(vals) * 1.2))
        ax.grid(axis="x", visible=False)
        for lbl in ax.get_xticklabels():
            lbl.set_fontsize(10.5)
    fig.tight_layout()
    save(fig, "shared_heads")
    t = pd.DataFrame(out); t.to_csv(TAB / "followup_shared.csv", index=False)
    print(t.to_string())



# ------------------------------------------------------------------ where the MSE difference comes from (9 Nov)
DEC_MODELS = ["mha8", "moh", "moa", "momha"]
AGES = ["0", "1", "2", "3", "4-7", "8+"]


def fig_dev_seq(ds="dyn", idx=0, t_max=64, seed=0, fam="moh"):
    """Prediction minus the Bayes prediction on one sequence: the part of each model's output that differs."""
    z = np.load(ROOT / "reports" / "cache" / f"dyn_preds_{ds}.npz")
    R = z["regimes"][idx]; b = z["bayes"][idx]
    models = [m[0] for m in FAMILIES[fam]]
    C = fam_colors(fam)
    fig, axes = plt.subplots(2, 1, figsize=(17, 7.6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    tn = np.arange(t_max)
    sw = np.nonzero(R[1:t_max] != R[:t_max - 1])[0] + 1
    for ax in axes:
        for t in range(t_max):
            ax.axvspan(t - 0.5, t + 0.5, color=REG[R[t]], alpha=0.09, lw=0)
        for t in sw:
            ax.axvline(t - 0.5, color=INK2, lw=0.8, alpha=0.6)
        ax.axhline(0, color=INK, lw=1.0)
        ax.axvspan(-0.5, 7.5, color="white", alpha=0.6, lw=0)
    for s in models:
        d = z[f"{s}_s{seed}"][idx].astype(float) - b
        axes[0].plot(tn[1:], d[1:t_max, 0], "o-", ms=3.5, lw=1.6, color=C[s], label=L[s])
        r = np.sqrt((d ** 2).mean(-1))
        axes[1].plot(tn[1:], r[1:t_max], "o-", ms=3.5, lw=1.6, color=C[s],
                     label=f"{L[s]}  (mean square over t ≥ 8: {np.mean(r[8:] ** 2) * 1e3:.3f}×10⁻³)")
    axes[0].set_ylabel("prediction − Bayes\n(coordinate $x_1$)", fontsize=13)
    axes[1].set_ylabel("distance to Bayes\n(RMS over $x_1..x_4$)", fontsize=13)
    axes[0].legend(ncol=3, fontsize=12, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False)
    axes[1].legend(fontsize=11.5, loc="upper right")
    axes[1].set_ylim(0, None)
    axes[1].set_xlim(-0.5, t_max - 0.5)
    axes[1].set_xlabel(f"time step  (background = hidden regime; vertical line = switch; seed {seed}; "
                       f"t = 0 left out; grey = first 8 steps, not scored; noise sd = 0.08 for comparison)", fontsize=12)
    for ax in axes:
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, f"dev_bayes_{ds}" + ("" if fam == "moh" else f"_{fam}"))


def fig_decomp():
    """Contribution of each regime-age bin to (model MSE - Dense MSE); the bars of one model add up to its total."""
    d = pd.read_csv(TAB / "dyn_error_decomp.csv")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.6), sharey=False)
    for ax, ds in zip(axes, ["dyn", "dyn_fast", "dyn_p50"]):
        x = d[d.dataset == ds]
        share = x[x.structure == "dense"][[f"share_{a}" for a in AGES]].mean().values
        w = 0.19
        for j, s in enumerate(DEC_MODELS):
            q = x[x.structure == s]
            m = q[[f"diff_{a}" for a in AGES]].mean().values * 1e3
            sd = q[[f"diff_{a}" for a in AGES]].std().values * 1e3
            tot = q.diff_vs_dense.mean() * 1e3
            ax.bar(np.arange(len(AGES)) + (j - 1.5) * w, m, width=w * 0.95, yerr=sd, capsize=2.5, color=C[s],
                   error_kw={"lw": 0.9, "ecolor": INK2}, label=f"{L[s]}  (total {tot:+.3f})")
        ax.axhline(0, color=INK, lw=1)
        ax.set_xticks(range(len(AGES)), [f"{a}\n({p * 100:.0f}%)" for a, p in zip(AGES, share)], fontsize=11)
        ax.set_xlabel("steps since the last regime switch (0 = the switch step; % of scored steps)", fontsize=12)
        ax.set_title(f"{ds}  (p_stay {dict(dyn=0.95, dyn_fast=0.80, dyn_p50=0.50)[ds]})", fontsize=14)
        ax.legend(fontsize=10.5, loc="lower left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("contribution to MSE − MSE(Dense)  (×10⁻³)\nbelow 0 = better than Dense",
                       fontsize=12)
    fig.tight_layout()
    save(fig, "mse_decomp_age")


def fig_dev_age():
    """RMS distance to the Bayes prediction by regime age; its square, averaged, is the excess."""
    d = pd.read_csv(TAB / "dyn_pred_distance.csv")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.2))
    for ax, ds in zip(axes, ["dyn", "dyn_fast", "dyn_p50"]):
        x = d[(d.dataset == ds) & (d.age != "all")]
        for s in ["dense", "mha8", "moh", "moa", "momha"]:
            q = x[(x.a == s) & (x.b == "bayes")].set_index("age").loc[AGES]
            ax.plot(range(len(AGES)), q.rms, "o-", color=C[s], lw=2, ms=6, label=L[s])
        q = x[(x.a == "dense") & (x.b == "(other seed)")].set_index("age").loc[AGES]
        ax.plot(range(len(AGES)), q.rms, ":", color=INK2, lw=1.6, label="Dense seed vs Dense seed")
        ax.set_xticks(range(len(AGES)), AGES)
        ax.set_xlabel("steps since the last regime switch", fontsize=12)
        ax.set_title(ds, fontsize=14)
        ax.set_ylim(0, None)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("RMS distance between predictions", fontsize=12)
    axes[0].legend(fontsize=10.5)
    fig.tight_layout()
    save(fig, "dev_bayes_age")


def fig_decomp_family():
    """Each routed model minus its own 4-head router-free control (MoH-MHA-4, MoA-MQA-4, MoMHA-GQA-4), by age."""
    d = pd.read_csv(TAB / "dyn_error_decomp.csv").set_index(["dataset", "structure", "seed"])
    pairs = [("moh", "dense", "MoH − MHA-4 (Dense)"), ("moa", "mqa4", "MoA − MQA-4"),
             ("momha", "gqa4", "MoMHA − GQA-4")]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.6))
    rows = []
    for ax, ds in zip(axes, ["dyn", "dyn_fast", "dyn_p50"]):
        share = d.loc[(ds, "dense")][[f"share_{a}" for a in AGES]].mean().values
        w = 0.26
        for j, (r, c, lab) in enumerate(pairs):
            cols = [f"diff_{a}" for a in AGES]
            q = (d.loc[(ds, r)][cols] - d.loc[(ds, c)][cols]) * 1e3          # paired by seed
            tot = (d.loc[(ds, r)].diff_vs_dense - d.loc[(ds, c)].diff_vs_dense).mean() * 1e3
            ax.bar(np.arange(len(AGES)) + (j - 1) * w, q.mean().values, width=w * 0.95, yerr=q.std().values,
                   capsize=2.5, color=C[r], error_kw={"lw": 0.9, "ecolor": INK2}, label=f"{lab}  (total {tot:+.3f})")
            rows += [dict(dataset=ds, pair=lab, age=a, contrib_e3=v) for a, v in zip(AGES, q.mean().values)]
            rows.append(dict(dataset=ds, pair=lab, age="total", contrib_e3=tot))
        ax.axhline(0, color=INK, lw=1)
        ax.set_xticks(range(len(AGES)), [f"{a}\n({p * 100:.0f}%)" for a, p in zip(AGES, share)], fontsize=11)
        ax.set_xlabel("steps since the last regime switch (0 = the switch step; % of scored steps)", fontsize=12)
        ax.set_title(f"{ds}  (p_stay {dict(dyn=0.95, dyn_fast=0.80, dyn_p50=0.50)[ds]})", fontsize=14)
        ax.legend(fontsize=10.5, loc="lower left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("contribution to MSE(routed) − MSE(its 4-head control)\n(×10⁻³, below 0 = routing better)",
                       fontsize=12)
    fig.tight_layout()
    save(fig, "mse_decomp_family")
    pd.DataFrame(rows).to_csv(TAB / "dyn_error_decomp_family.csv", index=False)


if __name__ == "__main__":
    fns = {"pred": lambda: (fig_pred_compare("dyn"), fig_pred_compare("dyn_fast"), fig_pred_each("dyn")),
           "mod": fig_mod, "dyn3": fig_dyn3, "shared": fig_shared,
           "decomp": lambda: (fig_dev_seq("dyn"), fig_decomp(), fig_dev_age()),
           "family": lambda: (fig_decomp_family(), fig_dev_seq("dyn", fam="moa"), fig_dev_seq("dyn", fam="momha"),
                              fig_pred_compare("dyn", fam="moa"), fig_pred_compare("dyn", fam="momha"))}
    for w in sys.argv[1:] or fns:
        fns[w]()
