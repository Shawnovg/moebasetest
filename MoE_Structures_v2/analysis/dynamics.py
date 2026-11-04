"""Tables and figures for the switching-dynamics study (datasets dyn / dyn_fast).

    python -m analysis.dynamics      -> reports/tables/dyn_*.csv, reports/figures/dyn_*.png

Main number: ``excess`` = test MSE (positions t >= 8, original units) minus the Bayes floor of the
dataset (analysis/dyn_reference.py, ``bayes_lastobs``), i.e. how much worse than the best possible
causal predictor a model is. Lower is better; 0 = optimal.

Routing vs regime: for every routed module the unit picks are counted per hidden regime. ``regime_mi``
= mutual information between the unit picked and the regime, divided by log(3) (0 = routing ignores
the regime, 1 = the unit identifies the regime exactly). ``_obs`` uses the regime of the last observed
transition r[t-1] (what the router could know); the plain version the regime of the predicted one r[t].
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.figures import (INK, INK2, STRUCT_COLOR, STRUCT_LABEL, STRUCT_ORDER, _save, plt)  # noqa: E402
from backbone.metrics_dyn import AGE_BINS  # noqa: E402
from backbone.paths import DATA_ROOT, FIG_ROOT, PROJECT_ROOT, RUNS_ROOT  # noqa: E402

TABLES = PROJECT_ROOT / "reports" / "tables"
DYN_DS = ["dyn", "dyn_fast"]
DS_LABEL = {"dyn": "dyn (switch every ~20 steps)", "dyn_fast": "dyn_fast (switch every ~5 steps)"}
ROUTED = ["moe", "moh", "moa", "momha"]
ID = ["dataset", "structure", "variant", "seed"]


# ---------------------------------------------------------------- collection
def load(runs_root=RUNS_ROOT, balance="none"):
    """Runs of one balancing mode; Dense / Dense-wide (no router) always come from the 'none' runs."""
    rows, routes, hist = [], [], []
    for p in sorted(Path(runs_root).glob("dyn*/result.json")):
        r = json.loads(p.read_text())
        if r.get("task") != "dynamics":
            continue
        want = "none" if r["structure"] in ("dense", "dense_wide", "mha8", "mqa8", "gqa8") else balance
        if r.get("balance", "none") != want:
            continue
        row = {k: r[k] for k in ID + ["params", "params_total", "train_time_s", "test_mse", "test_mse_ctx8",
                                      "valid_mse"]}
        for b, v in r["test_mse_age"].items():
            row[f"age_{b}"] = v
        for h, v in enumerate(r["test_rollout"], 1):
            row[f"roll_{h}"] = v
        rows.append(row)
        for mod, v in r.get("routing", {}).items():
            routes.append({**{k: r[k] for k in ID}, "module": mod, "entropy": v["entropy"],
                           "max_load": v["max_load"], "dead": v["dead"], "load": v["load"],
                           "regime_mi": regime_mi(v["per_regime"]), "regime_mi_obs": regime_mi(v["per_regime_obs"]),
                           "per_regime": v["per_regime"], "per_regime_obs": v["per_regime_obs"]})
        for line in (p.parent / "history.jsonl").read_text().splitlines():
            h = json.loads(line)
            hist.append({**{k: r[k] for k in ID}, "step": h["step"], "train_loss": h["train_loss"],
                         "valid_mse": h["valid_mse"]})
    return pd.DataFrame(rows), pd.DataFrame(routes), pd.DataFrame(hist)


def regime_mi(per_regime):
    C = np.array([per_regime[str(k)] for k in range(3)], dtype=float)      # regimes x units
    P = C / C.sum()
    pr, pu = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
    nz = P > 0
    return float((P[nz] * np.log(P[nz] / (pr @ pu)[nz])).sum() / np.log(3))


def references():
    p = TABLES / "dyn_reference.csv"
    if not p.exists():
        from analysis.dyn_reference import main as ref_main
        ref_main()
    return pd.read_csv(p)


def floor(ref, ds):
    return float(ref[(ref.dataset == ds) & (ref.reference == "bayes_lastobs")].mse_ctx8.iloc[0])


# ---------------------------------------------------------------- tables
def tables(df, rt, ref):
    df = df.copy()
    df["floor"] = df.dataset.map({d: floor(ref, d) for d in DYN_DS})
    df["excess"] = df.test_mse_ctx8 - df.floor
    g = df.groupby(["dataset", "structure", "variant"])
    main = g.agg(mse=("test_mse_ctx8", "mean"), mse_sd=("test_mse_ctx8", "std"), excess=("excess", "mean"),
                 excess_sd=("excess", "std"), n=("seed", "count"), params=("params", "first"),
                 time_s=("train_time_s", "mean")).reset_index()
    main["excess_pct_of_floor"] = 100 * main.excess / main.dataset.map({d: floor(ref, d) for d in DYN_DS})
    age = g[[f"age_{b}" for b, _, _ in AGE_BINS]].mean().reset_index()
    roll = g[[c for c in df.columns if c.startswith("roll_")]].mean().reset_index()
    rtab = pd.DataFrame()
    if len(rt):
        rt = rt.copy()
        rt["layer"] = rt.module.str.extract(r"blocks\.(\d)").astype(int)
        rt["kind"] = np.where(rt.module.str.contains("attn"), "attn", "ffn")
        rtab = rt.groupby(["dataset", "structure", "variant", "layer", "kind"]).agg(
            entropy=("entropy", "mean"), max_load=("max_load", "mean"), dead=("dead", "mean"),
            regime_mi=("regime_mi", "mean"), regime_mi_obs=("regime_mi_obs", "mean"),
            regime_mi_obs_sd=("regime_mi_obs", "std")).reset_index()
    # paired (same seed) difference to Dense
    pair = []
    for (ds, s, v), x in df.groupby(["dataset", "structure", "variant"]):
        base = df[(df.dataset == ds) & (df.structure == "dense")].set_index("seed").test_mse_ctx8
        d = x.set_index("seed").test_mse_ctx8 - base.reindex(x.seed).values
        pair.append({"dataset": ds, "structure": s, "variant": v, "minus_dense": d.mean(),
                     "minus_dense_sd": d.std(), "wins_vs_dense": int((d < 0).sum()), "n": len(d)})
    pair = pd.DataFrame(pair)
    main = main.merge(pair, on=["dataset", "structure", "variant"], how="left")
    return df, main, age, roll, rtab


# ---------------------------------------------------------------- figures
def _order(structs):
    return [s for s in STRUCT_ORDER if s in set(structs)]


def fig_example(ds="dyn", n_show=64, out=FIG_ROOT):
    """One test trajectory: the 4 coordinates with the hidden regime shaded."""
    z = np.load(Path(DATA_ROOT) / ds / "test.npz")
    X, R = z["states"][0], z["regimes"][0]
    shade = ["#dbe8f9", "#fbe3d6", "#d5f0e5"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    for t in range(n_show):
        ax.axvspan(t, t + 1, color=shade[R[t]], lw=0)
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
    for i in range(4):
        ax.plot(np.arange(n_show + 1), X[:n_show + 1, i], color=cols[i], lw=1.4, label=f"x{i + 1}")
    ax.set_ylabel("state value")
    ax.legend(ncol=4, loc="upper right", fontsize=8)
    ax.set_title(f"{ds}: one test sequence; background = hidden regime (never shown to the model)")
    axes[1].step(np.arange(n_show), R[:n_show], where="post", color=INK, lw=1.4)
    axes[1].set_yticks([0, 1, 2], ["A0", "A1", "A2"])
    axes[1].set_xlabel("time step t")
    axes[1].set_ylabel("regime")
    return _save(fig, f"dyn_example_{ds}", out)


def fig_phase(ds="dyn", out=FIG_ROOT):
    """The (x1, x2) plane: each regime is a rotation by its own angle per step (plus shrink + noise)."""
    from data_gen.dynamics import ANGLES
    z = np.load(Path(DATA_ROOT) / ds / "test.npz")
    X, R = z["states"][:200], z["regimes"][:200]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1), sharex=True, sharey=True)
    for k, ax in enumerate(axes):
        m = R == k
        a, b = X[:, :-1][m], X[:, 1:][m]
        idx = np.random.default_rng(0).choice(len(a), 60, replace=False)
        for i in idx:
            ax.annotate("", xy=b[i, :2], xytext=a[i, :2],
                        arrowprops=dict(arrowstyle="->", color=["#2a78d6", "#eb6834", "#1baf7a"][k], lw=0.8))
        ax.set_title(f"regime A{k}\nturn {np.degrees(ANGLES[k][0]):.0f}° in (x1,x2), "
                     f"{np.degrees(ANGLES[k][1]):.0f}° in (x3,x4)", fontsize=8)
        ax.set_xlim(-0.7, 0.7); ax.set_ylim(-0.7, 0.7); ax.set_aspect("equal")
        ax.set_xlabel("x1")
    axes[0].set_ylabel("x2")
    fig.suptitle("One step x[t] → x[t+1] in the (x1, x2) plane, per regime", fontsize=10, fontweight="bold")
    return _save(fig, f"dyn_phase_{ds}", out)


def fig_main(main, ref, out=FIG_ROOT):
    d = main[main.variant == "default"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, ds in zip(axes, DYN_DS):
        x = d[d.dataset == ds].set_index("structure")
        ss = _order(x.index)
        e, sd = x.loc[ss, "excess"].values * 1e3, x.loc[ss, "excess_sd"].values * 1e3
        ax.bar(range(len(ss)), e, yerr=sd, capsize=3, color=[STRUCT_COLOR[s] for s in ss], width=0.6)
        for i in range(len(ss)):
            ax.text(i, e[i] + sd[i] + 0.004, f"{e[i]:.3f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(range(len(ss)), [STRUCT_LABEL[s] for s in ss])
        f = floor(ref, ds)
        ax.set_title(f"{DS_LABEL[ds]}  (Bayes floor {f * 1e3:.2f} x 1e-3)", fontsize=9)
        ax.set_ylim(0, max(e + sd) * 1.18)
    axes[0].set_ylabel("test MSE − Bayes floor  (x 1e-3)")
    fig.suptitle("Error above the best possible causal predictor (mean ± sd over 3 seeds, lower is better)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "dyn_main_excess", out)


def fig_age(age, ref, ds, out=FIG_ROOT):
    bins = [b for b, _, _ in AGE_BINS]
    a = age[(age.dataset == ds) & (age.variant == "default")].set_index("structure")
    r = ref[ref.dataset == ds].set_index("reference")
    fl = np.array([r.loc["bayes_lastobs", f"age_{b}"] for b in bins])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, ls, lab in [("bayes_lastobs", "--", "Bayes floor"), ("oracle_true", ":", "noise floor (knows future regime)"),
                          ("infer_window1", "-.", "pick matrix from last transition")]:
        ax.plot(range(len(bins)), [r.loc[name, f"age_{b}"] for b in bins], ls, color=INK2, lw=1.2, label=lab)
    ax.plot(range(len(bins)), [a.loc["dense", f"age_{b}"] for b in bins], "o-", ms=4, lw=1.6,
            color=STRUCT_COLOR["dense"], label="Dense (all models overlap at this scale)")
    ax.set_yscale("log")
    ax.set_xticks(range(len(bins)), bins)
    ax.set_xlabel("steps since the last regime switch (0 = the switch itself)")
    ax.set_ylabel("test MSE (log scale)")
    ax.set_title("absolute error", fontsize=9)
    ax.legend(fontsize=7)
    ss = _order(a.index)
    w = 0.8 / len(ss)
    for j, s in enumerate(ss):
        v = (np.array([a.loc[s, f"age_{b}"] for b in bins[1:]]) - fl[1:]) * 1e3
        bx.bar(np.arange(1, len(bins)) + (j - (len(ss) - 1) / 2) * w, v, width=w * 0.9,
               color=STRUCT_COLOR[s], label=STRUCT_LABEL[s])
    bx.axhline(0, color=INK2, lw=0.8)
    bx.set_xticks(range(1, len(bins)), bins[1:])
    bx.set_xlabel("steps since the last regime switch (age 0 left out: same for every model)")
    bx.set_ylabel("model − Bayes floor  (x 1e-3)")
    bx.set_title("excess over the Bayes floor, by age", fontsize=9)
    bx.legend(fontsize=7, ncol=3)
    fig.suptitle(f"{DS_LABEL[ds]}: error around regime switches", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _save(fig, f"dyn_age_{ds}", out)


def fig_rollout(roll, ref, ds, out=FIG_ROOT):
    a = roll[(roll.dataset == ds) & (roll.variant == "default")].set_index("structure")
    H = [c for c in roll.columns if c.startswith("roll_")]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for s in _order(a.index):
        ax.plot(range(1, len(H) + 1), a.loc[s, H].values, "o-", ms=3, lw=1.5, color=STRUCT_COLOR[s],
                label=STRUCT_LABEL[s])
    zero = float(ref[(ref.dataset == ds) & (ref.reference == "zero")].mse.iloc[0])
    ax.axhline(zero, color=INK2, ls="--", lw=1, label="predict 0 (no information)")
    ax.set_xlabel("steps predicted ahead (own predictions fed back)")
    ax.set_ylabel("test MSE")
    ax.set_title(f"{ds}: 16-step rollout after 32 observed states")
    ax.legend(fontsize=7, ncol=2)
    return _save(fig, f"dyn_rollout_{ds}", out)


def fig_variants(main, ds="dyn", out=FIG_ROOT):
    d = main[(main.dataset == ds) & (main.structure.isin(ROUTED))]
    rows = []
    for s in ROUTED:
        x = d[d.structure == s]
        for v in ["default"] + sorted(set(x.variant) - {"default"}):
            if v in set(x.variant):
                rows.append(x[x.variant == v].iloc[0])
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(rows) + 1.2))
    for i, r in enumerate(rows):
        ax.errorbar(r.excess * 1e3, i, xerr=r.excess_sd * 1e3, fmt="o", color=STRUCT_COLOR[r.structure],
                    capsize=3, ms=6 if r.variant == "default" else 5,
                    mfc=STRUCT_COLOR[r.structure] if r.variant == "default" else "white")
    dense = main[(main.dataset == ds) & (main.structure == "dense")].excess.iloc[0] * 1e3
    ax.axvline(dense, color=STRUCT_COLOR["dense"], ls="--", lw=1, label="Dense")
    dw = main[(main.dataset == ds) & (main.structure == "dense_wide")]
    if len(dw):
        ax.axvline(dw.excess.iloc[0] * 1e3, color=STRUCT_COLOR["dense_wide"], ls=":", lw=1, label="Dense-wide")
    ax.set_yticks(range(len(rows)), [f"{STRUCT_LABEL[r.structure]} {r.variant}" for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("test MSE − Bayes floor (x 1e-3, lower is better); filled = default")
    ax.set_title(f"{ds}: variants (mean ± sd, 3 seeds)")
    ax.legend(fontsize=8)
    return _save(fig, f"dyn_variants_{ds}", out)


def fig_regime_routing(rt, ds="dyn", seed=0, out=FIG_ROOT):
    """Unit share per regime (rows) for each routed module, default configs, one seed."""
    d = rt[(rt.dataset == ds) & (rt.variant == "default") & (rt.seed == seed)]
    mods = [(s, m) for s in ROUTED for m in sorted(set(d[d.structure == s].module))]
    if not mods:
        return None
    fig, axes = plt.subplots(len(mods), 1, figsize=(5.6, 0.95 * len(mods) + 0.6))
    for ax, (s, m) in zip(np.atleast_1d(axes), mods):
        r = d[(d.structure == s) & (d.module == m)].iloc[0]
        C = np.array([r.per_regime_obs[str(k)] for k in range(3)], dtype=float)
        P = C / C.sum(1, keepdims=True)
        ax.imshow(P, cmap="Blues", vmin=0, vmax=max(0.5, P.max()), aspect="auto")
        for i in range(3):
            for j in range(P.shape[1]):
                ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if P[i, j] > 0.35 else INK)
        ax.set_yticks([0, 1, 2], ["A0", "A1", "A2"], fontsize=7)
        ax.set_xticks(range(P.shape[1]), [str(j) for j in range(P.shape[1])], fontsize=7)
        ax.set_ylabel(f"{STRUCT_LABEL[s]}\n{m.replace('blocks.', 'L').replace('.attn', ' attn').replace('.ffn', ' ffn')}",
                      fontsize=7, rotation=0, ha="right", va="center")
        ax.grid(False)
        ax.text(1.01, 0.5, f"MI {r.regime_mi_obs:.2f}", transform=ax.transAxes, fontsize=7, va="center")
    fig.suptitle(f"{ds}, seed {seed}: share of each unit's picks per (last observed) regime",
                 fontsize=9, fontweight="bold")
    fig.tight_layout()
    return _save(fig, f"dyn_regime_routing_{ds}", out)


def fig_curves(hist, ds, out=FIG_ROOT):
    h = hist[(hist.dataset == ds) & (hist.variant == "default")].groupby(["structure", "step"]).valid_mse.mean()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for s in _order(h.index.get_level_values(0)):
        x = h.loc[s]
        ax.plot(x.index, x.values, "o-", ms=3, color=STRUCT_COLOR[s], label=STRUCT_LABEL[s])
    ax.set_yscale("log")
    ax.set_xlabel("training step"); ax.set_ylabel("validation MSE (all positions)")
    ax.set_title(f"{ds}: learning curves (mean of seeds)")
    ax.legend(fontsize=7, ncol=2)
    return _save(fig, f"dyn_curves_{ds}", out)


def main(balance="none"):
    """balance 'none' -> reports/tables/dyn_*.csv and reports/figures/dyn_*.png;
    'lb' -> dyn_*_lb.csv and reports/figures/dyn_lb/."""
    TABLES.mkdir(parents=True, exist_ok=True)
    ref = references()
    df, rt, hist = load(balance=balance)
    sfx = "" if balance == "none" else f"_{balance}"
    out = FIG_ROOT if balance == "none" else Path(FIG_ROOT) / f"dyn_{balance}"
    if balance == "none":
        for ds in DYN_DS:
            fig_example(ds)
        fig_phase("dyn")
    if not len(df):
        print("no dynamics runs yet")
        return
    df, main_t, age, roll, rtab = tables(df, rt, ref)
    df.to_csv(TABLES / f"dyn_runs{sfx}.csv", index=False)
    main_t.to_csv(TABLES / f"dyn_main{sfx}.csv", index=False)
    age.to_csv(TABLES / f"dyn_age{sfx}.csv", index=False)
    roll.to_csv(TABLES / f"dyn_rollout{sfx}.csv", index=False)
    if len(rtab):
        rtab.to_csv(TABLES / f"dyn_routing{sfx}.csv", index=False)
    fig_main(main_t, ref, out=out)
    for ds in sorted(set(df.dataset)):
        fig_age(age, ref, ds, out=out)
        fig_rollout(roll, ref, ds, out=out)
        fig_curves(hist, ds, out=out)
        if len(rt):
            fig_regime_routing(rt, ds, out=out)
    if (main_t.variant != "default").any():
        fig_variants(main_t, "dyn", out=out)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(main_t.round(6).to_string(index=False))
        if len(rtab):
            print(rtab.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "none")
