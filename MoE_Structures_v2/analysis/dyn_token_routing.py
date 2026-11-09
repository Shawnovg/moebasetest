"""Token-level routing vs hidden regime on the switching-dynamics test set (question of 9 Nov).

For each routed attention layer, every token (state x[t]) gets a set of picked units (MoH / MoA: 4 of 8
heads, MoMHA: 2 of 4 blocks). The regime used is r_obs[t] = r[t-1], the regime of the last transition the
model has already seen (the router cannot know the regime of the transition it is predicting).

* ``jacc_*``   overlap |A∩B| / |A∪B| of two tokens' unit sets (1 = identical picks)
    - adj_same : neighbours t-1, t in one sequence, same regime
    - adj_switch: neighbours across a regime switch
    - rand_same / rand_diff: tokens from different sequences with the same / a different regime
      (the "no memory of the previous token" reference)
* ``decode``   accuracy of guessing the regime from the unit set alone (lookup table fitted on the first
               half of the test sequences, scored on the second half); ``chance`` = most frequent regime
* ``decode_age`` the same accuracy split by steps since the last switch (0 = the first token whose
               observed regime is new)

    python -m analysis.dyn_token_routing   -> reports/tables/dyn_token_routing.csv + figures
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.dyn_predictions import load_model  # noqa: E402
from backbone.data import DynamicsDataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STRUCTS = ["moh", "moa", "momha"]
AGES = [0, 1, 2, 3, 4, 6, 8, 16]


def record(model, X):
    model.state.record = True
    out = defaultdict(list)
    with torch.no_grad():
        for i in range(0, len(X), 250):
            model(X[i:i + 250])
            for n, mod in model.routed_modules():
                out[n].append(mod.route_record.clone())
    model.state.record = False
    return {n: torch.cat(v).numpy() for n, v in out.items()}, {n: m.num_units for n, m in model.routed_modules()}


def onehot(rec, U):
    B, T, k = rec.shape
    H = np.zeros((B, T, U), bool)
    np.put_along_axis(H, rec, True, axis=-1)
    return H


def jacc(a, b):
    i = (a & b).sum(-1); u = (a | b).sum(-1)
    return i / np.maximum(u, 1)


def ages(r):
    a = np.zeros_like(r); a[:, 0] = 99
    for t in range(1, r.shape[1]):
        a[:, t] = np.where(r[:, t] == r[:, t - 1], a[:, t - 1] + 1, 0)
    return a


def analyse(H, r, rng):
    B, T, U = H.shape
    out = {}
    same = r[:, 1:] == r[:, :-1]
    j = jacc(H[:, 1:], H[:, :-1])
    out["jacc_adj_same"], out["jacc_adj_switch"] = j[same].mean(), j[~same].mean()
    flat, rf = H.reshape(-1, U), r.reshape(-1)
    seq = np.repeat(np.arange(B), T)
    i, k = rng.integers(0, len(rf), 200000), rng.integers(0, len(rf), 200000)
    ok = seq[i] != seq[k]
    i, k = i[ok], k[ok]
    js = jacc(flat[i], flat[k]); sr = rf[i] == rf[k]
    out["jacc_rand_same"], out["jacc_rand_diff"] = js[sr].mean(), js[~sr].mean()
    # regime from the unit set: lookup table on the first half of sequences
    key = (flat * (1 << np.arange(U))).sum(-1)
    tr = seq < B // 2
    tab = defaultdict(Counter)
    for kk, rr in zip(key[tr], rf[tr]):
        tab[kk][rr] += 1
    maj = Counter(rf[tr]).most_common(1)[0][0]
    guess = np.array([tab[kk].most_common(1)[0][0] if kk in tab else maj for kk in key[~tr]])
    hit = guess == rf[~tr]
    out["decode"], out["chance"] = hit.mean(), (rf[~tr] == maj).mean()
    out["n_distinct_sets"] = len(set(key.tolist()))
    a = ages(r).reshape(-1)[~tr]
    for lo, hi in zip(AGES, AGES[1:] + [10 ** 6]):
        m = (a >= lo) & (a < hi)
        out[f"decode_age{lo}"] = hit[m].mean() if m.any() else np.nan
    # P(unit | regime)
    P = np.stack([flat[rf == q].mean(0) for q in range(3)])
    return out, P


def main(datasets=("dyn", "dyn_fast", "dyn_p50"), balance="lb", seeds=(0, 1, 2)):
    torch.set_num_threads(4)
    rng = np.random.default_rng(0)
    rows, mats = [], {}
    for ds in datasets:
        d = DynamicsDataset(ds, "test")
        R = d.regimes.numpy()
        r_obs = np.concatenate([R[:, :1], R[:, :-1]], 1)
        for s in STRUCTS:
            for sd in seeds:
                try:
                    m = load_model(s, ds, sd, balance=balance)
                except FileNotFoundError:
                    continue
                recs, units = record(m, d.inputs)
                for li, (n, rec) in enumerate(recs.items()):
                    res, P = analyse(onehot(rec, units[n]), r_obs, rng)
                    rows.append({"dataset": ds, "structure": s, "seed": sd, "layer": li + 1, **res})
                    mats[(ds, s, sd, li + 1)] = P
                print(ds, s, sd, flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "reports" / "tables" / "dyn_token_routing.csv"
    df.to_csv(out, index=False)
    np.save(ROOT / "reports" / "tables" / "dyn_token_routing_P.npy", mats, allow_pickle=True)
    print(df.groupby(["dataset", "structure", "layer"]).mean(numeric_only=True).round(3).drop(columns="seed")
          .to_string())


if __name__ == "__main__":
    main()


def fig_sequence(ds="dyn", idx=None, seed=0, balance="lb", t0=0, t1=64):
    """One test sequence: hidden (observed) regime as a colour band, then layer-2 picks per token."""
    from analysis.figures import INK, INK2, STRUCT_LABEL, _save, plt
    from analysis.dyn_predictions import SHADE
    d = DynamicsDataset(ds, "test")
    R = d.regimes.numpy(); r_obs = np.concatenate([R[:, :1], R[:, :-1]], 1)
    if idx is None:                                   # a sequence with several switches
        idx = int(np.argmax([(np.diff(r) != 0).sum() if (np.diff(r) != 0).sum() <= 6 else 0 for r in r_obs]))
    fig, axes = plt.subplots(len(STRUCTS) + 1, 1, figsize=(11, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [0.5] + [1] * len(STRUCTS)})
    ax = axes[0]
    for t in range(t0, t1):
        ax.axvspan(t - .5, t + .5, color=SHADE[r_obs[idx, t]], lw=0)
        ax.text(t, 0, "ABC"[r_obs[idx, t]], ha="center", va="center", fontsize=6, color=INK2)
    ax.set_yticks([]); ax.set_ylabel("regime", fontsize=8, rotation=0, ha="right", va="center")
    ax.set_title(f"{ds}, test sequence {idx}: which heads / blocks layer 2 picks for each token "
                 f"(seed {seed}, background = observed regime)", fontsize=10, fontweight="bold", loc="left")
    for ax, s in zip(axes[1:], STRUCTS):
        m = load_model(s, ds, seed, balance=balance)
        recs, units = record(m, d.inputs[idx:idx + 1])
        n = list(recs)[1]; U = units[n]
        H = onehot(recs[n], U)[0]
        for t in range(t0, t1):
            ax.axvspan(t - .5, t + .5, color=SHADE[r_obs[idx, t]], lw=0, alpha=.6)
        tt, uu = np.nonzero(H[t0:t1])
        ax.scatter(tt + t0, uu, s=14, marker="s", color=INK)
        ax.set_yticks(range(U)); ax.set_yticklabels([str(u) for u in range(U)], fontsize=7)
        ax.set_ylim(-.6, U - .4)
        ax.set_ylabel(f"{STRUCT_LABEL[s]}\n{'block' if s == 'momha' else 'head'}", fontsize=8,
                      rotation=0, ha="right", va="center")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[-1].set_xlabel("token position t (state x[t])")
    return _save(fig, f"dyn_token_routing_seq_{ds}")


def fig_decode(df=None):
    from analysis.figures import INK2, STRUCT_COLOR, STRUCT_LABEL, _save, plt
    if df is None:
        df = pd.read_csv(ROOT / "reports" / "tables" / "dyn_token_routing.csv")
    dss = ["dyn", "dyn_fast", "dyn_p50"]
    lab = {"dyn": "dyn (stay 0.95)", "dyn_fast": "dyn_fast (0.80)", "dyn_p50": "dyn_p50 (0.50)"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, layer in zip(axes, (1, 2)):
        g = df[df.layer == layer].groupby(["dataset", "structure"])["decode"].agg(["mean", "std"])
        for i, s in enumerate(STRUCTS):
            x = np.arange(len(dss)) + (i - 1) * 0.26
            mu = [g.loc[(ds, s), "mean"] for ds in dss]; sd = [g.loc[(ds, s), "std"] for ds in dss]
            ax.bar(x, mu, 0.24, yerr=sd, color=STRUCT_COLOR[s], label=STRUCT_LABEL[s], capsize=2,
                   error_kw={"lw": .8})
            for xx, m in zip(x, mu):
                ax.text(xx, m + .03, f"{m:.2f}", ha="center", fontsize=7)
        ax.axhline(1 / 3, color=INK2, ls="--", lw=1)
        ax.text(-0.42, 1 / 3 - .05, "chance 1/3", fontsize=7, color=INK2, ha="left")
        ax.set_xticks(range(len(dss))); ax.set_xticklabels([lab[d] for d in dss], fontsize=8)
        ax.set_title(f"layer {layer}", fontsize=9, loc="left")
        ax.set_ylim(0, 1.05)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("regime guessed from the\nrouter's pick alone (accuracy)")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("How much the router's choice reveals the hidden regime (mean ± sd over 3 seeds)",
                 fontsize=10, fontweight="bold", x=0.01, ha="left")
    return _save(fig, "dyn_token_routing_decode")
