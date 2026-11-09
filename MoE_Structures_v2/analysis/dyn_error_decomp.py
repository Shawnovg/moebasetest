"""Where does the MSE difference between models come from? (question of 9 Nov)

For every dynamics dataset and every attention model (router-free controls without balancing, routed
models with the load-balance loss), 3 seeds:

* per-position squared error on the whole test set (2000 sequences, positions t >= 8, as ``mse_ctx8``);
* the difference to Dense split by regime age (0 = the switch step itself, which no causal model can
  foresee; 1 = the first step after it; ...), as *contributions* that add up to the total difference;
* how concentrated the difference is: share of the net difference carried by the 1 % / 5 % / 10 % of
  positions with the largest |difference|;
* how far predictions are from each other: RMS distance seed-vs-seed (same model), model-vs-model and
  model-vs-Bayes, by age.

    python -m analysis.dyn_error_decomp            # writes reports/tables/dyn_error_*.csv + a cache
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import dyn_predictions as P  # noqa: E402
from backbone.metrics_dyn import AGE_BINS, CTX, regime_age  # noqa: E402
from data_gen.dynamics import transition_matrices  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "reports" / "tables"
CACHE = ROOT / "reports" / "cache"
DATASETS = ["dyn", "dyn_fast", "dyn_p50"]
MODELS = [("dense", "default", "none"), ("mha8", "default", "none"), ("mqa4", "default", "none"),
          ("mqa8", "default", "none"), ("gqa4", "default", "none"), ("gqa8", "default", "none"),
          ("moh", "default", "lb"), ("moa", "default", "lb"), ("momha", "default", "lb")]
SEEDS = (0, 1, 2)


def bayes_all(data):
    X = data.raw.double().numpy()
    R = data.regimes.numpy()
    A = transition_matrices().astype(np.float64)
    stay = data.manifest["stay_probability"]
    Pm = np.full((3, 3), (1 - stay) / 2); np.fill_diagonal(Pm, stay)
    r_obs = np.concatenate([R[:, :1], R[:, :-1]], axis=1)
    cand = np.einsum("kij,ntj->ntki", A, X[:, :-1])
    return np.einsum("ntk,ntki->nti", Pm[r_obs], cand)


@torch.no_grad()
def predict_all(model, data):
    out = []
    for i in range(0, len(data.inputs), 500):
        p, _ = model(data.inputs[i:i + 500])
        out.append(p * data.std + data.mean)
    return torch.cat(out).double().numpy()


def load_preds(ds):
    f = CACHE / f"dyn_preds_{ds}.npz"
    if f.exists():
        return dict(np.load(f))
    torch.set_num_threads(1)
    data = P.DynamicsDataset(ds, "test")
    out = {"x_next": data.raw[:, 1:].double().numpy(), "regimes": data.regimes.numpy(), "bayes": bayes_all(data)}
    for s, v, b in MODELS:
        for sd in SEEDS:
            out[f"{s}_s{sd}"] = predict_all(P.load_model(s, ds, sd, v, b), data).astype(np.float32)
            print(ds, s, sd, flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **out)
    return out


def analyse(ds):
    z = load_preds(ds)
    y = z["x_next"][:, CTX:]
    age = regime_age(torch.from_numpy(z["regimes"])).numpy()[:, CTX:]
    n = age.size
    se = lambda p: ((p[:, CTX:] - y) ** 2).mean(-1)
    se_b = se(z["bayes"])
    names = [m[0] for m in MODELS]
    se_m = {s: np.stack([se(z[f"{s}_s{sd}"]) for sd in SEEDS]) for s in names}      # seeds x N x T
    bins = [(b, (age >= lo) & (age <= hi)) for b, lo, hi in AGE_BINS]
    dec, conc, dist = [], [], []
    for s in names:
        for sd in SEEDS:
            d = se_m[s][sd] - se_m["dense"][sd]          # paired by seed: model minus Dense (negative = better)
            e = se_m[s][sd] - se_b                       # excess over Bayes
            row = {"dataset": ds, "structure": s, "seed": sd, "mse": se_m[s][sd].mean(), "excess": e.mean(),
                   "diff_vs_dense": d.mean()}
            for b, m in bins:
                row[f"share_{b}"] = m.sum() / n
                row[f"diff_{b}"] = d[m].sum() / n        # contribution: these add up to diff_vs_dense
                row[f"excess_{b}"] = e[m].sum() / n      # contribution: these add up to excess
            dec.append(row)
            if s != "dense":
                a = np.sort(np.abs(d).ravel())[::-1]
                order = np.argsort(-np.abs(d).ravel())
                cum = np.cumsum(d.ravel()[order])
                r = {"dataset": ds, "structure": s, "seed": sd, "net_diff": d.mean(),
                     "frac_positions_better": float((d < 0).mean())}
                for q in (0.01, 0.05, 0.10, 0.25):
                    k = int(q * d.size)
                    r[f"top{int(q * 100)}pct_share_of_net"] = cum[k - 1] / d.sum()
                    r[f"top{int(q * 100)}pct_share_of_abs"] = a[:k].sum() / a.sum()
                conc.append(r)
    # distances between predictions (RMS over the 4 coordinates), by age bin
    pr = {f"{s}_s{sd}": z[f"{s}_s{sd}"][:, CTX:].astype(np.float64) for s in names for sd in SEEDS}
    rms = lambda a, b, m: float(np.sqrt(((a - b) ** 2).mean(-1)[m].mean()))
    by = z["bayes"][:, CTX:]
    allm = np.ones_like(age, bool)
    for b, m in [("all", allm)] + bins:
        for s in names:
            seeds = [rms(pr[f"{s}_s{i}"], pr[f"{s}_s{j}"], m) for i, j in ((0, 1), (0, 2), (1, 2))]
            dist.append({"dataset": ds, "age": b, "a": s, "b": "(other seed)", "rms": np.mean(seeds)})
            dist.append({"dataset": ds, "age": b, "a": s, "b": "bayes",
                         "rms": np.mean([rms(pr[f"{s}_s{i}"], by, m) for i in SEEDS])})
            if s != "dense":
                dist.append({"dataset": ds, "age": b, "a": s, "b": "dense",
                             "rms": np.mean([rms(pr[f"{s}_s{i}"], pr[f"dense_s{i}"], m) for i in SEEDS])})
        dist.append({"dataset": ds, "age": b, "a": "truth", "b": "bayes", "rms": rms(y, by, m)})
    return pd.DataFrame(dec), pd.DataFrame(conc), pd.DataFrame(dist)


def main():
    out = [analyse(ds) for ds in DATASETS]
    for i, name in enumerate(["dyn_error_decomp", "dyn_error_concentration", "dyn_pred_distance"]):
        pd.concat([o[i] for o in out], ignore_index=True).to_csv(TAB / f"{name}.csv", index=False)
        print(TAB / f"{name}.csv")


if __name__ == "__main__":
    main()
