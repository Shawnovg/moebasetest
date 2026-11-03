"""Reference predictors for the dynamics task (no learning; computed from the test split).

* ``oracle_true``      knows the regime r[t] of the transition it predicts: x_hat = A[r[t]] x[t].
                       Its error is the noise floor (~ noise_std^2 = 0.0064) and needs information
                       no model has at a switch.
* ``oracle_lastobs``   knows the regime of the last observed transition r[t-1] (the best a model could
                       do if it identified the current regime perfectly from the past): x_hat = A[r[t-1]] x[t].
* ``bayes_lastobs``    the best possible label-free prediction in expectation once the past regime is known:
                       the Markov-chain mixture  sum_k P(r[t]=k | r[t-1]) A_k x[t]  (stay prob on the diagonal).
                       This is the practical floor of the task: no causal model can beat it on average.
* ``infer_window``     no regime labels: picks the matrix that best explains the last w observed
                       transitions (least squares, w = 1, 2, 4, 8), then predicts with it. A simple,
                       label-free estimate of how well the regime can be inferred from the context.
* ``persistence``      x_hat = x[t];  ``zero``  x_hat = 0 (the mean).

Errors in original units averaged over the 4 coordinates, positions t >= 8 (as ``mse_ctx8``), and by
regime age (same bins as backbone/metrics_dyn.py).

    python -m analysis.dyn_reference  -> reports/tables/dyn_reference.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backbone.metrics_dyn import AGE_BINS, CTX, regime_age  # noqa: E402
from backbone.paths import DATA_ROOT, PROJECT_ROOT  # noqa: E402
from data_gen.dynamics import transition_matrices  # noqa: E402

TABLES = PROJECT_ROOT / "reports" / "tables"


def references(name, root=DATA_ROOT, windows=(1, 2, 4, 8)):
    z = np.load(Path(root) / name / "test.npz")
    X, R = z["states"].astype(np.float64), z["regimes"]
    A = transition_matrices().astype(np.float64)
    N, T = R.shape
    x_t, x_next = X[:, :-1], X[:, 1:]
    preds = {}
    preds["oracle_true"] = np.einsum("ntij,ntj->nti", A[R], x_t)
    r_obs = np.concatenate([R[:, :1], R[:, :-1]], axis=1)
    preds["oracle_lastobs"] = np.einsum("ntij,ntj->nti", A[r_obs], x_t)
    import json
    stay = json.loads((Path(root) / name / "manifest.json").read_text())["stay_probability"]
    P = np.full((3, 3), (1 - stay) / 2); np.fill_diagonal(P, stay)
    preds["bayes_lastobs"] = np.einsum("ntk,ntkij,ntj->nti", P[r_obs], np.broadcast_to(A, (N, T, 3, 4, 4)), x_t)
    cand = np.einsum("kij,ntj->ntki", A, x_t)                                  # N T K 4: A_k x[t]
    res = ((cand - x_next[:, :, None, :]) ** 2).sum(-1)                         # residual of each matrix at t
    for w in windows:
        p = np.zeros_like(x_t)
        for t in range(T):
            lo = max(0, t - w)
            if t == 0:
                k = np.zeros(N, dtype=int)
            else:
                k = res[:, lo:t].sum(1).argmin(-1)                             # best matrix on transitions lo..t-1
            p[:, t] = cand[np.arange(N), t, k]
        preds[f"infer_window{w}"] = p
    preds["persistence"] = x_t
    preds["zero"] = np.zeros_like(x_t)
    age = regime_age(torch.from_numpy(R)).numpy()[:, CTX:]
    rows = []
    for k, p in preds.items():
        se = ((p - x_next) ** 2).mean(-1)
        row = {"dataset": name, "reference": k, "mse": se.mean(), "mse_ctx8": se[:, CTX:].mean()}
        for b, lo, hi in AGE_BINS:
            m = (age >= lo) & (age <= hi)
            row[f"age_{b}"] = se[:, CTX:][m].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    df = pd.concat([references(n) for n in ("dyn", "dyn_fast")], ignore_index=True)
    df.to_csv(TABLES / "dyn_reference.csv", index=False)
    with pd.option_context("display.width", 200):
        print(df.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
