"""Evaluation for the switching-dynamics task (datasets dyn / dyn_fast).

All errors are reported in ORIGINAL state units (the normalisation is undone), averaged over the 4
coordinates, so they compare directly with the noise floor noise_std^2 = 0.0064.

* ``mse``        one-step prediction MSE over every position t = 0..63 (context = states 0..t)
* ``mse_ctx8``   the same, positions t >= 8 only (at least 8 observed states)
* ``mse_age``    positions t >= 8 split by how long the current regime has lasted:
                 age 0 = state t+1 is the FIRST state produced by a new regime (a switch at t: cannot be
                 predicted from the past), age 1 = one transition of the new regime has been observed, ...
                 bins "0", "1", "2", "3", "4-7", "8+" (the 8+ bin includes regimes that started before the window)
* ``rollout``    feed the true states 0..31, then predict 16 steps ahead feeding back the model's own
                 predictions; MSE at horizon h = 1..16 (first 512 test sequences)
* ``routing``    per routed module: unit counts split by the regime r[t] of the predicted transition
                 (``per_regime``) and by the regime of the last OBSERVED transition r[t-1] (``per_regime_obs``),
                 plus load / entropy / max_load / dead as for the token tasks.
"""
from __future__ import annotations

from collections import defaultdict

import torch

from .metrics import route_groups, usage_summary

AGE_BINS = [("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-7", 4, 7), ("8+", 8, 10 ** 9)]
CTX = 8


def regime_age(regimes: torch.Tensor) -> torch.Tensor:
    """B x T: steps since the regime of transition t started (0 = switch at t). The first segment of
    every window gets age >= CTX (its start is unknown, at least t + CTX steps ago is assumed)."""
    age = torch.zeros_like(regimes)
    age[:, 0] = CTX
    for t in range(1, regimes.shape[1]):
        same = regimes[:, t] == regimes[:, t - 1]
        age[:, t] = torch.where(same, age[:, t - 1] + 1, torch.zeros_like(age[:, t]))
    return age


@torch.no_grad()
def rollout(model, ds, n_seq=512, context=32, horizon=16, batch_size=256):
    """MSE (original units) at horizons 1..horizon, predictions fed back autoregressively."""
    sq = torch.zeros(horizon)
    n = min(n_seq, len(ds))
    std2 = ds.std ** 2
    for i in range(0, n, batch_size):
        idx = torch.arange(i, min(i + batch_size, n))
        seq = ds.inputs[idx, :context].clone()
        truth = ds.targets[idx, context - 1:context - 1 + horizon]           # states context .. context+h-1
        for h in range(horizon):
            pred, _ = model(seq)
            nxt = pred[:, -1:]
            sq[h] += (((nxt[:, 0] - truth[:, h]) ** 2) * std2).mean(-1).sum()
            seq = torch.cat([seq, nxt], dim=1)
    return (sq / n).tolist()


@torch.no_grad()
def evaluate_dynamics(model, ds, batch_size=256, max_batches=None, routing=True, do_rollout=False):
    model.eval()
    st = model.state
    st.record = routing
    random_was = st.random_active
    st.random_active = False
    std2 = ds.std ** 2
    se_sum, n_all, se_ctx, n_ctx = 0.0, 0, 0.0, 0
    age_se, age_n = defaultdict(float), defaultdict(int)
    counts, counts_obs = {}, {}
    for inp, tgt, reg in ds.eval_batches(batch_size, max_batches):
        pred, _ = model(inp)
        se = (((pred - tgt) ** 2) * std2).mean(-1)                           # B x T, original units
        se_sum += float(se.sum()); n_all += se.numel()
        se_ctx += float(se[:, CTX:].sum()); n_ctx += se[:, CTX:].numel()
        age = regime_age(reg)[:, CTX:]
        s8 = se[:, CTX:]
        for name, lo, hi in AGE_BINS:
            m = (age >= lo) & (age <= hi)
            age_se[name] += float(s8[m].sum()); age_n[name] += int(m.sum())
        if routing:
            reg_obs = torch.cat([reg[:, :1], reg[:, :-1]], dim=1)            # r[t-1] (r[0] at t = 0)
            for mname, mod in model.routed_modules():
                for g, rec in route_groups(mod).items():
                    key = f"{mname}.{g}" if g else mname
                    U = mod.num_units
                    for store, lab in ((counts, reg), (counts_obs, reg_obs)):
                        if key not in store:
                            store[key] = defaultdict(lambda U=U: torch.zeros(U, dtype=torch.long))
                        for r in range(3):
                            sel = rec[lab == r].reshape(-1)
                            sel = sel[sel >= 0]
                            store[key][r] += torch.bincount(sel, minlength=U)[:U]
    st.record = False
    st.random_active = random_was
    out = {"mse": se_sum / max(n_all, 1), "mse_ctx8": se_ctx / max(n_ctx, 1),
           "mse_age": {k: age_se[k] / max(age_n[k], 1) for k, _, _ in AGE_BINS},
           "age_counts": {k: age_n[k] for k, _, _ in AGE_BINS}}
    if routing:
        r = {}
        for key, per_r in counts.items():
            s = usage_summary(sum(per_r.values()))
            s["per_regime"] = {str(k): v.tolist() for k, v in per_r.items()}
            s["per_regime_obs"] = {str(k): v.tolist() for k, v in counts_obs[key].items()}
            r[key] = s
        out["routing"] = r
    if do_rollout:
        out["rollout"] = rollout(model, ds)
    model.train()
    return out
