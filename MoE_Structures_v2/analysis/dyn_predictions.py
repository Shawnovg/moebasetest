"""Prediction vs truth on one test sequence (the sequence of the dyn_example figure).

For each seed of one trained structure: the one-step prediction x_hat[t+1] = f(x[0..t]) at every position,
in original units, overlaid on the true trajectory, with the hidden regime shaded; bottom panel: squared
error per step (mean over the 4 coordinates) for each seed and for the Bayes-floor predictor.

    python -m analysis.dyn_predictions moh dense      -> reports/figures/dyn_pred_<ds>_<structure>.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.figures import INK, INK2, STRUCT_LABEL, _save, plt  # noqa: E402
from backbone.data import DynamicsDataset  # noqa: E402
from backbone.model import EncoderModel  # noqa: E402
from backbone.paths import FIG_ROOT, RUNS_ROOT  # noqa: E402
from data_gen.dynamics import transition_matrices  # noqa: E402

SHADE = ["#dbe8f9", "#fbe3d6", "#d5f0e5"]
SEED_COL = ["#d1495b", "#edae49", "#00798c"]


def load_model(structure, ds, seed, variant="default", balance="none"):
    run = Path(RUNS_ROOT) / f"{ds}__{structure}__{variant}__{balance}__s{seed}"
    cfg = json.loads((run / "config.json").read_text())
    m = EncoderModel(cfg, 0)
    m.load_state_dict(torch.load(run / "model.pt", map_location="cpu", weights_only=True))
    return m.eval()


@torch.no_grad()
def predict(model, data, idx):
    pred, _ = model(data.inputs[idx:idx + 1])
    return (pred[0] * data.std + data.mean).numpy()                   # T x 4: prediction of x[t+1]


def bayes(data, idx):
    X = data.raw[idx].double().numpy()
    R = data.regimes[idx].numpy()
    A = transition_matrices().astype(np.float64)
    stay = data.manifest["stay_probability"]
    P = np.full((3, 3), (1 - stay) / 2); np.fill_diagonal(P, stay)
    r_obs = np.concatenate([R[:1], R[:-1]])
    return np.stack([sum(P[r_obs[t], k] * A[k] @ X[t] for k in range(3)) for t in range(len(R))])


def fig_prediction(structure="moh", ds="dyn", idx=0, seeds=(0, 1, 2), variant="default", balance="none",
                   coords=(0, 1, 2, 3), t_max=64, out=FIG_ROOT):
    data = DynamicsDataset(ds, "test")
    X, R = data.raw[idx].numpy(), data.regimes[idx].numpy()
    preds = [predict(load_model(structure, ds, s, variant, balance), data, idx) for s in seeds]
    bay = bayes(data, idx)
    n = len(coords)
    fig, axes = plt.subplots(n + 1, 1, figsize=(11, 1.75 * n + 2.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.6] * n + [1.4]})
    t_next = np.arange(1, t_max + 1)                                  # prediction at t is for x[t+1]
    for ax, c in zip(axes[:n], coords):
        for t in range(t_max):
            ax.axvspan(t, t + 1, color=SHADE[R[t]], lw=0)
        ax.plot(np.arange(t_max + 1), X[:t_max + 1, c], color=INK, lw=2.0, label="true state")
        for s, p in zip(seeds, preds):
            ax.plot(t_next, p[:t_max, c], color=SEED_COL[s % 3], lw=1.3, ls="--", label=f"prediction, seed {s}")
        ax.set_ylabel(f"$x_{c + 1}$", fontsize=13, rotation=0, labelpad=14)
        ax.grid(axis="x", visible=False)
    axes[0].legend(ncol=4, loc="upper right", fontsize=9, bbox_to_anchor=(1.0, 1.35))
    ax = axes[n]
    for t in range(t_max):
        ax.axvspan(t, t + 1, color=SHADE[R[t]], lw=0)
    for s, p in zip(seeds, preds):
        ax.plot(t_next, ((p[:t_max] - X[1:t_max + 1]) ** 2).mean(-1), color=SEED_COL[s % 3], lw=1.3,
                label=f"seed {s}")
    ax.plot(t_next, ((bay[:t_max] - X[1:t_max + 1]) ** 2).mean(-1), color=INK2, lw=1.2, ls=":",
            label="Bayes predictor")
    sw = np.nonzero(R[1:t_max] != R[:t_max - 1])[0] + 1
    for t in sw:
        for a in axes:
            a.axvline(t + 1, color=INK2, lw=0.8, ls="-", alpha=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("squared\nerror", fontsize=11)
    ax.set_xlabel("time step  (prediction made at t is drawn at t+1;  vertical lines = first state after a switch)")
    ax.legend(ncol=4, fontsize=9, loc="upper right")
    ax.set_xlim(0, t_max)
    fig.suptitle(f"{ds}: {STRUCT_LABEL.get(structure, structure)} one-step predictions vs the true state, "
                 f"test sequence {idx}; background = hidden regime", fontsize=12, fontweight="bold")
    fig.tight_layout()
    name = f"dyn_pred_{ds}_{structure}" + ("" if balance == "none" else f"_{balance}")
    return _save(fig, name, out)


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["moh", "dense"]):
        print(fig_prediction(s))
