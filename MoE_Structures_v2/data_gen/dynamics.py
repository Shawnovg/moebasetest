"""Switching linear dynamics: the state-transition task from the first project design.

Ported from D:/HarunoKen/Twilight_MoE_Dynamics/src/twilight_moe/dynamics.py (generator unchanged):

    x[t+1] = A[r[t]] x[t] + eps,   eps ~ N(0, noise_std^2 I),   x in R^4
    A[r] = 0.94 * blockdiag(Rot(a_r), Rot(b_r)),  (a_r, b_r) in {(0.2, 0.45), (1.1, -0.65), (-0.8, 1.35)}
    r[t] is a hidden 3-state Markov chain: stay with probability p_stay, else jump to one of the other two.

Each regime rotates the two coordinate planes (x1, x2) and (x3, x4) by its own pair of angles and shrinks
the state by 0.94 (stable under any switching). The model only sees the states, never the regime.

Datasets (all: 4-D states, noise 0.08, burn-in 128, sequences of 64 transitions):
* ``dyn``       p_stay 0.95 (one switch every ~20 steps, the Twilight default)
* ``dyn_fast``  p_stay 0.80 (one switch every ~5 steps: the regime must be re-inferred often)
* ``dyn_p50``   p_stay 0.50 (one switch every ~2 steps; added after the 5 Nov supervisor meeting)

Files: DATA_ROOT/<name>/{train,valid,test}.npz with ``states`` (N x 65 x 4, float32) and ``regimes``
(N x 64, int64; regimes[:, t] is the regime that maps states[:, t] -> states[:, t+1]) and manifest.json
(generator settings, the three matrices, training mean/std for normalisation, sha256 per file).

    python -m data_gen.dynamics [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backbone.paths import DATA_ROOT  # noqa: E402

ANGLES = [(0.2, 0.45), (1.1, -0.65), (-0.8, 1.35)]
CONTRACTION = 0.94

DYN_DATASETS = {
    "dyn": {"stay_probability": 0.95},
    "dyn_fast": {"stay_probability": 0.80},
    "dyn_p50": {"stay_probability": 0.50},
}
COMMON = {"length": 64, "burn_in": 128, "noise_std": 0.08,
          "n": {"train": 50_000, "valid": 1_000, "test": 2_000}}
SPLIT_SEED = {"train": 101, "valid": 102, "test": 103}


def transition_matrices():
    """Distinct rotations with a common contraction factor (Twilight, unchanged)."""
    mats = []
    for a, b in ANGLES:
        m = np.zeros((4, 4))
        for start, ang in [(0, a), (2, b)]:
            c, s = np.cos(ang), np.sin(ang)
            m[start:start + 2, start:start + 2] = [[c, -s], [s, c]]
        mats.append(CONTRACTION * m)
    return np.asarray(mats, dtype=np.float32)


def generate_sequences(n, length, burn_in, noise_std, stay_probability, seed):
    """Twilight ``generate_sequences``, unchanged: states n x (length+1) x 4, regimes n x length."""
    rng = np.random.default_rng(seed)
    matrices = transition_matrices()
    transition = np.full((3, 3), (1 - stay_probability) / 2)
    np.fill_diagonal(transition, stay_probability)
    x = rng.normal(scale=noise_std / np.sqrt(1 - CONTRACTION ** 2), size=(n, 4))
    regime = rng.integers(0, 3, n)
    states = np.empty((n, length + 1, 4), np.float32)
    regimes = np.empty((n, length), np.int64)
    for t in range(burn_in + length):
        if t >= burn_in:
            states[:, t - burn_in] = x
            regimes[:, t - burn_in] = regime
        x = np.einsum("nij,nj->ni", matrices[regime], x)
        x += rng.normal(scale=noise_std, size=x.shape)
        regime = (rng.random(n)[:, None] > transition[regime].cumsum(axis=1)).sum(axis=1)
    states[:, -1] = x
    return states, regimes


def build(name, out: Path = DATA_ROOT):
    spec = {**COMMON, **DYN_DATASETS[name]}
    d = Path(out) / name
    d.mkdir(parents=True, exist_ok=True)
    man = {"dataset": name, "task": "dynamics", "generator": "Twilight_MoE_Dynamics dynamics.py (ported unchanged)",
           **{k: v for k, v in spec.items() if k != "n"}, "angles": ANGLES, "contraction": CONTRACTION,
           "matrices": transition_matrices().tolist(), "splits": {}}
    for split in ("train", "valid", "test"):
        seed = SPLIT_SEED[split] * 1000 + int(spec["stay_probability"] * 100)
        states, regimes = generate_sequences(spec["n"][split], spec["length"], spec["burn_in"],
                                             spec["noise_std"], spec["stay_probability"], seed)
        if split == "train":
            man["mean"] = states.mean(axis=(0, 1)).tolist()
            man["std"] = states.std(axis=(0, 1)).tolist()
        p = d / f"{split}.npz"
        np.savez_compressed(p, states=states, regimes=regimes)
        sw = (regimes[:, 1:] != regimes[:, :-1]).mean()
        man["splits"][split] = {"n": int(states.shape[0]), "seed": seed, "switch_rate": float(sw),
                                "regime_freq": np.bincount(regimes.reshape(-1), minlength=3).tolist(),
                                "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    (d / "manifest.json").write_text(json.dumps(man, indent=1))
    print(name, {s: (v["n"], round(v["switch_rate"], 3)) for s, v in man["splits"].items()}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA_ROOT))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    for name in a.only or DYN_DATASETS:
        build(name, Path(a.out))


if __name__ == "__main__":
    main()
