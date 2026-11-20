"""Preview figures for the DRAFT dyn3 dataset -> reports/figures/dyn3_preview/*.png"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_gen import dyn3 as D  # noqa: E402

OUT = ROOT / "reports" / "figures" / "dyn3_preview"
OUT.mkdir(parents=True, exist_ok=True)
COL = ["#2a78d6", "#e3a21a", "#c2477a"]   # SHORT, REPLAY, CHORD
INK, INK2 = "#14213d", "#5b6478"
LAB = ["SHORT (momentum)", "REPLAY (repeat 16 steps ago)", "CHORD (two-speed clock)"]
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 160, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(OUT / f"{name}.png")


def shade(ax, reg):
    t0 = 0
    for t in range(1, len(reg) + 1):
        if t == len(reg) or reg[t] != reg[t0]:
            ax.axvspan(t0, t, color=COL[reg[t0]], alpha=0.13, lw=0)
            t0 = t


def fig_examples():
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for i, rule in enumerate(("coin", "timer")):
        s, r = D.generate(8, rule=rule, seed=7)
        k = 3
        for j, (a, b, nm) in enumerate(((0, 1, "hand 1"), (2, 3, "hand 2"))):
            ax = axes[2 * i + j]
            shade(ax, r[k])
            ax.plot(s[k, :, a], color=INK, lw=1.4, label="x")
            ax.plot(s[k, :, b], color=INK2, lw=1.1, ls="--", label="y")
            ax.set_ylabel(nm, fontsize=11)
            ax.set_ylim(-3.6, 3.6)
            if j == 0:
                ax.set_title({"coin": "Switch rule = coin flip (stay with p = 0.95 each step)",
                              "timer": "Switch rule = timer (SHORT 10, REPLAY 20, CHORD 16 steps, ±2)"}[rule],
                             loc="left", fontsize=13, fontweight="bold", color=INK)
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.3) for c in COL] +
                   [plt.Line2D([], [], color=INK, lw=1.4), plt.Line2D([], [], color=INK2, ls="--")],
                   labels=LAB + ["x coordinate", "y coordinate"], ncol=5, fontsize=10, frameon=False,
                   loc="lower left", bbox_to_anchor=(0, 1.12))
    axes[-1].set_xlabel("time step")
    save(fig, "1_example_sequences")


def single_regime(k, n=400, length=256, seed=3):
    old = D.generate.__defaults__
    s, r = D.generate(n * 3, length=length, rule="coin", p_stay=1.0, seed=seed)
    return s[r[:, 0] == k][:n]


def fig_clock():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for k, ax in enumerate(axes):
        s = single_regime(k, n=5)[0]
        seg = s[100:100 + 40, :2]
        ax.plot(seg[:, 0], seg[:, 1], color=COL[k], lw=1.3, marker="o", ms=3.5)
        ax.scatter(*seg[0], color=INK, s=40, zorder=3)
        ax.annotate("start", seg[0], xytext=(6, 6), textcoords="offset points", fontsize=10, color=INK)
        ax.set_title(LAB[k], fontsize=12, fontweight="bold", color=INK, loc="left")
        ax.set_aspect("equal"); ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)
        ax.axhline(0, color="#dddddd", lw=0.8); ax.axvline(0, color="#dddddd", lw=0.8)
        ax.set_xlabel("hand 1, x"); ax.set_ylabel("hand 1, y")
    fig.suptitle("Hand 1 drawn as a moving point, 40 steps of each regime", x=0.01, ha="left", fontsize=13, color=INK)
    save(fig, "2_hand_paths")


def fig_spectrum():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    for k in range(3):
        s = single_regime(k)
        z = s[:, 32:, 0] + 1j * s[:, 32:, 1]
        F = np.abs(np.fft.fft(z - z.mean(1, keepdims=True), axis=1)) ** 2
        f = np.fft.fftfreq(z.shape[1])
        o = np.argsort(f)
        p = F.mean(0)[o]
        ax.plot(f[o], p / p.sum(), color=COL[k], lw=1.6, label=LAB[k])
    for m in range(-8, 9):
        ax.axvline(m / 16, color="#eeeeee", lw=0.8, zorder=0)
    ax.set_yscale("log"); ax.set_ylim(1e-5, 1)
    ax.set_xlabel("frequency (cycles per step; negative = turning clockwise)")
    ax.set_ylabel("share of power (log)")
    ax.set_title("Which rhythms each regime contains (hand 1)", loc="left", fontsize=13, fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=10)
    save(fig, "3_frequency")


def fig_lookback():
    s, r = D.generate(6000, rule="coin", seed=11)
    Ls = [1, 2, 3, 4, 6, 8, 12, 15, 16, 20, 24, 32, 40]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    out = {}
    for k in range(3):
        errs = []
        for L in Ls:
            X, Y = [], []
            for t in range(40, s.shape[1] - 1):
                m = r[:, t] == k
                X.append(s[m, t - L + 1:t + 1].reshape(m.sum(), -1)); Y.append(s[m, t + 1])
            X, Y = np.concatenate(X), np.concatenate(Y)
            h = len(X) // 2
            W = np.linalg.lstsq(np.c_[X[:h], np.ones(h)], Y[:h], rcond=None)[0]
            errs.append(float(((np.c_[X[h:], np.ones(len(X) - h)] @ W - Y[h:]) ** 2).mean()))
        out[k] = errs
        ax.plot(Ls, errs, color=COL[k], marker="o", ms=6, lw=2, label=LAB[k], markeredgecolor="white")
    ax.axvline(16, color=INK2, ls=":", lw=1)
    ax.text(16.4, ax.get_ylim()[1] * 0.97, "16 steps back", color=INK2, fontsize=10, va="top")
    ax.set_xlabel("how many past steps the predictor may look at")
    ax.set_ylabel("prediction error (MSE)")
    ax.set_title("Best linear predictor, regime known: each regime needs a different memory", loc="left",
                 fontsize=13, fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=10)
    save(fig, "4_lookback")
    return Ls, out


def fig_durations():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, rule in zip(axes, ("coin", "timer")):
        s, r = D.generate(3000, rule=rule, seed=5)
        for k in range(3):
            d = []
            for row in r:
                ch = np.flatnonzero(np.diff(row)) + 1
                b = np.r_[0, ch, len(row)]
                for i in range(1, len(b) - 2):           # complete segments only
                    if row[b[i]] == k:
                        d.append(b[i + 1] - b[i])
            ax.hist(d, bins=np.arange(1, 61), color=COL[k], alpha=0.55, label=LAB[k].split(" (")[0], density=True)
        ax.set_title({"coin": "coin flip: any length possible", "timer": "timer: length nearly fixed per regime"}[rule],
                     loc="left", fontsize=12, fontweight="bold", color=INK)
        ax.set_xlabel("how long a regime lasts (steps)")
    axes[0].set_ylabel("share of segments")
    axes[1].legend(frameon=False)
    save(fig, "5_durations")


if __name__ == "__main__":
    fig_examples(); fig_clock(); fig_spectrum(); L, e = fig_lookback(); fig_durations()
    for k in range(3):
        print(D.NAMES[k], dict(zip(L, np.round(e[k], 3))))
