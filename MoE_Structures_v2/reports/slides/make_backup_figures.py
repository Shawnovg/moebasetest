"""Figures for the 'backup' slides (data, settings, metrics explained simply). White background.

    python reports/slides/make_backup_figures.py      -> reports/slides/figures/bk_*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reports" / "slides"))
from analysis import figures as F  # noqa: E402
import make_slide_figures as S  # noqa: E402
from data_gen.vocab import VOCAB  # noqa: E402

OUT = S.OUT
INK, INK2 = F.INK, F.INK2
BLUE, BLUE_BG = "#2a78d6", "#e8f1fc"
GREY_BG, GREY_EDGE = "#f1f0ec", "#a8a7a2"
C = F.STRUCT_COLOR
REG = ["#2a78d6", "#e8743b", "#19a979"]          # regime colours A0, A1, A2
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                     "savefig.dpi": 200, "font.size": 12})


def save(fig, name):
    p = OUT / f"{name}.png"
    fig.savefig(p, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(p)


# ------------------------------------------------------------------ one example sequence per task
def _short(tok, task):
    if tok.startswith("<t:"):
        return "[" + tok[3:-1] + "]"
    if tok == "<sep>":
        return "|"
    if tok.startswith("n"):
        i = int(tok[1:])
        if task == "mqar":
            return f"k{i}" if i < 64 else f"v{i}"
        if task in ("copy", "rev"):
            return f"s{i}"
        return str(i)
    if tok.startswith("x"):
        return tok[1:]
    if tok.startswith("L"):
        return tok[1:]
    return tok


def fig_sequences():
    rows = []
    for task, idx in (("mqar", 0), ("fact", 0), ("mod", 0), ("copy", None), ("rev", None), ("logic", None)):
        z = np.load(ROOT / "_data" / task / "test.npz")
        if idx is None:                                  # pick a short example so it fits on one row
            idx = int(np.argmin(np.abs(z["lengths"] - (18 if task in ("copy", "rev") else 16))))
        n = int(z["lengths"][idx])
        toks = [VOCAB[i] for i in z["tokens"][idx][:n]]
        tg = [VOCAB[i] if i >= 0 else None for i in z["targets"][idx][:n]]
        rows.append((task, toks, tg, n, int(z["lengths"].min()), int(z["lengths"].max())))
    W = 50
    fig, ax = plt.subplots(figsize=(20, 9.6))
    bw, gap, rh = 0.36, 0.035, 1.45
    x0 = 3.3
    for r, (task, toks, tg, n, lo, hi) in enumerate(rows):
        y = -r * rh
        n_ans = sum(t == "<ans>" for t in toks)
        ln = f"{n} tokens" if lo == hi else f"{lo}–{hi} tokens (this one {n})"
        ax.text(0, y + 0.02, task, fontsize=17, fontweight="bold", va="center", color=INK)
        ax.text(0, y - 0.42, f"{ln}\n{n_ans} answer slot{'s' if n_ans > 1 else ''}", fontsize=11.5, va="center",
                color=INK2, linespacing=1.25)
        for i, (t, a) in enumerate(zip(toks, tg)):
            x = x0 + i * (bw + gap)
            ans = t == "<ans>"
            ax.add_patch(FancyBboxPatch((x, y - 0.22), bw, 0.44, boxstyle="round,pad=0,rounding_size=0.05",
                                        fc=BLUE_BG if ans else GREY_BG, ec=BLUE if ans else GREY_EDGE,
                                        lw=1.6 if ans else 0.8))
            s = "?" if ans else _short(t, task)
            fs = 8.5 if len(s) <= 3 else (7 if len(s) <= 5 else 6)
            ax.text(x + bw / 2, y, s, ha="center", va="center", fontsize=fs if not ans else 11,
                    color=BLUE if ans else INK, fontweight="bold" if ans else None)
            if a is not None:
                ax.text(x + bw / 2, y - 0.42, _short(a, task), ha="center", va="center", fontsize=7.5,
                        color=BLUE, fontweight="bold")
    ax.text(x0, 0.62, "input tokens (grey);  ? = <ans> slot the model must fill;  blue text below = the label "
                      "(the loss uses only these)", fontsize=12.5, color=INK2)
    ax.set_xlim(-0.1, x0 + W * (bw + gap) + 0.1)
    ax.set_ylim(-(len(rows) - 1) * rh - 0.75, 0.9)
    ax.axis("off")
    save(fig, "bk_sequences")


# ------------------------------------------------------------------ learning-rate schedule
def fig_lr():
    steps = np.arange(3001)
    lr = np.where(steps < 300, steps / 300, 0.5 * (1 + np.cos(np.pi * np.clip((steps - 300) / 2700, 0, 1)))) * 1e-3
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.plot(steps, lr * 1e3, color=BLUE, lw=2.6)
    ax.axvspan(0, 300, color="#fdf1e3")
    ax.text(150, 0.52, "warm-up\n0 → 1e-3\n(300 steps)", ha="center", fontsize=13, color="#9a5b00")
    ax.annotate("large steps: move fast", xy=(700, 0.97), xytext=(900, 1.08), fontsize=13, color=INK,
                arrowprops={"arrowstyle": "->", "color": INK2})
    ax.annotate("small steps: settle into the minimum", xy=(2750, 0.05), xytext=(1700, 0.32), fontsize=13,
                color=INK, arrowprops={"arrowstyle": "->", "color": INK2})
    ax.set_xlabel("training step  (1 step = 1 mini-batch of 64 sequences)", fontsize=13)
    ax.set_ylabel(r"learning rate ($\times10^{-3}$)", fontsize=13)
    ax.set_xlim(0, 3000); ax.set_ylim(0, 1.2)
    save(fig, "bk_lr")


# ------------------------------------------------------------------ RoPE
def _arrow(ax, ang_deg, col, lab, r=1.0, lw=3):
    a = np.deg2rad(ang_deg)
    ax.annotate("", xy=(r * np.cos(a), r * np.sin(a)), xytext=(0, 0),
                arrowprops={"arrowstyle": "-|>", "color": col, "lw": lw, "mutation_scale": 22})
    ax.text(1.42 * r * np.cos(a), 1.38 * r * np.sin(a), lab, ha="center", va="center", fontsize=14, color=col,
            fontweight="bold")


def fig_rope():
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), gridspec_kw={"width_ratios": [1, 1, 1.35]})
    q, k = "#d1495b", "#00798c"
    for ax, (pq, pk) in zip(axes[:2], ((2, 5), (12, 15))):
        ax.add_patch(Circle((0, 0), 1, fill=False, ec="#d6d5d0", lw=1.2))
        ax.axhline(0, color="#ecebe7", lw=1); ax.axvline(0, color="#ecebe7", lw=1)
        _arrow(ax, 30 * pq, q, f"query, pos {pq}\nturned {30 * pq}°")
        _arrow(ax, 30 * pk, k, f"key, pos {pk}\nturned {30 * pk}°")
        a0, a1 = np.deg2rad(30 * pq), np.deg2rad(30 * pk)
        t = np.linspace(a0, a1, 40)
        ax.plot(0.42 * np.cos(t), 0.42 * np.sin(t), color=INK, lw=1.8)
        ax.text(0.62 * np.cos((a0 + a1) / 2), 0.62 * np.sin((a0 + a1) / 2), "90°", fontsize=15, ha="center",
                va="center", fontweight="bold")
        ax.set_xlim(-2.0, 2.0); ax.set_ylim(-1.4, 1.95); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"positions {pq} and {pk}  (distance {pk - pq})", fontsize=15)
    ax = axes[2]
    pos = np.arange(0, 41)
    for i, col in zip((0, 3, 7, 15), ("#1b4f99", "#2a78d6", "#86b6f0", "#b9b8b3")):
        th = 10000 ** (-2 * i / 32)
        ax.plot(pos, np.rad2deg(pos * th), color=col, lw=2.4, label=f"pair {i + 1}:  {np.rad2deg(th):.2f}° per position")
    ax.set_xlabel("position", fontsize=13); ax.set_ylabel("rotation angle (degrees)", fontsize=13)
    ax.set_title("each head: 16 pairs, each turning at its own speed", fontsize=15)
    ax.legend(fontsize=12); ax.set_ylim(0, 800)
    save(fig, "bk_rope")


# ------------------------------------------------------------------ attention widths
def _box(ax, x, y, w, h, fc, ec, txt="", fs=11, col=INK, lw=1.2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fc, ec=ec, lw=lw, ls=ls))
    if txt:
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=col)


def fig_attn():
    """Each row: query heads (top, filled = used for this token) and the key/value heads they read (bottom)."""
    rows = [  # name, n_q, active q indices, kv groups (list of q indices per kv), block grouping, router?, colour key
        ("Dense (4 heads)", 4, [0, 1, 2, 3], [[0], [1], [2], [3]], None, False, "dense"),
        ("MoH", 8, [0, 2, 5, 6], [[i] for i in range(8)], None, True, "moh"),
        ("MHA-8", 8, list(range(8)), [[i] for i in range(8)], None, False, "mha8"),
        ("MoA", 8, [1, 3, 4, 6], [list(range(8))], None, True, "moa"),
        ("MQA-8", 8, list(range(8)), [list(range(8))], None, False, "mqa8"),
        ("MoMHA", 8, [2, 3, 6, 7], [[0, 2, 4, 6], [1, 3, 5, 7]], [[0, 1], [2, 3], [4, 5], [6, 7]], True, "momha"),
        ("GQA-8", 8, list(range(8)), [[0, 1, 2, 3], [4, 5, 6, 7]], None, False, "gqa8"),
    ]
    fig, ax = plt.subplots(figsize=(19, 11.5))
    qw, qh, qg = 0.62, 0.42, 0.14
    x0 = 3.6
    rowgap = [0, 1.75, 1.45, 1.75, 1.45, 1.75, 1.45]
    y = 0
    for (name, nq, act, kv, blocks, router, ck), dy in zip(rows, rowgap):
        y -= dy
        col = C[ck]
        ax.text(0, y + 0.1, name, fontsize=17, fontweight="bold", va="center", color=INK)
        sub = ("router picks 4 of 8 per token" if ck in ("moh", "moa") else
               "router picks 2 of 4 blocks per token" if ck == "momha" else
               "no router: all heads, every token")
        ax.text(0, y - 0.35, sub, fontsize=11.5, va="center", color=INK2)
        qx = [x0 + i * (qw + qg) for i in range(nq)]
        if blocks:
            for b in blocks:
                bx = qx[b[0]] - 0.05
                _box(ax, bx, y - 0.04, 2 * qw + qg + 0.1, qh + 0.12, "none", col, lw=1.4, ls="--")
        for i, x in enumerate(qx):
            on = i in act
            _box(ax, x, y + 0.02, qw, qh, col if on else "white", col, f"Q{i + 1}", fs=11,
                 col="white" if on else col, lw=1.4)
        # key/value heads
        kvx0 = x0 + 8 * (qw + qg) + 1.3
        ax.text(kvx0 - 0.65, y + 0.23, "reads", fontsize=11, color=INK2, ha="center", va="center")
        for j, grp in enumerate(kv):
            kx = kvx0 + j * (qw + 0.18)
            _box(ax, kx, y + 0.02, qw, qh, "#f1f0ec", "#7d7c78", f"KV{j + 1}", fs=10.5)
        nkv = len(kv)
        ax.text(kvx0 + nkv * (qw + 0.18) + 0.3, y + 0.23,
                f"{nkv} key/value head{'s' if nkv > 1 else ''}" + (" (own, one per query)" if nkv == nq else
                                                                    " shared by all" if nkv == 1 else
                                                                    f" shared ({nq // nkv} queries each)")
                + ("\n(1st head of every block → KV1, 2nd → KV2)" if blocks else "\n(Q1–Q4 → KV1, Q5–Q8 → KV2)" if nkv == 2 else ""),
                fontsize=12.5, va="center", color=INK)
        nact = len(act)
        ax.text(x0 + 8 * (qw + qg) - 0.1, y - 0.33, f"{nact} active × 32 = {nact * 32} dims", fontsize=11.5,
                ha="right", va="center", color=INK2)
    ax.text(x0, 0.85, "query heads  (filled = used for this token, outline = skipped)", fontsize=13, color=INK2)
    ax.set_xlim(-0.1, 19.5); ax.set_ylim(y - 0.7, 1.15); ax.axis("off")
    save(fig, "bk_attn_schematic")


# ------------------------------------------------------------------ dynamics: one step, and why shrink
def fig_dyn_step():
    from data_gen.dynamics import ANGLES
    x = np.array([0.30, 0.0, 0.0, 0.20])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8))
    for p, ax in enumerate(axes):
        v = x[2 * p:2 * p + 2]
        ax.add_patch(Circle((0, 0), np.linalg.norm(v), fill=False, ec="#d6d5d0", lw=1.1, ls="--"))
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 3,
                                                          "mutation_scale": 22})
        ax.text(*(v * 1.12 + np.array([0.05, -0.035])), r"$x_t$", fontsize=17, color=INK, ha="center")
        for k, (a, b) in enumerate(ANGLES):
            ang = (a, b)[p]
            R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            w = 0.94 * R @ v
            ax.annotate("", xy=w, xytext=(0, 0), arrowprops={"arrowstyle": "-|>", "color": REG[k], "lw": 2.6,
                                                              "mutation_scale": 20})
            ax.add_patch(Circle(w, 0.08, fc=REG[k], alpha=0.13, ec="none"))
            ax.text(*(w * 1.25), f"A{k}: {np.rad2deg(ang):+.0f}°", fontsize=13.5, color=REG[k], ha="center",
                    va="center", fontweight="bold")
        lim = 0.45
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.axhline(0, color="#ecebe7", lw=1, zorder=0); ax.axvline(0, color="#ecebe7", lw=1, zorder=0)
        ax.set_xlabel(["$x_1$", "$x_3$"][p], fontsize=15); ax.set_ylabel(["$x_2$", "$x_4$"][p], fontsize=15)
        ax.set_title(["plane 1:  $(x_1, x_2)$", "plane 2:  $(x_3, x_4)$"][p], fontsize=15)
        ax.grid(False)
    fig.text(0.5, -0.02, "arrows: where the state goes under each regime (turned, then × 0.94);  "
                         "shaded disc: noise, sd 0.08", ha="center", fontsize=13, color=INK2)
    save(fig, "bk_dyn_step")


def fig_rho():
    rng = np.random.default_rng(0)
    from data_gen.dynamics import ANGLES
    T, N = 300, 400
    fig, ax = plt.subplots(figsize=(13, 5.2))
    for rho, col, lab in ((1.0, "#d1495b", "no shrink  (ρ = 1)"), (0.94, BLUE, "ρ = 0.94  (ours)"),
                          (0.7, "#a8a7a2", "ρ = 0.7")):
        x = np.zeros((N, 4)); r = rng.integers(3, size=N); rms = []
        for t in range(T):
            a = np.array([ANGLES[i][0] for i in r]); b = np.array([ANGLES[i][1] for i in r])
            c, s, c2, s2 = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
            x = rho * np.stack([c * x[:, 0] - s * x[:, 1], s * x[:, 0] + c * x[:, 1],
                                c2 * x[:, 2] - s2 * x[:, 3], s2 * x[:, 2] + c2 * x[:, 3]], 1)
            x += rng.normal(0, 0.08, x.shape)
            sw = rng.random(N) > 0.95
            r = np.where(sw, (r + rng.integers(1, 3, N)) % 3, r)
            rms.append(np.sqrt((x ** 2).mean()))
        ax.plot(rms, color=col, lw=2.4, label=lab)
    ax.axhline(np.sqrt(0.0064 / (1 - 0.94 ** 2)), color=BLUE, ls=":", lw=1.5)
    ax.text(305, np.sqrt(0.0064 / (1 - 0.94 ** 2)), "  0.235 = √(0.08² / (1 − 0.94²))", va="center",
            fontsize=12, color=BLUE)
    ax.axhline(0.08, color="#a8a7a2", ls=":", lw=1.2)
    ax.text(305, 0.08, "  0.08 = noise size", va="center", fontsize=12, color=INK2)
    ax.set_xlabel("time step", fontsize=13); ax.set_ylabel("typical size of the state (rms)", fontsize=13)
    ax.set_xlim(0, 300); ax.set_ylim(0, 1.5)
    ax.legend(fontsize=13, loc="upper left")
    save(fig, "bk_rho")


# ------------------------------------------------------------------ Bayes floor on one real sequence
def fig_floor_seq():
    from analysis.dyn_predictions import DynamicsDataset, bayes, load_model, predict
    data = DynamicsDataset("dyn", "test")
    X, R = data.raw[0].numpy(), data.regimes[0].numpy()
    b = bayes(data, 0)
    m = predict(load_model("moh", "dyn", 0, balance="lb"), data, 0)
    t = np.arange(8, 64)
    sb = ((b[t] - X[t + 1]) ** 2).mean(-1)
    sm = ((m[t] - X[t + 1]) ** 2).mean(-1)
    fig, ax = plt.subplots(figsize=(17, 5.4))
    for i in range(len(R)):
        ax.axvspan(i - 0.5, i + 0.5, color=REG[R[i]], alpha=0.09, lw=0)
    ax.bar(t - 0.2, sb, width=0.4, color="#14213d", label="Bayes predictor (knows the last regime)")
    ax.bar(t + 0.2, sm, width=0.4, color=C["moh"], label="MoH (trained model)")
    ax.axhline(sb.mean(), color="#14213d", ls="--", lw=1.4)
    ax.axhline(sm.mean(), color=C["moh"], ls=":", lw=1.8)
    ax.text(64, sb.mean(), f"  mean {sb.mean():.4f}", va="top", fontsize=12.5, color="#14213d")
    ax.text(64, sm.mean(), f"  mean {sm.mean():.4f}", va="bottom", fontsize=12.5, color=C["moh"])
    for s in np.where(R[1:] != R[:-1])[0] + 1:
        if s >= 8:
            ax.annotate("regime switch", xy=(s, ((b[s] - X[s + 1]) ** 2).mean() + 0.003), xytext=(s, 0.128),
                        ha="center", fontsize=12, arrowprops={"arrowstyle": "->", "color": INK2})
    ax.set_xlabel("position t  (error of predicting $x_{t+1}$; positions 0–7 are not scored)", fontsize=13)
    ax.set_ylabel("squared error, mean of 4 coords", fontsize=13)
    ax.set_xlim(7.3, 70); ax.set_ylim(0, 0.14)
    ax.legend(fontsize=12.5, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    for k in range(3):
        ax.text(28 + k * 3.2, 0.134, f"A{k}", color=REG[k], fontsize=13, fontweight="bold", va="center",
                bbox={"fc": REG[k], "alpha": 0.15, "ec": "none"})
    ax.text(37.5, 0.134, "background = hidden regime", fontsize=12, color=INK2, va="center")
    save(fig, "bk_floor_seq")
    return dict(bayes=float(sb.mean()), moh=float(sm.mean()))


# ------------------------------------------------------------------ guessing tree for entropy
def fig_guess():
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.6))
    def node(ax, x, y, txt, fc="#f1f0ec", ec="#7d7c78", fs=13, w=1.9):
        _box(ax, x - w / 2, y - 0.3, w, 0.6, fc, ec, txt, fs=fs)
    def edge(ax, a, b, lab):
        ax.plot([a[0], b[0]], [a[1] - 0.3, b[1] + 0.3], color=INK2, lw=1.3)
        ax.text((a[0] + b[0]) / 2 + (0.12 if b[0] > a[0] else -0.12), (a[1] + b[1]) / 2, lab, fontsize=12,
                color=INK2, ha="left" if b[0] > a[0] else "right")
    ax = axes[0]
    node(ax, 0, 2, "first half?")
    node(ax, -2, 0.6, "is it 1?"); node(ax, 2, 0.6, "is it 3?")
    for x, lab in zip((-3, -1, 1, 3), "1234"):
        node(ax, x, -0.8, f"option {lab}", fc=BLUE_BG, ec=BLUE, w=1.5)
    edge(ax, (0, 2), (-2, 0.6), "yes"); edge(ax, (0, 2), (2, 0.6), "no")
    for p, (a, b) in (((-2, 0.6), (-3, -1)), ((2, 0.6), (1, 3))):
        edge(ax, p, (a, -0.8), "yes"); edge(ax, p, (b, -0.8), "no")
    ax.set_title("4 equally likely options: always 2 questions = log₂ 4", fontsize=15)
    ax.set_xlim(-4, 4); ax.set_ylim(-1.3, 2.5); ax.axis("off")
    ax = axes[1]
    node(ax, 0, 2, "is it A0?")
    node(ax, -1.8, 0.6, "A0", fc=BLUE_BG, ec=BLUE, w=1.3); node(ax, 1.8, 0.6, "is it A1?")
    node(ax, 0.8, -0.8, "A1", fc=BLUE_BG, ec=BLUE, w=1.3); node(ax, 2.8, -0.8, "A2", fc=BLUE_BG, ec=BLUE, w=1.3)
    edge(ax, (0, 2), (-1.8, 0.6), "yes  (⅓)"); edge(ax, (0, 2), (1.8, 0.6), "no  (⅔)")
    edge(ax, (1.8, 0.6), (0.8, -0.8), "yes"); edge(ax, (1.8, 0.6), (2.8, -0.8), "no")
    ax.text(-1.8, -0.15, "1 question", ha="center", fontsize=12.5, color=BLUE)
    ax.text(1.8, -1.5, "2 questions", ha="center", fontsize=12.5, color=BLUE)
    ax.set_title("3 equally likely regimes: ⅓·1 + ⅔·2 = 1.67 questions", fontsize=15)
    ax.set_xlim(-4, 4); ax.set_ylim(-1.8, 2.5); ax.axis("off")
    save(fig, "bk_guess")


EQS = {
    "bk_eq_rope": r"$\mathrm{score}(m,n)=\left(R_{m\theta}\,q\right)^{\!\top}\!\left(R_{n\theta}\,k\right)"
                  r"=q^{\top}R_{(n-m)\theta}\,k\qquad\text{depends only on the distance } n-m$",
    "bk_eq_mse": r"$\displaystyle \mathrm{MSE}=\frac{1}{N}\sum_{\text{sequences}}\ \frac{1}{56}\sum_{t=8}^{63}\ "
                 r"\frac{1}{4}\sum_{i=1}^{4}\left(\hat x_{t+1,i}-x_{t+1,i}\right)^{2}$",
    "bk_eq_bayes": r"$\displaystyle \hat x^{\,\mathrm{Bayes}}_{t+1}=\sum_{k=0}^{2}P(r_t=k\mid r_{t-1})\;"
                   r"\rho A_k x_t,\qquad P(\text{stay})=0.95,\ \ P(\text{each other})=0.025$",
    "bk_eq_entropy": r"$\displaystyle H(R)=\sum_r p(r)\,\log_2\frac{1}{p(r)}\quad"
                     r"\xrightarrow{\ p=\frac13\ \text{each}\ }\quad 3\times\tfrac13\log_2 3=\log_2 3=1.585\ \text{bits}$",
    "bk_eq_mi": r"$\displaystyle I(U;R)=H(R)-H(R\mid U),\qquad \mathrm{MI}_{\mathrm{norm}}=\frac{I(U;R)}{H(R)}"
                r"=\frac{I(U;R)}{\log 3}\in[0,1]$",
    "bk_eq_dyn": r"$x_{t+1}=\underbrace{0.94}_{\text{shrink}}\ \cdot\underbrace{A_{r_t}}_{\text{turn}}\,x_t\ +"
                 r"\underbrace{\varepsilon_t}_{\text{noise, sd }0.08}$",
    "bk_eq_rho": r"$\mathrm{Var}(x)=\rho^2\,\mathrm{Var}(x)+\sigma^2\ \ \Rightarrow\ \ "
                 r"\mathrm{Var}(x)=\frac{\sigma^2}{1-\rho^2}=\frac{0.0064}{1-0.94^2}=0.055$",
}


def equations():
    S.EQS = EQS
    S.equations()


if __name__ == "__main__":
    which = sys.argv[1:] or ["eq", "seq", "lr", "rope", "attn", "dyn", "rho", "floor", "guess"]
    fns = {"eq": equations, "seq": fig_sequences, "lr": fig_lr, "rope": fig_rope, "attn": fig_attn,
           "dyn": fig_dyn_step, "rho": fig_rho, "floor": fig_floor_seq, "guess": fig_guess}
    for w in which:
        r = fns[w]()
        if r:
            print(json.dumps(r))
