"""Figures for the lab-meeting deck (white background), written to reports/slides/figures/.

    python reports/slides/make_slide_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis import figures as F  # noqa: E402  (sets the house style)
from analysis import dynamics as D  # noqa: E402

OUT = ROOT / "reports" / "slides" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "reports" / "tables"
SRC = ROOT / "reports" / "figures"
WHITE = "#ffffff"
plt.rcParams.update({"figure.facecolor": WHITE, "axes.facecolor": WHITE, "savefig.facecolor": WHITE,
                     "savefig.dpi": 200, "font.size": 11, "axes.titlesize": 12})
C, L = F.STRUCT_COLOR, F.STRUCT_LABEL
SIX = ["dense", "dense_wide", "moe", "moh", "moa", "momha"]
ROUTED = ["moe", "moh", "moa", "momha"]
ATT = False   # True: attention structures only (no MoE / Dense-wide), after the 5 Nov supervisor meeting
INK, INK2, GRID = F.INK, F.INK2, F.GRID
ACC = "#2a78d6"


def save(fig, name):
    p = OUT / f"{name}.png"
    fig.savefig(p, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(p)


def recolor(name, out_name=None):
    """Copy an existing figure with its off-white background turned pure white."""
    im = np.array(Image.open(SRC / f"{name}.png").convert("RGB")).astype(int)
    bg = (np.abs(im - np.array([252, 252, 251])).sum(-1) <= 3)
    im[bg] = 255
    Image.fromarray(im.astype(np.uint8)).save(OUT / f"{out_name or name}.png")
    print(OUT / f"{out_name or name}.png")


# ----------------------------------------------------------------- equations (real LaTeX)
EQS = {
    "eq_backbone": r"$h' = h + \mathrm{Attn}(\mathrm{LN}(h)),\qquad y = h' + \mathrm{FFN}(\mathrm{LN}(h'))$",
    "eq_moe": r"$y=\sum_{i\in \mathrm{Top}_2(g)} g_i(x)\,E_i(x),\qquad g(x)=\mathrm{softmax}(W_g x),\ \ E_i:\ \mathrm{FFN},\ N=8$",
    "eq_moh": r"$\mathrm{MoH}(x)=\sum_{h=1}^{8} \tilde g_h(x)\,\mathrm{head}_h(x)\,W_O^{h},\qquad \tilde g_h=0\ \ \mathrm{if}\ h\notin\mathrm{Top}_4(g)$",
    "eq_moa": r"$\mathrm{MoA}(x)=\sum_{i\in\mathrm{Top}_4(g)} g_i(x)\,\mathrm{Attn}\!\left(xW_Q^{i},\,XW_K,\,XW_V\right)W_O^{i},\qquad W_K,W_V\ \mathrm{shared}$",
    "eq_momha": r"$\mathrm{MoMHA}(x)=\sum_{j\in\mathrm{Top}_2(g)} g_j(x)\,\mathrm{MHA}_j(x),\qquad 4\ \mathrm{blocks}\times 2\ \mathrm{heads}$",
    "eq_dyn": r"$x_{t+1}=\rho\,A_{r_t}\,x_t+\varepsilon_t,\quad A_r=\begin{pmatrix}R(a_r)&0\\0&R(b_r)\end{pmatrix},\quad \rho=0.94,\quad \varepsilon_t\sim\mathcal{N}(0,\,0.08^2 I_4)$",
    "eq_markov": r"$P(r_{t+1}=r_t)=p_{\mathrm{stay}},\qquad p_{\mathrm{stay}}=0.95\ (\texttt{dyn}),\ \ 0.80\ (\texttt{dyn\_fast}),\ \ 0.50\ (\texttt{dyn\_p50})$",
    "eq_loss": r"$\hat x_{t+1}=f_\theta(x_0,\dots,x_t),\qquad \mathcal{L}=\frac{1}{T}\sum_{t=0}^{T-1}\left\lVert \hat x_{t+1}-x_{t+1}\right\rVert_2^2$",
    "eq_excess": r"$\mathrm{excess}=\mathrm{MSE}_{\mathrm{model}}-\mathrm{MSE}_{\mathrm{Bayes}},\qquad \hat x^{\mathrm{Bayes}}_{t+1}=\sum_{k} P(r_t=k\mid r_{t-1})\,\rho A_k x_t$",
    "eq_mi_full": r"$\displaystyle I(U;R)=\sum_{u,r} p(u,r)\,\log\frac{p(u,r)}{p(u)\,p(r)}=H(R)-H(R\mid U),\qquad \mathrm{MI}_{\mathrm{norm}}=\frac{I(U;R)}{\log 3}\in[0,1]$",
    "eq_mi": r"$\mathrm{MI}_{\mathrm{norm}}=\frac{I(\mathrm{unit};\,r_{t-1})}{\log 3}\ \in[0,1]$",
}


def equations():
    """Real LaTeX: standalone + pdflatex, rasterised with pdftoppm (white background)."""
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for name, tex in EQS.items():
            doc = ("\\documentclass[border=4pt,varwidth=60cm]{standalone}\\usepackage{amsmath,amssymb}"
                   "\\begin{document}\\Large " + tex + "\\end{document}")
            (Path(tmp) / f"{name}.tex").write_text(doc)
            subprocess.run(["pdflatex", "-interaction=nonstopmode", f"{name}.tex"], cwd=tmp, check=True,
                           stdout=subprocess.DEVNULL)
            subprocess.run(["pdftoppm", "-png", "-r", "400", "-singlefile", f"{name}.pdf", str(OUT / name)],
                           cwd=tmp, check=True)
            print(OUT / f"{name}.png")


# ----------------------------------------------------------------- memory tasks
def _tok(ax, x, y, text, kind, w=None):
    styles = {"p": ("#f1f0ec", "#8a8985", INK), "sep": ("#ffffff", "#bdbcb7", INK2),
              "ans": ("#e8f1fc", ACC, ACC), "tag": ("#ffffff", "#bdbcb7", INK2)}
    fc, ec, tc = styles[kind]
    w = w or max(0.62, 0.17 * len(text) + 0.25)
    ax.add_patch(FancyBboxPatch((x, y - 0.32), w, 0.64, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.6 if kind == "ans" else 1.0))
    ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=12, color=tc,
            fontweight="bold" if kind == "ans" else "normal")
    return x + w + 0.12


def fig_memory_tasks():
    rows = [
        ("mqar", [("k3", "p"), ("v71", "p"), ("k9", "p"), ("v80", "p"), ("k5", "p"), ("v66", "p"), ("…", "sep"),
                  ("|", "sep"), ("k9", "p"), ("v80", "ans"), ("k3", "p"), ("v71", "ans")]),
        ("fact", [("Ann", "p"), ("Lee", "p"), ("born-in", "p"), ("v57", "ans"), ("Bo", "p"), ("Kim", "p"),
                  ("job", "p"), ("v12", "ans")]),
        ("mod", [("23", "p"), ("+", "p"), ("81", "p"), ("=", "p"), ("7", "ans"), ("40", "p"), ("×", "p"),
                 ("5", "p"), ("=", "p"), ("6", "ans")]),
        ("copy", [("a", "p"), ("c", "p"), ("f", "p"), ("b", "p"), ("d", "p"), ("|", "sep"), ("a", "ans"),
                  ("c", "ans"), ("f", "ans"), ("b", "ans"), ("d", "ans")]),
        ("rev", [("a", "p"), ("c", "p"), ("f", "p"), ("b", "p"), ("d", "p"), ("|", "sep"), ("d", "ans"),
                 ("b", "ans"), ("f", "ans"), ("c", "ans"), ("a", "ans")]),
        ("logic", [("(", "p"), ("not", "p"), ("p3", "p"), (")", "p"), ("or", "p"), ("p1", "p"), ("|", "sep"),
                   ("p3", "p"), ("and", "p"), ("p1", "p"), ("⊐", "ans")]),
        ("mix", [("[fact]", "tag"), ("Ann", "p"), ("Lee", "p"), ("job", "p"), ("v12", "ans"), ("‖", "sep"),
                 ("[rev]", "tag"), ("a", "p"), ("c", "p"), ("|", "sep"), ("c", "ans"), ("a", "ans")]),
    ]
    probe = ["in-context key→value recall", "facts stored in the weights (8,100 entities × 4 relations)",
             "(a op b) mod 97, half of (a, b) pairs held out", "copy a string of 8–24 symbols",
             "write the string reversed", "logical relation between two formulas (7 classes)",
             "one sequence per task, task tag first (all six)"]
    fig, ax = plt.subplots(figsize=(15, 6.6))
    ax.set_xlim(-1.9, 17.5); ax.set_ylim(-1.0, len(rows) - 0.3); ax.axis("off")
    for i, (name, toks) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.text(-1.75, y, name, ha="left", va="center", fontsize=15, fontweight="bold", color=INK)
        x = 0
        for t, k in toks:
            x = _tok(ax, x, y, t, k)
        ax.text(10.9, y, probe[i], ha="left", va="center", fontsize=11.5, color=INK2)
    ax.add_patch(FancyBboxPatch((0, -0.92), 0.5, 0.3, boxstyle="round,pad=0.02", fc="#f1f0ec", ec="#8a8985"))
    ax.text(0.62, -0.77, "prompt token", va="center", fontsize=11, color=INK2)
    ax.add_patch(FancyBboxPatch((2.6, -0.92), 0.5, 0.3, boxstyle="round,pad=0.02", fc="#e8f1fc", ec=ACC, lw=1.6))
    ax.text(3.22, -0.77, "answer slot (target shown)", va="center", fontsize=11, color=INK2)
    ax.text(6.6, -0.77, "bidirectional encoder, loss on answer slots only", va="center", fontsize=11, color=INK2)
    save(fig, "memory_tasks")


def _acc():
    d = pd.read_csv(TAB / "acc_by_config.csv")
    return d[d.structure.isin(SIX)]


def fig_memory_heatmap():
    d = _acc()
    d = d[(d.variant == "default") & (d.balance == "none")]
    m = d.pivot(index="structure", columns="dataset", values="mean").loc[SIX, F.DS_ORDER]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    im = ax.imshow(m.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=13, color="white" if v > 0.6 else INK)
    ax.set_xticks(range(m.shape[1]), m.columns, fontsize=13)
    ax.set_yticks(range(m.shape[0]), [L[s] for s in m.index], fontsize=13)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("test accuracy", fontsize=12); cb.outline.set_visible(False)
    ax.tick_params(length=0)
    save(fig, "memory_heatmap")


def fig_fact_mod_balance():
    d = _acc()
    d = d[d.variant == "default"]
    bal = [("none", "no balancing", "#8a8985"), ("lb", "load-balance loss", "#1b4f99"), ("warmup", "random warm-up", "#86b6f0")]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
    for ax, ds in zip(axes, ["fact", "mod"]):
        x = d[d.dataset == ds]
        for i, s in enumerate(SIX):
            for j, (b, lab, col) in enumerate(bal):
                r = x[(x.structure == s) & (x.balance == b)]
                if not len(r):
                    continue
                off = 0 if s in ("dense", "dense_wide") else (j - 1) * 0.27
                ax.bar(i + off, r["mean"].iloc[0], width=0.25, yerr=r["std"].iloc[0], capsize=2.5, color=col,
                       label=lab if (i == 2) else None, error_kw={"lw": 1, "ecolor": INK2})
        ax.set_xticks(range(len(SIX)), [L[s] for s in SIX], fontsize=12)
        ax.set_title(ds, fontsize=15)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("test accuracy", fontsize=12)
    axes[0].legend(loc="upper left", fontsize=11, ncol=3, bbox_to_anchor=(0, 1.0))
    save(fig, "memory_balance")


def fig_fact_ablation(balance="none"):
    d0 = _acc()
    d0 = d0[d0.dataset == "fact"]
    d = d0[d0.balance == balance]
    if ATT:   # grouped by the number of active heads, so MoH / MoA / MoMHA are compared at the same width
        rows = [("g4", "moh", "default", "4 of 8 heads (default)"), ("g4", "moa", "default", "4 of 8 heads (default)"),
                ("g4", "momha", "default", "2 of 4 blocks × 2 heads (default)"),
                ("g6", "moh", "k6", "6 of 8 heads"), ("g6", "moa", "k6", "6 of 8 heads"), ("g6", "momha", "k3", "3 of 4 blocks × 2 heads"),
                ("g8", "moh", "k8", "8 of 8 heads"), ("g8", "moa", "k8", "8 of 8 heads"), ("g8", "momha", "k4", "4 of 4 blocks × 2 heads")]
        rows = [(g, s, v, lab) for g, s, v, lab in rows if len(d0[(d0.structure == s) & (d0.variant == v)])]
        gname = {"g4": "4 active heads", "g6": "6 active heads", "g8": "8 active heads (all)"}
    else:
        rows = [(s, s, v, lab) for s, v, lab in [
            ("moe", "default", "top-2 of 8 (default)"), ("moe", "top1", "top-1"), ("moe", "hash", "hash routing"),
            ("moh", "default", "4 of 8 heads (default)"), ("moh", "k6", "6 of 8 heads"), ("moh", "shared2", "2 shared + 2 of 6"),
            ("moa", "default", "4 of 8 heads (default)"), ("moa", "k8", "8 of 8 heads"), ("moa", "renorm_detached", "detached renorm"),
            ("momha", "default", "top-2 of 4 blocks (default)"), ("momha", "top1", "top-1")]]
        gname = {}
    fig, ax = plt.subplots(figsize=(13, 0.5 * len(rows) + 1.6))
    y, ticks, labs, prev = 0, [], [], None
    for g, s, v, lab in rows:
        if g != prev and g in gname:
            if prev is not None:
                y += 0.4
            ax.text(-0.01, y - 0.15, gname[g], transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=12.5, fontweight="bold", color=INK)
            y += 0.9
        elif prev is not None and g != prev:
            y += 0.6
        r = d[(d.structure == s) & (d.variant == v)]
        if not len(r):                       # hash routing is fixed: only run without balancing
            r = d0[(d0.structure == s) & (d0.variant == v) & (d0.balance == "none")]
            lab = lab + " (fixed, no router)"
        r = r.iloc[0]
        ax.barh(y, r["mean"], xerr=r["std"], height=0.7, color=C[s], alpha=1.0 if v == "default" else 0.55,
                capsize=3, error_kw={"lw": 1, "ecolor": INK2})
        ax.text(r["mean"] + r["std"] + 0.012, y, f"{r['mean']:.2f}", va="center", fontsize=11.5, color=INK)
        ticks.append(y); labs.append(f"{L[s]}  ·  {lab}")
        y += 1; prev = g
    for s, ls in ((("dense", "--"), ("mha8", ":")) if ATT else (("dense", "--"), ("dense_wide", ":"))):
        if s == "mha8":
            ab = pd.read_csv(TAB / "attn_baselines.csv")
            v = ab[(ab.dataset == "fact") & (ab.structure == "mha8")]["mean"].iloc[0]
        else:
            v = d0[(d0.structure == s) & (d0.variant == "default") & (d0.balance == "none")]["mean"].iloc[0]
        ax.axvline(v, color=C[s], ls=ls, lw=1.8, label=f"{L[s]} {v:.2f}")
    ax.set_yticks(ticks, labs, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    blab = {"none": "no balancing", "lb": "load-balance loss", "warmup": "random warm-up"}[balance]
    ax.set_xlabel(f"fact test accuracy ({blab}, mean ± sd over seeds)", fontsize=12)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(axis="y", visible=False)
    save(fig, "memory_ablation_fact" + ("" if balance == "none" else f"_{balance}"))


# ----------------------------------------------------------------- dynamics
REG_COL = ["#2a78d6", "#eb6834", "#1baf7a"]


def fig_markov():
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.2, 2.6); ax.axis("off"); ax.set_aspect("equal")
    pos = [(0, 1.3), (-1.3, -0.95), (1.3, -0.95)]
    for k, (x, y) in enumerate(pos):
        ax.add_patch(Circle((x, y), 0.48, fc=REG_COL[k], ec="none", alpha=0.18))
        ax.add_patch(Circle((x, y), 0.48, fc="none", ec=REG_COL[k], lw=2.2))
        ax.text(x, y, f"$A_{k}$", ha="center", va="center", fontsize=24, color=INK)
    for a in range(3):
        for b in range(3):
            if a == b:
                continue
            (x1, y1), (x2, y2) = pos[a], pos[b]
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), connectionstyle="arc3,rad=0.18", arrowstyle="-|>",
                                         mutation_scale=16, lw=1.3, color=INK2, shrinkA=30, shrinkB=30))
    for k, (x, y) in enumerate(pos):
        ang = [90, 210, 330][k]
        for_ = np.deg2rad(ang - 28); to_ = np.deg2rad(ang + 28)
        p1 = (x + 0.5 * np.cos(for_), y + 0.5 * np.sin(for_)); p2 = (x + 0.5 * np.cos(to_), y + 0.5 * np.sin(to_))
        ax.add_patch(FancyArrowPatch(p1, p2, connectionstyle="arc3,rad=1.8", arrowstyle="-|>",
                                     mutation_scale=16, lw=1.3, color=INK2))
        lx, ly = x + 1.05 * np.cos(np.deg2rad(ang)), y + 1.05 * np.sin(np.deg2rad(ang))
        ax.text(lx, ly, r"$p_{\mathrm{stay}}$", ha="center", va="center", fontsize=17, color=INK)
    ax.text(0, -0.55, r"$\frac{1-p_{\mathrm{stay}}}{2}$", ha="center", fontsize=18, color=INK)
    save(fig, "dyn_markov")


def fig_causal():
    T = 8
    fig = plt.figure(figsize=(15, 4.4))
    ax = fig.add_axes([0.0, 0.0, 0.66, 1.0]); ax.axis("off")
    ax.set_xlim(-0.6, T + 0.6); ax.set_ylim(-0.9, 3.2)
    names = ["0", "1", "2", r"\dots", "t", r"\dots", "62", "63"]
    outs = ["1", "2", "3", r"\dots", "t+1", r"\dots", "63", "64"]
    ax.add_patch(FancyBboxPatch((-0.45, 0.95), T - 0.1, 0.9, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="#f6f5f2", ec="#bdbcb7"))
    ax.text((T - 1) / 2 - 0.05, 1.4, "2-layer encoder, causal attention (structure under test)", ha="center",
            va="center", fontsize=13, color=INK)
    for i in range(T):
        dots = names[i] == r"\dots"
        if not dots:
            ax.add_patch(FancyBboxPatch((i - 0.32, -0.25), 0.64, 0.6, boxstyle="round,pad=0.02", fc="#f1f0ec",
                                        ec="#8a8985"))
            ax.add_patch(FancyBboxPatch((i - 0.38, 2.42), 0.76, 0.6, boxstyle="round,pad=0.02", fc="#e8f1fc",
                                        ec=ACC, lw=1.5))
            ax.annotate("", (i, 0.95), (i, 0.37), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2))
            ax.annotate("", (i, 2.42), (i, 1.85), arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1.2))
        ax.text(i, 0.05, f"$x_{{{names[i]}}}$" if not dots else r"$\dots$", ha="center", va="center", fontsize=17)
        ax.text(i, 2.72, f"$\\hat x_{{{outs[i]}}}$" if not dots else r"$\dots$", ha="center", va="center",
                fontsize=17, color=ACC)
    ax.text(-0.45, -0.7, r"input: states $x_0..x_{63}\in\mathbb{R}^4$ (linear $4\to128$)", fontsize=13, color=INK2)
    ax.text(4.3, -0.7, r"target at position $t$: next state $x_{t+1}$", fontsize=13, color=INK2)
    bx = fig.add_axes([0.71, 0.12, 0.26, 0.8])
    m = np.tril(np.ones((T, T)))
    bx.imshow(m, cmap="Blues", vmin=-0.3, vmax=1.6)
    bx.set_xticks(range(T), [f"${n}$" for n in names], fontsize=11)
    bx.set_yticks(range(T), [f"${n}$" for n in names], fontsize=11)
    bx.set_xlabel("key position (attended)", fontsize=12); bx.set_ylabel("query position", fontsize=12)
    bx.set_title("causal mask", fontsize=13); bx.grid(False); bx.tick_params(length=0)
    for s in bx.spines.values():
        s.set_visible(False)
    save(fig, "dyn_causal")


def fig_matched_k(main):
    d = main[main.dataset == "dyn"]
    get = lambda s, v: d[(d.structure == s) & (d.variant == v)].iloc[0]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for s, vs in (("moh", ["default", "k6", "k8"]), ("moa", ["default", "k6", "k8"]), ("momha", ["default", "k3", "k4"])):
        if not all(len(d[(d.structure == s) & (d.variant == v)]) for v in vs):
            continue
        e = np.array([get(s, v).excess for v in vs]) * 1e3
        sd = np.array([get(s, v).excess_sd for v in vs]) * 1e3
        ax.errorbar([4, 6, 8], e, yerr=sd, fmt="o-", color=C[s], lw=2.2, ms=8, capsize=4, label=L[s])
    for s, ls in ((("dense", "--"), ("mha8", ":")) if ATT else (("dense", "--"), ("dense_wide", ":"))):
        ax.axhline(get(s, "default").excess * 1e3, color=C[s], ls=ls, lw=1.8, label=L[s])
    ax.set_xticks([4, 6, 8], ["4 of 8\n(default)", "6 of 8", "8 of 8\n(all heads)"], fontsize=12)
    ax.set_ylabel(r"MSE $-$ Bayes floor  ($\times 10^{-3}$)", fontsize=12)
    ax.set_xlabel("active heads per token", fontsize=12)
    ax.legend(fontsize=11, ncol=2, loc="upper left")
    ax.set_xlim(3.4, 8.6)
    save(fig, "dyn_matched_k")


def fig_dyn_variants_other(main):
    d = main[main.dataset == "dyn"]
    rows = ([] if ATT else [("moe", "default", "top-2 (default)"), ("moe", "top1", "top-1")]) + [
            ("moa", "default", "default"), ("moa", "renorm_detached", "detached renorm"),
            ("momha", "default", "top-2 (default)"), ("momha", "top1", "top-1")]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for i, (s, v, lab) in enumerate(rows):
        r = d[(d.structure == s) & (d.variant == v)].iloc[0]
        ax.barh(i + (i // 2) * 0.4, r.excess * 1e3, xerr=r.excess_sd * 1e3, height=0.7, color=C[s],
                alpha=1.0 if v == "default" else 0.55, capsize=3, error_kw={"lw": 1, "ecolor": INK2})
    ax.set_yticks([i + (i // 2) * 0.4 for i in range(len(rows))], [f"{L[s]} · {lab}" for s, v, lab in rows], fontsize=12)
    ax.invert_yaxis()
    ax.axvline(d[d.structure == "dense"].excess.iloc[0] * 1e3, color=C["dense"], ls="--", lw=1.8, label="Dense")
    ax.set_xlabel(r"MSE $-$ Bayes floor  ($\times 10^{-3}$)", fontsize=12)
    ax.legend(fontsize=11, loc="lower right"); ax.grid(axis="y", visible=False)
    save(fig, "dyn_variants_other")


def fig_regime_mi(rtab):
    r = rtab[rtab.variant == "default"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, ds in zip(axes, ["dyn", "dyn_fast"]):
        x = r[r.dataset == ds]
        for i, s in enumerate(ROUTED):
            for layer in (0, 1):
                q = x[(x.structure == s) & (x.layer == layer)].iloc[0]
                ax.bar(i + (layer - 0.5) * 0.36, q.regime_mi_obs, width=0.34, yerr=q.regime_mi_obs_sd, capsize=3,
                       color=C[s], alpha=0.35 if layer == 0 else 1.0, error_kw={"lw": 1, "ecolor": INK2})
        ax.set_xticks(range(len(ROUTED)), [L[s] + (" (FFN)" if s == "moe" else " (attn)") for s in ROUTED], fontsize=12)
        ax.set_title(D.DS_LABEL[ds], fontsize=13)
    axes[0].set_ylabel("normalised MI(routing; last regime)", fontsize=12)
    from matplotlib.patches import Patch
    axes[1].legend([Patch(color="#777", alpha=0.35), Patch(color="#777")], ["layer 1", "layer 2"], fontsize=11)
    save(fig, "dyn_regime_mi")


def dynamics():
    ref = D.references()
    df, rt, hist = D.load()
    df, main, age, roll, rtab = D.tables(df, rt, ref)
    D.fig_main(main, ref, out=OUT)
    D.fig_age(age, ref, "dyn", out=OUT)
    D.fig_example("dyn", out=OUT)
    D.fig_example("dyn_fast", out=OUT)
    D.fig_phase("dyn", out=OUT)
    D.fig_curves(hist, "dyn", out=OUT)
    fig_matched_k(main)
    fig_dyn_variants_other(main)
    fig_regime_mi(rtab)
    D.fig_regime_routing(rt, "dyn", out=OUT)


def dynamics_lb():
    """Same dynamics slide figures, routed models trained with the load-balance loss."""
    global OUT
    base = OUT
    OUT = base / "lb"; OUT.mkdir(exist_ok=True)
    ref = D.references()
    df, rt, hist = D.load(balance="lb")
    keep = SIX
    df = df[df.structure.isin(keep)]; rt = rt[rt.structure.isin(keep)]
    df, main, age, roll, rtab = D.tables(df, rt, ref)
    D.fig_main(main, ref, out=OUT)
    D.fig_age(age, ref, "dyn", out=OUT)
    fig_matched_k(main)
    fig_dyn_variants_other(main)
    fig_regime_mi(rtab)
    OUT = base


if __name__ == "__main__":
    if sys.argv[1:] == ["lb"]:
        dynamics_lb(); sys.exit()
    if sys.argv[1:] == ["att"]:          # attention-only versions (lb for routed models) -> figures/att/
        ATT = True
        SIX = ["dense", "mha8", "moh", "moa", "momha"]; ROUTED = ["moh", "moa", "momha"]
        OUT = OUT / "att"; OUT.mkdir(exist_ok=True)
        SIX = ["dense", "moh", "moa", "momha"]; fig_memory_heatmap(); fig_fact_mod_balance()
        SIX = ["dense", "mha8", "moh", "moa", "momha"]; fig_fact_ablation("lb")
        ref = D.references()
        df, rt, hist = D.load(balance="lb")
        df = df[df.structure.isin(SIX)]; rt = rt[rt.structure.isin(SIX)]
        df, main, age, roll, rtab = D.tables(df, rt, ref)
        D.fig_age(age, ref, "dyn", out=OUT)
        fig_matched_k(main); fig_dyn_variants_other(main); fig_regime_mi(rtab)
        sys.exit()
    if sys.argv[1:] == ["eq"]:
        equations(); sys.exit()
    for s in ["dense", "moe", "moh", "moa", "momha"]:
        recolor(f"schematic_{s}")
    equations()
    fig_memory_tasks(); fig_memory_heatmap(); fig_fact_mod_balance(); fig_fact_ablation()
    fig_markov(); fig_causal()
    dynamics()


def fig_fact_matched(balance="lb"):
    """fact accuracy at matched active heads: routed models (load-balance loss) next to their router-free
    attention controls (MHA / MQA / GQA). Read straight from runs/*/result.json."""
    import json
    runs = Path(__file__).resolve().parents[2] / "runs"

    def vals(s, v, b):
        out = []
        for sd in range(5):
            p = runs / f"fact__{s}__{v}__{b}__s{sd}" / "result.json"
            if p.exists():
                out.append(json.loads(p.read_text())["test_acc"])
        return np.array(out)

    # (label, structure for colour, variant, balance, routed?)
    # families: own K/V (MHA), 1 shared K/V (MQA), 2 shared K/V (GQA); "|" starts a new family
    SUT = "MoH\nblock2\nSUT"
    groups = [
        ("4 active heads", [("MHA-4\n(Dense)", "dense", "default", "none", 0), ("MoH", "moh", "default", balance, 1),
                            (SUT, "moh", "block2_sut", balance, 2), "|",
                            ("MQA-4", "mqa4", "default", "none", 0), ("MoA", "moa", "default", balance, 1), "|",
                            ("GQA-4", "gqa4", "default", "none", 0), ("MoMHA", "momha", "default", balance, 1)]),
        ("6 active heads", [("MHA-6", "mha8", "h6", "none", 0), ("MoH", "moh", "k6", balance, 1),
                            (SUT, "moh", "block2_sut_k3", balance, 2), "|",
                            ("MQA-6", "mqa8", "h6", "none", 0), ("MoA", "moa", "k6", balance, 1), "|",
                            ("GQA-6", "gqa8", "h6", "none", 0), ("MoMHA", "momha", "k3", balance, 1)]),
        ("8 active heads (all)", [("MHA-8", "mha8", "default", "none", 0), ("MoH", "moh", "k8", balance, 1),
                                  (SUT, "moh", "block2_sut_k4", balance, 2), "|",
                                  ("MQA-8", "mqa8", "default", "none", 0), ("MoA", "moa", "k8", balance, 1), "|",
                                  ("GQA-8", "gqa8", "default", "none", 0), ("MoMHA", "momha", "k4", balance, 1)]),
    ]
    groups = [(t, [r for r in rows if r == "|" or len(vals(*r[1:4]))]) for t, rows in groups]
    widths = [sum(0.6 if r == "|" else 1 for r in g[1]) + 0.6 for g in groups]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), sharey=True, gridspec_kw={"width_ratios": widths})
    dense, mha8 = vals("dense", "default", "none").mean(), vals("mha8", "default", "none").mean()
    for ax, (title, rows) in zip(axes, groups):
        x = 0
        xs = []
        for i, r in enumerate(rows):
            if r == "|":
                x += 0.6
                continue
            lab, s, v, b, kind = r
            a = vals(s, v, b)
            col = C[s]
            ax.bar(x, a.mean(), 0.78, yerr=a.std(ddof=1) if len(a) > 1 else 0, color=col,
                   alpha=0.55 if kind == 2 else 1.0, hatch="//" if kind == 0 else None, edgecolor="white" if kind == 0 else col,
                   lw=0, capsize=3, error_kw={"lw": 1, "ecolor": INK2})
            ax.scatter([x] * len(a), a, s=10, color=INK, zorder=3)
            ax.text(x, min(a.mean() + (a.std(ddof=1) if len(a) > 1 else 0), 0.98) + 0.025, f"{a.mean():.2f}",
                    ha="center", fontsize=11, color=INK)
            xs.append((x, lab)); x += 1
        ax.set_xticks([p for p, _ in xs], [l for _, l in xs], fontsize=10)
        ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
        ax.axhline(dense, color=C["dense"], ls="--", lw=1.4)
        ax.axhline(mha8, color=C["mha8"], ls=":", lw=1.8)
        ax.set_xlim(-0.6, x - 0.4)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("fact test accuracy", fontsize=12)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    axes[-1].legend(handles=[Line2D([], [], color=C["dense"], ls="--", label=f"Dense (MHA-4) {dense:.2f}"),
                             Line2D([], [], color=C["mha8"], ls=":", lw=1.8, label=f"MHA-8 {mha8:.2f}"),
                             Patch(facecolor="#bbbbbb", hatch="//", edgecolor="white", label="no router (all heads on)")],
                    fontsize=11, loc="upper center", bbox_to_anchor=(-0.9, -0.2), ncol=3, frameon=False)
    save(fig, "memory_ablation_fact_lb")


def fig_task_heatmap_lb():
    """Test accuracy on every task. fact and mod: routed models with the load-balance loss (as on the
    matched-heads slide), 3 seeds. Other tasks were only run without balancing (2 seeds); all structures
    are at the same level there."""
    import json
    runs = Path(__file__).resolve().parents[2] / "runs"
    d = pd.read_csv(TAB / "acc_by_config.csv")
    rows = [("dense", "default", "Dense (MHA-4)"), ("moh", "default", "MoH"), ("moh", "block2_sut", "MoH-block2-SUT"),
            ("moa", "default", "MoA"), ("momha", "default", "MoMHA")]
    cols = F.DS_ORDER
    M = np.full((len(rows), len(cols)), np.nan)
    for i, (s, v, _) in enumerate(rows):
        for j, ds in enumerate(cols):
            b = "lb" if (ds in ("fact", "mod") and s != "dense") else "none"
            a = [json.loads((runs / f"{ds}__{s}__{v}__{b}__s{k}" / "result.json").read_text())["test_acc"]
                 for k in range(5) if (runs / f"{ds}__{s}__{v}__{b}__s{k}" / "result.json").exists()]
            if a:
                M[i, j] = np.mean(a)
    fig, ax = plt.subplots(figsize=(11, 4.9))
    im = ax.imshow(np.nan_to_num(M, nan=-1), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#f0efec"))
                ax.text(j, i, "not run", ha="center", va="center", fontsize=10, color=INK2)
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=13, color="white" if v > 0.6 else INK)
    ax.set_xticks(range(len(cols)), [c + ("*" if c in ("fact", "mod") else "") for c in cols], fontsize=13)
    ax.set_yticks(range(len(rows)), [r[2] for r in rows], fontsize=13)
    for k in (1, 2):                      # outline the two load-balance columns
        pass
    j0 = cols.index("fact")
    ax.add_patch(plt.Rectangle((j0 - .5, -.5), 2, len(rows), fill=False, ec=INK, lw=1.6))
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("test accuracy", fontsize=12); cb.outline.set_visible(False)
    ax.tick_params(length=0)
    save(fig, "memory_heatmap_lb")
