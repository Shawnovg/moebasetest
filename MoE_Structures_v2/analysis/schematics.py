"""Schematic figures of each structure (reports/figures/schematic_<structure>.png).

Each figure has two panels: left, the 2-layer backbone with the swapped sub-layer highlighted;
right, the inside of the routed module with the reference's shapes for this study's defaults.
    python -m analysis.schematics          # draws all eight
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.figures import STRUCT_COLOR, STRUCT_LABEL, _save  # noqa: E402

INK, INK2, FILL, ROUTE = "#0b0b0b", "#52514e", "#f0efec", "#fff4d6"


def box(ax, x, y, w, h, text, fc=FILL, ec=INK2, fs=8, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.0, ls=ls))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, color=INK2, ls="-", lw=1.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, color=color, lw=lw, ls=ls,
                                 shrinkA=1, shrinkB=1))


def backbone_panel(ax, structure, swap):
    """swap: 'attn' | 'ffn' | 'both' (MoM: the whole stack is a routed pool)."""
    c = STRUCT_COLOR[structure]
    ax.set_xlim(0, 4); ax.set_ylim(0, 10); ax.axis("off")
    box(ax, 2, 0.6, 2.6, 0.6, "tokens -> embedding (RoPE in attention)")
    y = 1.6
    for layer in range(2):
        if structure == "mom":
            box(ax, 2, y + 1.6, 3.0, 2.8, "", fc="#ffffff", ec=c, ls="--")
            ax.text(0.55, y + 3.12, f"assembly step {layer + 1}", fontsize=7, color=INK2)
        else:
            ax.text(0.45, y + 1.6, f"layer {layer + 1}", fontsize=7, color=INK2, rotation=90, va="center")
        box(ax, 2, y + 0.35, 2.2, 0.45, "LayerNorm" if structure != "mom" else "LN of chosen module", fs=7)
        fa = c if swap in ("attn", "both") else FILL
        box(ax, 2, y + 1.1, 2.4, 0.6, "routed attention" if swap in ("attn", "both") else "multi-head attention",
            fc=fa if fa == FILL else ROUTE, ec=c if fa != FILL else INK2, bold=swap in ("attn", "both"))
        box(ax, 2, y + 1.85, 2.2, 0.45, "+ residual / LayerNorm", fs=7)
        box(ax, 2, y + 2.6, 2.4, 0.6, "routed FFN" if swap in ("ffn", "both") else "FFN 128-512-128",
            fc=ROUTE if swap in ("ffn", "both") else FILL, ec=c if swap in ("ffn", "both") else INK2,
            bold=swap in ("ffn", "both"))
        for a, b in ((y - 0.1 if layer else 0.9, y + 0.12), (y + 0.58, y + 0.8), (y + 1.4, y + 1.62),
                     (y + 2.08, y + 2.3)):
            arrow(ax, 2, a, 2, b)
        y += 3.4
    box(ax, 2, 9.0, 2.6, 0.5, "+ residual -> final LN -> answer head", fs=7)
    arrow(ax, 2, y - 0.5, 2, 8.75)
    ax.set_title(f"{STRUCT_LABEL[structure]}: where it changes the backbone", fontsize=9)


def detail_dense(ax):
    box(ax, 5, 8.5, 5, 0.7, "x (token)")
    for i in range(4):
        box(ax, 2 + 2 * i, 6, 1.6, 0.9, f"head {i + 1}\nq,k,v 32-d")
        arrow(ax, 5, 8.1, 2 + 2 * i, 6.5)
        arrow(ax, 2 + 2 * i, 5.5, 5, 3.4)
    box(ax, 5, 3, 3, 0.7, "concat -> W_o")
    ax.text(5, 1.4, "every token uses all 4 heads and the full FFN\n(Dense-wide: FFN width 2,048)", ha="center",
            fontsize=8, color=INK2)


def detail_moe(ax):
    box(ax, 5, 9, 4, 0.6, "x (token)")
    box(ax, 5, 7.6, 5.4, 0.8, "router: Linear-Tanh-Linear -> softmax over 8\n(top-1 variant: linear; hash: token id table)",
        fc=ROUTE, fs=7)
    arrow(ax, 5, 8.7, 5, 8.0)
    for i in range(8):
        x = 1.2 + i * 1.1
        on = i in (1, 5)
        box(ax, x, 5.6, 0.95, 1.1, f"E{i + 1}\n128-256\n-128", fc="#ffffff" if not on else ROUTE,
            ec=STRUCT_COLOR["moe"] if on else INK2, fs=6, ls="-" if on else ":")
        if on:
            arrow(ax, 5, 7.2, x, 6.15, color=STRUCT_COLOR["moe"])
            arrow(ax, x, 5.05, 5, 3.6, color=STRUCT_COLOR["moe"])
    box(ax, 5, 3.2, 4.6, 0.7, "sum  g1*E2(x) + g2*E6(x)   (g renormalised over top-2)", fs=7)
    ax.text(5, 1.6, "top-2 of 8 experts per token; balancing: Switch loss 0.01", ha="center", fontsize=8, color=INK2)


def detail_moh(ax):
    box(ax, 5, 9, 4, 0.6, "x (token)")
    box(ax, 5, 7.7, 5.4, 0.7, "router W_g: softmax over 8 heads, keep top-4, renormalise x4", fc=ROUTE, fs=7)
    for i in range(8):
        x = 1.2 + i * 1.1
        on = i in (0, 2, 3, 6)
        box(ax, x, 5.7, 0.95, 1.0, f"head {i + 1}", fc=ROUTE if on else "#ffffff", ec=STRUCT_COLOR["moh"] if on else INK2,
            fs=7, ls="-" if on else ":")
        if on:
            arrow(ax, x, 5.2, 5, 3.6, color=STRUCT_COLOR["moh"])
    box(ax, 5, 3.2, 5.4, 0.7, "g_h * head_h, unchosen heads x 0 -> concat -> W_o", fs=7)
    ax.text(5, 1.5, "8 heads x 32 (all computed, 4 kept per token)\nshared2 variant: 2 always-on heads + 2 of 6 routed",
            ha="center", fontsize=8, color=INK2)


def detail_moa(ax, momha=False):
    s = "momha" if momha else "moa"
    box(ax, 5, 9, 4, 0.6, "x (token)")
    box(ax, 5, 7.7, 5.6, 0.7, ("SUT MLP router: top-2 of 4 experts" if momha else "router x W_g (zero-init): top-4 of 8 experts"),
        fc=ROUTE, fs=7)
    n, on = (4, (0, 2)) if momha else (8, (1, 2, 5, 7))
    for i in range(n):
        x = 1.2 + i * (7.6 / (n - 1))
        box(ax, x, 5.8, 1.5 if momha else 0.95, 1.1,
            (f"expert {i + 1}\n2 query heads\n+ 2 out-proj" if momha else f"E{i + 1}\nW_q, W_o"),
            fc=ROUTE if i in on else "#ffffff", ec=STRUCT_COLOR[s] if i in on else INK2, fs=6, ls="-" if i in on else ":")
        if i in on:
            arrow(ax, x, 5.25, 4, 3.5, color=STRUCT_COLOR[s])
    box(ax, 8.3, 3.1, 2.6, 0.7, "shared K, V\n(" + ("2 heads" if momha else "1 head") + ", 32-d)", fs=7)
    box(ax, 4, 3.1, 3.6, 0.7, "sum_e g_e * W_o,e attn(q_e, K, V)", fs=7)
    ax.text(5, 1.4, ("SUT MoMHA: 4 heads active per token, renormalised gates" if momha else
                     "MoA: 4 single-head experts per token, raw softmax gates (code)\nbalancing: Switch 0.01 + z-loss 0.001"),
            ha="center", fontsize=8, color=INK2)


def detail_mom(ax):
    box(ax, 5, 9.2, 4, 0.55, "x at step t")
    box(ax, 5, 8.2, 6.4, 0.6, "attention router (MLP 128-256-5): pick 1 of {A1..A4, SKIP}", fc=ROUTE, fs=7)
    for i, n in enumerate(["A1", "A2", "A3", "A4", "SKIP"]):
        x = 1.4 + i * 1.8
        on = i == 2
        box(ax, x, 7.0, 1.3, 0.6, n, fc=ROUTE if on else "#ffffff", ec=STRUCT_COLOR["mom"] if on else INK2, fs=7,
            ls="-" if on else ":")
    box(ax, 5, 5.9, 8.2, 0.6, "own LN -> q,k,v of chosen module -> attention -> r * W_o (raw score r)", fs=7)
    box(ax, 5, 4.7, 6.4, 0.6, "FFN router: pick 1 of {F1..F4, SKIP}", fc=ROUTE, fs=7)
    for i, n in enumerate(["F1", "F2", "F3", "F4", "SKIP"]):
        x = 1.4 + i * 1.8
        on = i == 2
        box(ax, x, 3.6, 1.3, 0.6, n, fc=ROUTE if on else "#ffffff", ec=STRUCT_COLOR["mom"] if on else INK2, fs=7,
            ls="-" if on else ":")
    box(ax, 5, 2.5, 5, 0.6, "x + r * FFN_chosen(LN(x))  -> step t+1 (2 steps)", fs=7)
    for a, b in ((8.9, 8.5), (7.9, 7.3), (6.7, 6.2), (5.6, 5.0), (4.4, 3.9), (3.3, 2.8)):
        arrow(ax, 5, a, 5, b)
    ax.text(5, 1.3, "one shared pool reused at every step; each token assembles its own path", ha="center",
            fontsize=8, color=INK2)


def detail_mita(ax):
    box(ax, 5, 9.2, 6, 0.55, "queries q_1..q_N (per head)")
    box(ax, 5, 8.1, 6.4, 0.6, "landmarks: average q over m = 8 windows -> r_1..r_8", fs=7)
    for i in range(8):
        x = 1.2 + i * 1.1
        on = i == 3
        box(ax, x, 6.6, 0.95, 1.2, f"expert {i + 1}\ntop-8 keys\nof r_{i + 1}", fc=ROUTE if on else "#ffffff",
            ec=STRUCT_COLOR["mita"] if on else INK2, fs=6, ls="-" if on else ":")
    box(ax, 5, 5.0, 6.6, 0.6, "query routed to argmax_j  r_j . q   (no learned router)", fc=ROUTE, fs=7)
    box(ax, 2.6, 3.7, 4.2, 0.7, "shared: keys r_j, values\nagent_j = softmax(r_j K) V", fs=7)
    box(ax, 7.4, 3.7, 4.2, 0.7, "routed: its expert's 8 keys/values", fs=7)
    box(ax, 5, 2.4, 6.4, 0.6, "one softmax over [8 shared + 8 routed] -> output", fs=7)
    arrow(ax, 2.6, 3.3, 4.3, 2.7); arrow(ax, 7.4, 3.3, 5.7, 2.7)
    ax.text(5, 1.2, "key/value pairs are the experts; each query reads 16 slots instead of N", ha="center",
            fontsize=8, color=INK2)


SPECS = {"dense": ("none", detail_dense), "dense_wide": ("none", detail_dense), "moe": ("ffn", detail_moe),
         "moh": ("attn", detail_moh), "moa": ("attn", detail_moa), "momha": ("attn", lambda ax: detail_moa(ax, True)),
         "mom": ("both", detail_mom), "mita": ("attn", detail_mita)}


def draw(structure, out=None):
    swap, detail = SPECS[structure]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 5.2), gridspec_kw={"width_ratios": [1, 1.6]})
    backbone_panel(a, structure, swap)
    b.set_xlim(0, 10); b.set_ylim(0, 10); b.axis("off")
    detail(b)
    b.set_title(f"inside the {'module' if swap != 'none' else 'layer'} (this study's defaults)", fontsize=9)
    kw = {} if out is None else {"out": out}
    return _save(fig, f"schematic_{structure}", **kw)


if __name__ == "__main__":
    for s in SPECS:
        print(draw(s))
