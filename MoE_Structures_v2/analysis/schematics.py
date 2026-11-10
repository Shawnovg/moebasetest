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


def kv_controls(out=None):
    """MHA / MQA / GQA: 4 query heads, and how many K/V heads they read (no router)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    specs = [("MHA-4  (= Dense)", 4, "dense", "MoH", "every query head has its own K, V"),
             ("MQA-4", 1, "mqa4", "MoA", "all query heads share 1 K, V head"),
             ("GQA-4", 2, "gqa4", "MoMHA", "query heads in groups of 2 share a K, V head")]
    for ax, (name, nkv, c, routed, note) in zip(axes, specs):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
        col = STRUCT_COLOR[c]
        box(ax, 5, 9.2, 4, 0.6, "x (token)", fs=10)
        qx = [1.6 + i * 2.27 for i in range(4)]
        kx = qx if nkv == 4 else ([5.0] if nkv == 1 else [2.7, 7.3])
        for i, x in enumerate(qx):
            box(ax, x, 7.3, 1.8, 0.8, f"Q head {i + 1}", fc=ROUTE, ec=col, fs=10)
            arrow(ax, 5, 8.9, x, 7.75)
        for j, x in enumerate(kx):
            box(ax, x, 4.9, 1.8 if nkv == 4 else 2.6, 0.8, f"K,V {j + 1}" if nkv > 1 else "K,V (one)", fs=10)
        for i, x in enumerate(qx):
            k = kx[i if nkv == 4 else (0 if nkv == 1 else i // 2)]
            arrow(ax, x, 6.9, k, 5.32, color=col)
        box(ax, 5, 3.0, 8.6, 0.7, "attention per Q head -> concat -> output proj.", fs=10)
        ax.text(5, 1.9, note, ha="center", fontsize=11, color=INK)
        ax.text(5, 1.0, f"routed counterpart: {routed}   |   8-head version: {name.split('-')[0]}-8",
                ha="center", fontsize=10, color=INK2)
        ax.set_title(name, fontsize=14, fontweight="bold", color=col)
    fig.suptitle("Router-free attention controls: all heads are used for every token", fontsize=12, x=0.01,
                 ha="left", color=INK2)
    kw = {} if out is None else {"out": out}
    return _save(fig, "schematic_kv_controls", **kw)


def moh_blocks(out=None):
    """MoH-block2 / MoH-block2-SUT vs MoMHA: same block routing, own vs shared K/V."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    c = STRUCT_COLOR["moh"]
    for ax, sut in zip(axes, (False, True)):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
        box(ax, 5, 9.3, 4, 0.6, "x (token)", fs=10)
        box(ax, 5, 8.1, 9.6, 0.75, ("MLP router (128-128-tanh-4): top-2 of 4 blocks, gates sum to 1"
                                     if sut else "linear router W_g: top-2 of 4 blocks, gates x 2"), fc=ROUTE, fs=10)
        arrow(ax, 5, 9.0, 5, 8.5)
        for b in range(4):
            x = 1.35 + b * 2.43
            on = b in (0, 2)
            ax.add_patch(FancyBboxPatch((x - 1.05, 4.0), 2.1, 3.1, boxstyle="round,pad=0.02,rounding_size=0.1",
                                        fc="#ffffff", ec=c if on else INK2, lw=1.3 if on else 1.0, ls="-" if on else ":"))
            ax.text(x, 6.85, f"block {b + 1}", ha="center", fontsize=10, fontweight="bold" if on else "normal")
            for h in range(2):
                box(ax, x, 6.15 - h * 1.15, 1.75, 0.85, f"head {2 * b + h + 1}\nown Q, K, V", fs=8.5,
                    fc=ROUTE if on else "#ffffff", ec=c if on else INK2, ls="-" if on else ":")
            if on:
                arrow(ax, x, 3.95, 5, 3.15, color=c)
        box(ax, 5, 2.75, 8.8, 0.7, "sum over chosen blocks:  g_b x (heads of block b) -> output proj.", fs=10)
        ax.text(5, 1.75, ("MoH-block2-SUT: MoMHA's router, gates and switch balance loss" if sut else
                          "MoH-block2: MoH's router, gates and balance loss, heads grouped in pairs"),
                ha="center", fontsize=11, color=INK)
        ax.text(5, 0.95, ("only difference from MoMHA: K, V per head (8) instead of 2 shared by all blocks" if sut
                          else "4 active heads per token, same as MoH, MoA and MoMHA"),
                ha="center", fontsize=10, color=INK2)
        ax.set_title("MoH-block2-SUT" if sut else "MoH-block2", fontsize=14, fontweight="bold", color=c)
    kw = {} if out is None else {"out": out}
    return _save(fig, "schematic_moh_blocks", **kw)


# ------------------------------------------------------------- attention-only pair figures (Nov 10)
LIGHT = "#b9b8b4"


def attn_panel(ax, title, color, kv_of, n_kv, sel=None, blocks=None, block_name="block", router=None,
               qlab="Q", out="", notes=("", ""), kv_label=None):
    """One attention layer: 8 query heads (optionally grouped into blocks) reading n_kv K/V heads.
    sel = set of chosen query heads (None: no router, all used)."""
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    on = set(range(8)) if sel is None else set(sel)
    box(ax, 5, 9.35, 3.6, 0.55, "x (token)", fs=10)
    if router:
        box(ax, 5, 8.35, 9.4, 0.7, router, fc=ROUTE, ec=color, fs=10)
        arrow(ax, 5, 9.07, 5, 8.72)
    else:
        ax.text(5, 8.35, "no router: every head is used for every token", ha="center", va="center", fontsize=10,
                color=INK2, style="italic")
    qx = [0.75 + i * 1.214 for i in range(8)]
    if blocks:
        for b, hs in enumerate(blocks):
            x0, x1 = qx[hs[0]] - 0.58, qx[hs[-1]] + 0.58
            chosen = all(h in on for h in hs)
            ax.add_patch(FancyBboxPatch((x0, 5.95), x1 - x0, 1.75, boxstyle="round,pad=0.02,rounding_size=0.1",
                                        fc="none", ec=color if chosen else INK2, lw=1.4 if chosen else 1.0,
                                        ls="-" if chosen else ":"))
            ax.text((x0 + x1) / 2, 7.45, f"{block_name} {b + 1}", ha="center", va="center", fontsize=9,
                    fontweight="bold" if chosen else "normal", color=INK)
    for i, x in enumerate(qx):
        c = i in on
        box(ax, x, 6.5, 1.0, 0.75, f"{qlab}{i + 1}", fc=ROUTE if c else "#ffffff", ec=color if c else INK2,
            fs=9.5, ls="-" if c else ":")
    kx = []
    for j in range(n_kv):
        hs = [i for i in range(8) if kv_of[i] == j]
        kx.append(sum(qx[i] for i in hs) / len(hs) if n_kv in (1, 8) or kv_label is None else 2.6 + j * 4.8)
    for j, x in enumerate(kx):
        w = 1.0 if n_kv == 8 else (3.9 if kv_label and len(kv_label[j]) > 16 else 2.4)
        box(ax, x, 4.35, w, 0.75, (kv_label[j] if kv_label else ("K,V" if n_kv == 8 else f"K,V {j + 1}")),
            fs=9 if n_kv == 8 else 10)
    for i, x in enumerate(qx):
        c = i in on
        arrow(ax, x, 6.1, kx[kv_of[i]], 4.75, color=color if c else LIGHT, ls="-" if c else ":", lw=1.1 if c else 0.8)
    box(ax, 5, 2.75, 9.4, 0.7, out, fs=10)
    ax.text(5, 1.75, notes[0], ha="center", fontsize=11, color=INK)
    ax.text(5, 0.95, notes[1], ha="center", fontsize=10, color=INK2)
    ax.set_title(title, fontsize=15, fontweight="bold", color=color)


def pair(name, left, right, out=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.0))
    attn_panel(axes[0], **left); attn_panel(axes[1], **right)
    fig.subplots_adjust(wspace=0.06, left=0.01, right=0.99)
    kw = {} if out is None else {"out": out}
    return _save(fig, f"pair_{name}", **kw)


def draw_pairs(out=None):
    C = STRUCT_COLOR
    MO = list(range(8))
    SEL4 = (0, 2, 3, 6)
    p = []
    p.append(pair("moh", dict(title="MHA-8 (no router)", color=C["mha8"], kv_of=MO, n_kv=8,
                              out="concat 8 heads -> W_O", notes=("8 query heads, each with its own K, V",
                                                                    "MHA-4 (= Dense): the same with 4 heads")),
                  dict(title="MoH", color=C["moh"], kv_of=MO, n_kv=8, sel=SEL4,
                       router="linear router: softmax over 8 heads, keep top-4, gates renormalised x 4",
                       out="sum over chosen heads:  g_h x head_h W_O^h   (unchosen heads x 0)",
                       notes=("same heads as MHA-8; the router keeps 4 per token", "MoH balance loss 0.01")), out))
    gq = [i // 4 for i in range(8)]
    p.append(pair("moa", dict(title="MQA-8 (no router)", color=C["mqa8"], kv_of=[0] * 8, n_kv=1,
                              out="concat 8 heads -> W_O", notes=("8 query heads share one K, V head",
                                                                    "MQA-4: the same with 4 query heads")),
                  dict(title="MoA", color=C["moa"], kv_of=[0] * 8, n_kv=1, sel=(1, 2, 5, 7), qlab="E",
                       router="linear router (zero-init): softmax over 8 experts, keep top-4, raw gates",
                       out="sum over chosen experts:  g_e x attn(q_e, K, V) W_O,e",
                       notes=("expert e = one query head with its own W_q and W_O",
                              "switch loss 0.01 + z-loss 0.001")), out))
    blocks = [[0, 1], [2, 3], [4, 5], [6, 7]]
    mm = [i % 2 for i in range(8)]          # MoMHA: head h of every expert uses K/V head h
    p.append(pair("momha", dict(title="GQA-8 (no router)", color=C["gqa8"], kv_of=gq, n_kv=2,
                                kv_label=["K,V 1 (Q1-Q4)", "K,V 2 (Q5-Q8)"],
                                out="concat 8 heads -> W_O", notes=("8 query heads, groups of 4 share a K, V head",
                                                                      "GQA-4: 4 query heads, 2 K, V heads")),
                  dict(title="MoMHA", color=C["momha"], kv_of=mm, n_kv=2, sel=(0, 1, 4, 5), blocks=blocks,
                       block_name="expert", kv_label=["K,V 1 (1st head of each expert)", "K,V 2 (2nd head)"],
                       router="MLP router (128-128-tanh-4): keep top-2 of 4 experts, gates sum to 1",
                       out="sum over chosen experts:  g_e x MHA_e(x)   (own W_q, W_O; shared K, V)",
                       notes=("expert = 2 query heads + own output projection", "switch loss 0.01")), out))
    p.append(pair("sut", dict(title="grouped MHA-8 (no router)", color=C["mha8"], kv_of=MO, n_kv=8, blocks=blocks,
                              out="sum over all 4 blocks:  (2 heads of block b) W_O",
                              notes=("all blocks always on: the grouping changes nothing",
                                     "so this is exactly MHA-8 (and MHA-4 with 2 blocks)")),
                  dict(title="MoH-block2-SUT", color=C["moh"], kv_of=MO, n_kv=8, sel=(0, 1, 4, 5), blocks=blocks,
                       router="MLP router (128-128-tanh-4): keep top-2 of 4 blocks, gates sum to 1",
                       out="sum over chosen blocks:  g_b x (2 heads of block b) W_O",
                       notes=("MoMHA's router, gates and switch loss, but every head has its own Q, K, V",
                              "only difference from MoMHA: 8 K, V heads instead of 2 shared")), out))
    return p
