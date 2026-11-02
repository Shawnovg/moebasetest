"""One shared vocabulary for every task, so one model (and the mixed dataset) can read all of them.

Layout (ids are fixed; ``build_vocab()`` returns the list, index = id):
    0 <pad>   1 <ans> (answer slot: the model predicts the answer token at this position)
    2 <sep>   3 =   4 +   5 -   6 *
    task tags  <t:mqar> <t:fact> <t:mod> <t:copy> <t:rev> <t:logic>
    n0 .. n127      generic symbols / numbers (MQAR keys & values, residues mod 97, copy symbols, fact values)
    e0 .. e511      entities of the fact tables
    r0 .. r3        relations (= the 4 fact tables)
    logic           ( ) a b c d e f and or not
    logic labels    L< L> L= L^ L| L# Lv
"""
from __future__ import annotations

TASKS = ["mqar", "fact", "mod", "copy", "rev", "logic"]
N_SYM, N_ENT, N_REL = 128, 512, 4
LOGIC_SYMBOLS = ["(", ")", "a", "b", "c", "d", "e", "f", "and", "or", "not"]
LOGIC_LABELS = ["<", ">", "=", "^", "|", "#", "v"]


def build_vocab():
    v = ["<pad>", "<ans>", "<sep>", "=", "+", "-", "*"]
    v += [f"<t:{t}>" for t in TASKS]
    v += [f"n{i}" for i in range(N_SYM)]
    v += [f"e{i}" for i in range(N_ENT)]
    v += [f"r{i}" for i in range(N_REL)]
    v += [f"x{s}" for s in LOGIC_SYMBOLS]
    v += [f"L{s}" for s in LOGIC_LABELS]
    return v


VOCAB = build_vocab()
ID = {t: i for i, t in enumerate(VOCAB)}
PAD, ANS, SEP, EQ = ID["<pad>"], ID["<ans>"], ID["<sep>"], ID["="]
OPS = {"+": ID["+"], "-": ID["-"], "*": ID["*"]}
TAG = {t: ID[f"<t:{t}>"] for t in TASKS}
SYM0, ENT0, REL0 = ID["n0"], ID["e0"], ID["r0"]
LOGIC = {s: ID[f"x{s}"] for s in LOGIC_SYMBOLS}
LABEL = {s: ID[f"L{s}"] for s in LOGIC_LABELS}
