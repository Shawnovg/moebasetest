"""Synthetic task generators. Every example is a pair of equal-length id lists:

    tokens  the prompt, with <ans> at every position whose answer the model must produce
    targets the answer id at those positions, -100 elsewhere (ignored by the loss)

The model is a bidirectional encoder, so it sees the whole prompt and fills all answer slots at once.

Tasks (what each one probes):
* mqar   multi-query associative recall (Arora et al. 2023, Zoology): 16 key-value pairs, then 8
         queried keys. In-context retrieval -> attention routing (MoH / MoA / MoMHA / MiTA).
* fact   4 fixed tables (two-token entity name) x relation -> value (8,100 x 4 = 32,400 facts),
         8 queries per sequence.
         Pure memorisation in the weights -> FFN capacity (MoE, Dense-wide).
* mod    (a op b) mod 97 with op in {+, -, *}; 4 equations per sequence; half of all (a, b) pairs
         per op are held out for test. Algorithmic generalisation.
* copy   copy a random string of 8-24 symbols into the answer slots.
* rev    write the string reversed. copy / rev probe positional attention.
* logic  propositional-logic relation classification (Bowman et al. 2015, pilot v1 data,
         'length' split, train lengths <= 6): p1 <sep> p2 <ans> -> one of 7 relations.
* mix    equal shares of all six (each sequence keeps its task tag).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .vocab import ANS, ENT0, EQ, LABEL, LOGIC, N_ENT, N_REL, OPS, REL0, SEP, SYM0, TAG

IGN = -100


def mqar(rng, n_pairs=16, n_queries=8, n_keys=64):
    keys = rng.choice(n_keys, n_pairs, replace=False)
    vals = rng.integers(0, 128 - n_keys, n_pairs) + n_keys   # keys n0..n63, values n64..n127
    toks = [TAG["mqar"]]
    for k_, v_ in zip(keys, vals):
        toks += [SYM0 + int(k_), SYM0 + int(v_)]
    toks.append(SEP)
    tg = [IGN] * len(toks)
    for qi in rng.choice(n_pairs, n_queries, replace=False):
        toks += [SYM0 + int(keys[qi]), ANS]
        tg += [IGN, SYM0 + int(vals[qi])]
    return toks, tg


FACT_PARTS = 90   # an entity is named by TWO tokens (first, last) from e0..e89 -> 8,100 entities


def fact_tables(seed=12345):
    g = np.random.default_rng(seed)
    return g.integers(0, 128, size=(N_REL, FACT_PARTS, FACT_PARTS))   # value per (relation, first, last)


def fact(rng, table, n_queries=8):
    """Two-token entity names stop the embedding table from storing a fact per entity token:
    the value depends on the (first, last, relation) combination, so it must be stored in the layers."""
    toks, tg = [TAG["fact"]], [IGN]
    for _ in range(n_queries):
        r, a, b = int(rng.integers(N_REL)), int(rng.integers(FACT_PARTS)), int(rng.integers(FACT_PARTS))
        toks += [ENT0 + a, ENT0 + b, REL0 + r, ANS]
        tg += [IGN, IGN, IGN, SYM0 + int(table[r, a, b])]
    return toks, tg


MOD_P = 97
MOD_OPS = {"+": lambda a, b: (a + b) % MOD_P, "-": lambda a, b: (a - b) % MOD_P, "*": lambda a, b: (a * b) % MOD_P}


def mod_split(seed=777):
    """For each op, a boolean P x P mask: True = (a, b) in the training half."""
    g = np.random.default_rng(seed)
    return {op: (g.random((MOD_P, MOD_P)) < 0.5) for op in MOD_OPS}


def mod(rng, split, train, n_eq=4):
    toks, tg = [TAG["mod"]], [IGN]
    for _ in range(n_eq):
        op = list(MOD_OPS)[int(rng.integers(3))]
        while True:
            a, b = int(rng.integers(MOD_P)), int(rng.integers(MOD_P))
            if bool(split[op][a, b]) == train:
                break
        toks += [SYM0 + a, OPS[op], SYM0 + b, EQ, ANS]
        tg += [IGN] * 4 + [SYM0 + MOD_OPS[op](a, b)]
        if _ < n_eq - 1:
            toks.append(SEP)
            tg.append(IGN)
    return toks, tg


def copyrev(rng, reverse, lo=8, hi=24, n_sym=64):
    L = int(rng.integers(lo, hi + 1))
    s = rng.integers(0, n_sym, L)
    out = s[::-1] if reverse else s
    toks = [TAG["rev" if reverse else "copy"]] + [SYM0 + int(x) for x in s] + [SEP] + [ANS] * L
    tg = [IGN] * (L + 2) + [SYM0 + int(x) for x in out]
    return toks, tg


def logic_rows(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            toks = [TAG["logic"]] + [LOGIC[t] for t in r["p1"]] + [SEP] + [LOGIC[t] for t in r["p2"]] + [ANS]
            yield toks, [IGN] * (len(toks) - 1) + [LABEL[r["label"]]]
