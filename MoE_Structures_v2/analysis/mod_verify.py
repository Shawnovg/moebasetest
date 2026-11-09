"""Independent check of the mod result (question of 9 Nov: is test accuracy 1.00 real?).

1. Data audit: sequence length, answer slots, modulus, and whether any test equation (op, a, b) also
   occurs in the training file. Targets are recomputed from the tokens with plain Python arithmetic.
2. Exhaustive re-test: every held-out (op, a, b) triple of the split (about 14,000), packed in fresh
   random order, either 4 per sequence (as in training) or 1 per sequence (nothing else in context).
   The same is done for the training-half triples. Both checkpoints from 3k and 30k steps are scored.

    python -m analysis.mod_verify
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backbone.data import load_vocab  # noqa: E402
from backbone.model import EncoderModel  # noqa: E402
from backbone.paths import DATA_ROOT  # noqa: E402
from data_gen.tasks import MOD_P, mod_split  # noqa: E402
from data_gen.vocab import ANS, EQ, OPS, SEP, SYM0, TAG  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OPNAME = {v: k for k, v in OPS.items()}
PY = {"+": lambda a, b: (a + b) % MOD_P, "-": lambda a, b: (a - b) % MOD_P, "*": lambda a, b: (a * b) % MOD_P}


def triples(split):
    z = np.load(DATA_ROOT / "mod" / f"{split}.npz")
    tok, tgt = z["tokens"], z["targets"]
    out, bad = [], 0
    for i, j in zip(*np.nonzero(tok == ANS)):
        a, o, b = int(tok[i, j - 4]) - SYM0, OPNAME[int(tok[i, j - 3])], int(tok[i, j - 2]) - SYM0
        out.append((o, a, b))
        bad += int(tgt[i, j]) - SYM0 != PY[o](a, b)
    lens = (tok != 0).sum(1)
    return out, bad, tok.shape, int(lens.min()), int(lens.max()), int((tok == ANS).sum(1).max())


def pack(trip, per_seq, rng):
    trip = [trip[i] for i in rng.permutation(len(trip))]
    toks, tg = [], []
    for s in range(0, len(trip), per_seq):
        t, y = [TAG["mod"]], [-100]
        for k, (o, a, b) in enumerate(trip[s:s + per_seq]):
            if k:
                t.append(SEP); y.append(-100)
            t += [SYM0 + a, OPS[o], SYM0 + b, EQ, ANS]
            y += [-100] * 4 + [SYM0 + PY[o](a, b)]
        toks.append(t); tg.append(y)
    L = max(map(len, toks))
    T = torch.zeros(len(toks), L, dtype=torch.long); Y = torch.full((len(toks), L), -100)
    for i, (t, y) in enumerate(zip(toks, tg)):
        T[i, :len(t)] = torch.tensor(t); Y[i, :len(y)] = torch.tensor(y)
    return T, Y, trip


@torch.no_grad()
def score(model, T, Y, trip):
    hits = []
    for i in range(0, len(T), 500):
        t, y = T[i:i + 500], Y[i:i + 500]
        sel = t == ANS
        lg, _ = model(t, t == 0, select=sel)
        hits += (lg.argmax(-1) == y[sel]).tolist()
    by = {}
    for (o, _, _), h in zip(trip, hits):
        by.setdefault(o, []).append(h)
    return float(np.mean(hits)), {o: round(float(np.mean(v)), 4) for o, v in sorted(by.items())}


def main():
    torch.set_num_threads(4)
    rep = {}
    tr, bad_tr, sh_tr, lo, hi, na = triples("train")
    te, bad_te, sh_te, *_ = triples("test")
    S = mod_split()
    held = [(o, a, b) for o in PY for a in range(MOD_P) for b in range(MOD_P) if not S[o][a, b]]
    seen = [(o, a, b) for o in PY for a in range(MOD_P) for b in range(MOD_P) if S[o][a, b]]
    rep["data"] = {
        "modulus": MOD_P, "train_tokens_shape": list(sh_tr), "test_tokens_shape": list(sh_te),
        "sequence_length_nonpad_min_max": [lo, hi], "answer_slots_per_sequence": na,
        "train_equations": len(tr), "distinct_train_triples": len(set(tr)),
        "test_equations": len(te), "distinct_test_triples": len(set(te)),
        "test_triples_also_in_train_file": len(set(te) & set(tr)),
        "wrong_targets_train": bad_tr, "wrong_targets_test": bad_te,
        "all_heldout_triples": len(held), "all_train_half_triples": len(seen),
        "train_file_covers_fraction_of_train_half": round(len(set(tr)) / len(seen), 4),
    }
    print(json.dumps(rep["data"], indent=1))
    runs = ["mod__dense__default__none__s0", "mod__dense__long__none__s0",
            "mod__moh__default__lb__s0", "mod__moh__long__lb__s0"]
    rng = np.random.default_rng(12345)
    for run in runs:
        d = ROOT / "runs" / run
        cfg = json.loads((d / "config.json").read_text())
        m = EncoderModel(cfg, len(load_vocab()))
        m.load_state_dict(torch.load(d / "model.pt", map_location="cpu")); m.eval()
        r = {}
        for name, trip in (("heldout", held), ("train_half", seen)):
            for per in (4, 1):
                acc, by = score(m, *pack(trip, per, rng))
                r[f"{name}_{per}_per_seq"] = {"acc": round(acc, 4), "by_op": by}
        rep[run] = r
        print(run, json.dumps(r))
    out = ROOT / "reports" / "tables" / "mod_verify.json"
    out.write_text(json.dumps(rep, indent=1))
    print(out)


if __name__ == "__main__":
    main()
