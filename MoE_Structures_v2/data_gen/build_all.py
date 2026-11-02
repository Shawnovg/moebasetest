"""Build every dataset into DATA_ROOT/<dataset>/{train,valid,test}.npz + manifest.json.

    python -m data_gen.build_all [--out DIR] [--n-train 200000]

Each .npz holds ``tokens`` and ``targets`` (int16, N x T_max, padded with <pad> / -100) and
``lengths``. ``DATA_ROOT/vocab.json`` is the shared vocabulary. Generation is seeded, so the
files are reproducible; manifest.json records sizes, seeds, lengths and a sha256 per file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backbone.paths import DATA_ROOT, PROJECT_ROOT  # noqa: E402
from data_gen import tasks  # noqa: E402
from data_gen.vocab import PAD, VOCAB  # noqa: E402

LOGIC_SRC = Path("/home/claude/work/MoE_Base_Test/data/logic-v1/length")
SPLIT_SEED = {"train": 1, "valid": 2, "test": 3}
TASK_SEED = {"mqar": 10, "fact": 20, "mod": 30, "copy": 40, "rev": 50}


def pack(rows):
    T = max(len(t) for t, _ in rows)
    tok = np.full((len(rows), T), PAD, dtype=np.int16)
    tgt = np.full((len(rows), T), -100, dtype=np.int16)
    lens = np.zeros(len(rows), dtype=np.int16)
    for i, (t, g) in enumerate(rows):
        tok[i, :len(t)], tgt[i, :len(g)], lens[i] = t, g, len(t)
    return {"tokens": tok, "targets": tgt, "lengths": lens}


def generate(task, split, n, logic_src=LOGIC_SRC):
    if task == "logic":
        f = {"train": "train.jsonl", "valid": "valid.jsonl", "test": "test_iid.jsonl"}[split]
        rows = list(tasks.logic_rows(logic_src / f))
        if n < len(rows):
            idx = np.random.default_rng(SPLIT_SEED[split]).choice(len(rows), n, replace=False)
            rows = [rows[i] for i in sorted(idx)]
        return rows
    rng = np.random.default_rng(TASK_SEED[task] * 100 + SPLIT_SEED[split])
    if task == "mqar":
        return [tasks.mqar(rng) for _ in range(n)]
    if task == "fact":
        table = tasks.fact_tables()
        return [tasks.fact(rng, table) for _ in range(n)]
    if task == "mod":
        sp = tasks.mod_split()
        return [tasks.mod(rng, sp, train=(split != "test")) for _ in range(n)]
    if task in ("copy", "rev"):
        return [tasks.copyrev(rng, reverse=(task == "rev")) for _ in range(n)]
    raise ValueError(task)


def save(out_dir: Path, name, split_rows, extra):
    d = out_dir / name
    d.mkdir(parents=True, exist_ok=True)
    man = {"dataset": name, **extra, "splits": {}}
    for split, rows in split_rows.items():
        arr = pack(rows)
        p = d / f"{split}.npz"
        np.savez_compressed(p, **arr)
        n_ans = int((arr["targets"] >= 0).sum())
        man["splits"][split] = {"n": len(rows), "max_len": int(arr["tokens"].shape[1]),
                                "mean_len": float(arr["lengths"].mean()), "answer_slots": n_ans,
                                "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    (d / "manifest.json").write_text(json.dumps(man, indent=1))
    print(name, {s: (v["n"], v["max_len"]) for s, v in man["splits"].items()}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA_ROOT))
    ap.add_argument("--n-train", type=int, default=200_000)
    ap.add_argument("--n-valid", type=int, default=2_000)
    ap.add_argument("--n-test", type=int, default=4_000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps(VOCAB))
    sizes = {"train": a.n_train, "valid": a.n_valid, "test": a.n_test}
    per_task = {}
    for task in tasks_list():
        per_task[task] = {s: generate(task, s, n) for s, n in sizes.items()}
        save(out, task, per_task[task], {"task": task, "doc": DOCS[task]})
    # mix: equal shares of every task, shuffled
    rng = np.random.default_rng(99)
    mix = {}
    for s, n in sizes.items():
        rows = []
        for task in tasks_list():
            src = per_task[task][s]
            idx = rng.choice(len(src), min(len(src), n // len(tasks_list())), replace=False)
            rows += [src[i] for i in idx]
        rng.shuffle(rows)
        mix[s] = rows
    save(out, "mix", mix, {"task": "mix", "doc": "equal shares of " + ", ".join(tasks_list())})


def tasks_list():
    return ["mqar", "fact", "mod", "copy", "rev", "logic"]


DOCS = {
    "mqar": "16 key-value pairs (keys n0-n63, values n64-n127), then 8 queried keys each followed by <ans>",
    "fact": "8 queries 'first last relation <ans>' (entity named by two tokens e0-e89, 8,100 entities) into 4 fixed tables -> value n0-n127 (seed 12345); test = same facts, new query mixes",
    "mod": "4 equations 'a op b = <ans>' mod 97, op in + - *; test uses the held-out half of (a,b) pairs per op",
    "copy": "string of 8-24 symbols (n0-n63), <sep>, then one <ans> per symbol: copy",
    "rev": "as copy, answer reversed",
    "logic": "Bowman et al. propositional logic, pilot v1 'length' split (train ops<=6): p1 <sep> p2 <ans> -> relation",
}

if __name__ == "__main__":
    main()
