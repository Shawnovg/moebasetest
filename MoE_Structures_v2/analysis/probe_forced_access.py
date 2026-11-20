"""Probe 1, forced access (fact): does a routed model already STORE facts it fails to ACCESS?

Take each trained routed model (runs_probe/) and, at test time only, change how many heads the router
switches on (k_eval), without retraining. If accuracy rises when more heads are switched on, part of the
knowledge sat in heads the router did not pick for that token (an access problem, hypothesis H1).
If accuracy only falls, the model relies on its routing and the missing facts are not stored (H2).

* MoA: q_proj.k = k_eval (raw softmax probabilities as gates, as trained; extra heads get their small gate)
* MoH-block2-SUT: routed_head = 2 * groups (gates renormalised to sum to 1, as trained)

Output: reports/probe/forced_access.json and reports/figures/probe/forced_access.png
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backbone.data import Dataset, load_vocab  # noqa: E402
from backbone.metrics import evaluate  # noqa: E402
from backbone.model import EncoderModel  # noqa: E402
from backbone.paths import DATA_ROOT  # noqa: E402

RUNS = ROOT / "runs_probe"
OUT = ROOT / "reports" / "probe"
OUT.mkdir(parents=True, exist_ok=True)


def load(run):
    cfg = json.loads((RUNS / run / "config.json").read_text())
    vocab = load_vocab(DATA_ROOT)
    m = EncoderModel(cfg, len(vocab))
    sd = torch.load(RUNS / run / "model.pt", map_location="cpu")
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    m.eval()
    return cfg, m, vocab


def set_k(model, cfg, k):
    for _, mod in model.routed_modules():
        if cfg["structure"] == "moa":
            mod.q_proj.k = k
        else:
            mod.routed_head = k


def main():
    torch.set_num_threads(2)
    test = Dataset("fact", "test", DATA_ROOT)
    res = {}
    for d in sorted(RUNS.iterdir()):
        cfg = json.loads((d / "config.json").read_text())
        if cfg["structure"] not in ("moa", "moh"):
            continue
        sp = cfg["structure_params"]
        if cfg["structure"] == "moa":
            E, k0 = sp["num_experts"], sp["k"]
            ks = list(range(1, E + 1)) if E == 8 else [2, 4, 6, 8, 10, 12, 14, 16]
        else:
            E, k0 = sp["num_heads"], sp["routed_head"]
            ks = [2, 4, 6, 8]
        _, model, vocab = load(d.name)
        task_ids = {i: tok[3:-1] for i, tok in enumerate(vocab) if tok.startswith("<t:")}
        row = {}
        with torch.no_grad():
            for k in ks:
                set_k(model, cfg, k)
                row[k] = evaluate(model, test, routing=False, task_ids=task_ids)["acc"]
        res[d.name] = {"structure": cfg["structure"], "variant": cfg["variant"], "pool": E, "k_train": k0,
                       "seed": cfg["train"]["seed"], "acc": row}
        print(d.name, k0, {k: round(v, 3) for k, v in row.items()}, flush=True)
    (OUT / "forced_access.json").write_text(json.dumps(res, indent=1))

    # controls' accuracy (all heads always on) for reference
    ctrl = defaultdict(list)
    for d in RUNS.iterdir():
        r = json.loads((d / "result.json").read_text())
        if r["structure"] not in ("moa", "moh"):
            ctrl[f"{r['structure']}:{r['variant']}"].append(r["test_acc"])
    print({k: round(float(np.mean(v)), 3) for k, v in ctrl.items()})


if __name__ == "__main__":
    main()
