"""ENTRY POINT for one trial.

    python -m experiments.run --structure moe --variant top1 --dataset mqar --balance lb --seed 0 \
        [--set train.steps=3000 --set backbone.d_ff=1024] [--runs DIR] [--data DIR]

Call chain (see docs/code_map.md):
    experiments/run.py  main()
      -> backbone/config.py   load_run_config(structure, variant, overrides)
             configs/base.yaml + models/<structure>/configs/<variant>.yaml + overrides
      -> backbone/train.py    train_run(cfg)
           -> backbone/data.py     Dataset(<DATA_ROOT>/<dataset>/{train,valid,test}.npz)
           -> backbone/model.py    EncoderModel(cfg)  -> models/<structure>/__init__.py
                                     build_sublayers(...) or build_body(...)
           -> backbone/metrics.py  evaluate(...)  (accuracy + routing usage)
      -> runs/<dataset>__<structure>__<variant>__<balance>__s<seed>/
             config.json  history.jsonl  result.json  model.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml  # noqa: E402

from backbone.config import load_run_config  # noqa: E402
from backbone.paths import DATA_ROOT, RUNS_ROOT  # noqa: E402
from backbone.train import train_run  # noqa: E402


def parse_sets(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structure", required=True)
    ap.add_argument("--variant", default="default")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--balance", default="none", choices=["none", "lb", "warmup"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", action="append", help="dotted override, e.g. train.steps=3000")
    ap.add_argument("--runs", default=str(RUNS_ROOT))
    ap.add_argument("--data", default=str(DATA_ROOT))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    over = {"dataset": a.dataset, "routing.balance": a.balance, "train.seed": a.seed, **parse_sets(a.set)}
    cfg = load_run_config(a.structure, a.variant, over)
    res = train_run(cfg, runs_root=Path(a.runs), data_root=Path(a.data), verbose=not a.quiet)
    keys = ("run", "test_mse", "test_mse_ctx8", "params", "train_time_s") if cfg.get("task") == "dynamics" \
        else ("run", "test_acc", "valid_acc", "params", "train_time_s")
    print(json.dumps({k: res[k] for k in keys}))


if __name__ == "__main__":
    main()
