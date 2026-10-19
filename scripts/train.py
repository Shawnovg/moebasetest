"""Train one configuration.

    python scripts/train.py configs/logic/sut_common6.yaml --seed 1
    python scripts/train.py configs/logic/sut_common6.yaml --smoke            # tiny run, CPU friendly
    python scripts/train.py configs/logic/mom_ref6_causal.yaml \
        --set init_from.checkpoint=runs/mom_donor6_causal_s0_xxxx/best.pt

Overrides use dotted keys: --set model.params.block.miloss=0.0 --set train.max_steps=2000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moebase.train.config import apply_overrides, load_config  # noqa: E402
from moebase.train.loop import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--set", action="append", default=[], help="dotted.key=value (value parsed as YAML/JSON)")
    ap.add_argument("--smoke", action="store_true", help="2k train / 500 valid examples, 200 steps")
    ap.add_argument("--device")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--no-test", action="store_true", help="skip test partitions (screening on validation only)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.smoke:
        cfg["data"]["train_limit"] = 2000
        cfg["data"]["valid_limit"] = 500
        cfg["train"].update(max_steps=200, eval_every=100, log_every=20)
        cfg["name"] = cfg["name"] + "_smoke"
    cfg = apply_overrides(cfg, args.set)
    res = run(cfg, out_dir=args.out, device=args.device, evaluate_tests=not args.no_test)
    print(json.dumps({k: v for k, v in res.items() if k in ("best_step", "best_valid_brier")}, indent=1))
    print("valid acc", round(res["valid"]["acc"], 4))
    for part, m in res.get("test", {}).items():
        print(part, "acc", round(m["acc"], 4), "brier", round(m["brier"], 4))


if __name__ == "__main__":
    main()
