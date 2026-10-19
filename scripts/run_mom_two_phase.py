"""MoM two-phase protocol with its essential controls (plan sec. 8).

    python scripts/run_mom_two_phase.py --seed 0 [--interface causal|cls] [--smoke]

1. phase 1   : dense donor                         (mom_donor6_*)
2. phase 2   : MoM initialised from the donor      (mom_ref6_*)
3. control A : donor continued for the same budget (donor6_continue_causal; causal only)
4. control B : MoM from scratch, donor+phase-2 budget (mom_scratch6_causal; causal only)
Costs: report donor steps + phase-2 steps for the MoM path (non-amortised).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moebase.train.config import load_config  # noqa: E402
from moebase.train.loop import config_hash, run  # noqa: E402
from moebase import paths  # noqa: E402


def prep(cfg, seed, smoke):
    cfg["train"]["seed"] = seed
    if smoke:
        cfg["data"]["train_limit"], cfg["data"]["valid_limit"] = 2000, 500
        cfg["train"].update(max_steps=100, eval_every=50, log_every=10)
        cfg["name"] += "_smoke"
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--interface", choices=["causal", "cls"], default="causal")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-controls", action="store_true")
    args = ap.parse_args()
    cdir = ROOT / "configs" / "logic"
    sfx = args.interface

    donor = prep(load_config(cdir / f"mom_donor6_{sfx}.yaml"), args.seed, args.smoke)
    donor_dir = paths.RUNS_DIR / f"{donor['name']}_s{args.seed}_{config_hash(donor)}"
    results = {"donor": run(donor, out_dir=donor_dir)}
    ckpt = str(donor_dir / "best.pt")

    mom = prep(load_config(cdir / f"mom_ref6_{sfx}.yaml"), args.seed, args.smoke)
    mom["init_from"]["checkpoint"] = ckpt
    results["mom"] = run(mom)

    if sfx == "causal" and not args.skip_controls:
        cont = prep(load_config(cdir / "donor6_continue_causal.yaml"), args.seed, args.smoke)
        cont["init_from"]["checkpoint"] = ckpt
        results["donor_continue"] = run(cont)
        scratch = prep(load_config(cdir / "mom_scratch6_causal.yaml"), args.seed, args.smoke)
        if args.smoke:
            scratch["train"]["max_steps"] = 200
        results["mom_scratch"] = run(scratch)

    summary = {k: {"best_valid_brier": v["best_valid_brier"], "valid_acc": v["valid"]["acc"],
                   **{p: m["acc"] for p, m in v.get("test", {}).items()}} for k, v in results.items()}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
