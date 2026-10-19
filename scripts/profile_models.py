"""Parameter counts, MAC estimates and measured latency for every config (plan sec. 10).

    python scripts/profile_models.py                    # all configs in configs/logic
    python scripts/profile_models.py --device cuda --lengths 32 64 120 --batches 1 128

Run this before training to check legal configurations, memory and throughput (the 'OOM and
throughput pilot'). Latency: warm-up, synchronised timing, median of repeats, eval mode.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moebase.compute import count_params, mom_macs, sut_block_macs  # noqa: E402
from moebase.data.logic import Vocab  # noqa: E402
from moebase.train.build import build_model  # noqa: E402
from moebase.train.config import load_config  # noqa: E402
from moebase.models.sut.model import clear_moe_statistics  # noqa: E402


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def make_inputs(kind, vocab, B, L, device):
    ids = torch.randint(5, 16, (B, L), device=device)  # symbol tokens only
    if kind == "sut_seq2seq":
        return (ids, torch.full((B, 2), vocab.eos, device=device))
    if kind.startswith("sut"):
        return (ids,)
    return (ids, torch.ones(B, L, device=device))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lengths", nargs="*", type=int, default=[32, 64, 120])
    ap.add_argument("--batches", nargs="*", type=int, default=[1, 128])
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()
    paths = args.configs or sorted(p for p in (ROOT / "configs" / "logic").glob("*.yaml") if not p.name.startswith("_"))
    vocab = Vocab()
    rows = []
    for p in paths:
        cfg = load_config(p)
        model, interface, mcfg = build_model(cfg["model"], vocab)
        model.to(args.device).eval()
        kind = cfg["model"]["kind"]
        row = {"config": p.stem, "kind": kind, **count_params(model)}
        for L in args.lengths:
            if kind.startswith("sut"):
                steps = mcfg.encoder_layers
                row[f"macs_L{L}_upper"] = sut_block_macs(mcfg.block, L, [L] * steps)["total"]
            else:
                row[f"macs_L{L}"] = mom_macs(mcfg, L)["total"] if kind.startswith("mom") else None
            for B in args.batches:
                x = make_inputs(kind, vocab, B, L, args.device)
                for _ in range(3):
                    model(*x)
                    clear_moe_statistics(model)
                times = []
                for _ in range(args.repeats):
                    sync(args.device); t = time.perf_counter()
                    model(*x)
                    sync(args.device); times.append(time.perf_counter() - t)
                    clear_moe_statistics(model)
                row[f"lat_ms_L{L}_B{B}"] = 1000 * statistics.median(times)
        if args.device.startswith("cuda"):
            row["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
            torch.cuda.reset_peak_memory_stats()
        rows.append(row)
        print(json.dumps(row))
    out = ROOT / "runs" / "profile.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"device": args.device, "torch": torch.__version__, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
