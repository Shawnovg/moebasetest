"""Evaluation metrics. Selection uses the Brier-style probability MSE (project rule, plan sec. 11)."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import torch

NUM_CLASSES = 7


def brier(probs: torch.Tensor, y: torch.Tensor) -> float:
    onehot = torch.nn.functional.one_hot(y, probs.size(-1)).float()
    return ((probs - onehot) ** 2).sum(-1).mean().item() / probs.size(-1)  # mean over classes and examples


def macro_f1(pred: torch.Tensor, y: torch.Tensor, k: int = NUM_CLASSES) -> float:
    f1s = []
    for c in range(k):
        tp = ((pred == c) & (y == c)).sum().item()
        fp = ((pred == c) & (y != c)).sum().item()
        fn = ((pred != c) & (y == c)).sum().item()
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / max(len(f1s), 1)


def ece(probs: torch.Tensor, y: torch.Tensor, bins: int = 15) -> float:
    conf, pred = probs.max(-1)
    acc = (pred == y).float()
    edges = torch.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.float().mean().item() * abs(acc[m].mean().item() - conf[m].mean().item())
    return total


def summarize(probs: torch.Tensor, y: torch.Tensor, len_bin: torch.Tensor, invalid: torch.Tensor = None,
              dup: torch.Tensor = None, depth: torch.Tensor = None) -> Dict:
    pred = probs.argmax(-1)
    out = {
        "n": int(y.numel()),
        "brier": brier(probs, y),
        "acc": (pred == y).float().mean().item(),
        "macro_f1": macro_f1(pred, y),
        "nll": torch.nn.functional.nll_loss(torch.log(probs.clamp_min(1e-12)), y).item(),
        "ece": ece(probs, y),
        "confusion": torch.bincount(y * NUM_CLASSES + pred, minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES).tolist(),
    }
    per_len = {}
    for b in sorted(set(len_bin.tolist())):
        m = len_bin == b
        per_len[int(b)] = {"n": int(m.sum()), "acc": (pred[m] == y[m]).float().mean().item()}
        if depth is not None:
            per_len[int(b)]["expected_depth"] = depth[m].mean().item()
    out["per_length"] = per_len
    if depth is not None:
        out["expected_depth"] = depth.mean().item()
    if dup is not None and dup.any():
        keep = ~dup
        out["n_swap_dup"] = int(dup.sum())
        out["acc_excl_swap_dup"] = (pred[keep] == y[keep]).float().mean().item()
        out["brier_excl_swap_dup"] = brier(probs[keep], y[keep])
    if invalid is not None:
        out["invalid_rate"] = invalid.float().mean().item()
    return out
