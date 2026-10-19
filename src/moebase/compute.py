"""Arithmetic estimates (MACs) for what the implementation executes, plus parameter counts.

Convention: one multiply-accumulate = 1 MAC; LayerNorm, softmax, activations, gate renormalisation
and dispatch/gather are not counted (report measured latency for those). Counts are per sequence
of length L for one forward pass; ``active`` gives the number of non-halted tokens at each step.
A dense-executed path is charged as dense.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from torch import nn


def count_params(model: nn.Module) -> Dict[str, int]:
    seen, total, emb = set(), 0, 0
    for n, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
        if "embed" in n or n.endswith("wte.weight") or n.endswith("wpe.weight"):
            emb += p.numel()
    return {"total": total, "non_embedding": total - emb}


def sut_block_macs(b, L: int, active: Sequence[float], src_len: Optional[int] = None, cross: bool = False) -> Dict[str, float]:
    """b: SUTBlockConfig. active[s] = dispatched tokens at step s (<= L)."""
    d, Ea, ka, Ed = b.embed_dim, b.attn_num_expert, b.attn_k, b.attn_expert_dim
    H = Ed // b.head_dim
    F, kf, Ef = b.ff_expert_dim, b.ff_k, b.ff_num_expert
    out = {"router": 0.0, "qo_proj": 0.0, "kv_proj": 0.0, "scores": 0.0, "ffn": 0.0, "relpos": 0.0}
    for a in active:
        # MLP routers run on all L tokens (masking happens after the gate): d*d + d*E each
        out["router"] += L * (d * d + d * Ea) + L * (d * d + d * Ef)
        out["qo_proj"] += a * ka * (d * Ed) * 2
        out["kv_proj"] += L * 2 * d * Ed  # computed for every key position
        # scores and weighted values are computed densely for all L queries x top_k x heads
        out["scores"] += L * ka * H * L * b.head_dim * 2
        out["relpos"] += L * ka * H * b.head_dim * (2 * b.max_rel_positions + 1)
        out["ffn"] += a * kf * (d * F) * 2
    out["total"] = sum(out.values())
    return out


def mom_macs(c, L: int) -> Dict[str, float]:
    """c: MoMConfig. Assumes every token selects routing_top_k real modules (SKIP lowers this)."""
    d, inner, K = c.n_embd, c.n_inner or 4 * c.n_embd, c.routing_top_k
    dense_layer = L * (3 * d * d + d * d + 2 * d * inner) + L * L * d * 2
    depth = c.depth_per_chunk if c.depth_per_chunk is not None else int(c.num_ffn * c.depth_multiplier)
    k = max(c.num_ffn, c.num_attn) + (1 if c.enable_pause else 0)
    router = L * (3 * c.router_dim * (d + c.router_dim) + c.router_dim * k) * 2  # attn + ffn GRU routers
    routed_step = L * K * (3 * d * d + d * d + 2 * d * inner) + L * L * d * 2 + router
    n_dense = c.num_bottom_layers + c.num_top_layers + (c.chunks if c.require_anchor_layer else 0)
    return {"dense_layers": n_dense * dense_layer, "routed_steps": c.chunks * depth * routed_step,
            "total": n_dense * dense_layer + c.chunks * depth * routed_step}
