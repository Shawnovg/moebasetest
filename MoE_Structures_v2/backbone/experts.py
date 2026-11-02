"""Expert bookkeeping shared by MoE, MoA and MoMHA.

Ported from the released code of SUT / MoA (same authors, same file layout):
  SUT@192cbd3  sut_layer/parallel_linear/parallel_experts/{moe.py, parallel_experts.py}
  MoA@96c33ca  moa_layer/parallel_linear/{moe.py, parallel_experts.py}

* ``ParallelExperts``: E independent linear maps applied to expert-sorted rows. Upstream uses a
  hand-written autograd ``ParallelLinear``; here the same split/matmul loop runs under ordinary
  autograd (identical values and gradients in float32).
* ``compute_gating``: SUT's helper that turns per-token top-k (gate, index) pairs into the
  expert-sorted dispatch order used by ``ParallelExperts`` (MoA inlines the same lines).
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


def compute_gating(k: int, probs: torch.Tensor, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor):
    zeros = torch.zeros_like(probs)
    gates = zeros.scatter(1, top_k_indices, top_k_gates)
    top_k_gates = top_k_gates.flatten()
    top_k_experts = top_k_indices.flatten()
    nonzeros = top_k_gates.nonzero().squeeze(-1)
    top_k_experts_nonzero = top_k_experts[nonzeros]
    _, _index_sorted_experts = top_k_experts_nonzero.sort(0)
    expert_size = (gates > 0).long().sum(0)
    index_sorted_experts = nonzeros[_index_sorted_experts]
    batch_index = index_sorted_experts.div(k, rounding_mode="trunc")
    batch_gates = top_k_gates[index_sorted_experts]
    return batch_gates, batch_index, expert_size, gates, index_sorted_experts


class ParallelExperts(nn.Module):
    def __init__(self, num_experts: int, input_size: int, output_size: int, bias: bool = False):
        super().__init__()
        self.w = nn.Parameter(torch.empty(num_experts, input_size, output_size))
        self.b = nn.Parameter(torch.zeros(num_experts, output_size)) if bias else None
        self.reset_parameters()

    def extra_repr(self):
        return "num_experts={}, input_size={}, output_size={}".format(*self.w.shape)

    def reset_parameters(self) -> None:
        # upstream: uniform(-1/fan_out_dim, 1/fan_out_dim) on w (dim 1 = input size)
        nn.init.uniform_(self.w, -1.0 / self.w.size(1), 1.0 / self.w.size(1))
        if self.b is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.b, -bound, bound)

    def forward(self, inputs: torch.Tensor, expert_size: torch.Tensor) -> torch.Tensor:
        sizes: List[int] = expert_size.tolist()
        chunks = inputs.split(sizes, dim=0)
        outs = []
        for i, chunk in enumerate(chunks):
            o = chunk @ self.w[i]
            if self.b is not None:
                o = o + self.b[i]
            outs.append(o)
        return torch.cat(outs, dim=0)
