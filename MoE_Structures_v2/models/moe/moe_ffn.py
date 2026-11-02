"""MoE: the FFN sublayer replaced by E small experts, top-k per token.

Reference code: SUT@192cbd3 ``sut_layer/parallel_linear/parallel_experts/moe.py`` (class ``MoE``:
``top_k_gating`` + ``forward``), which is also MoA@96c33ca's ``moa_layer/parallel_linear/moe.py``
with an MLP gate and renormalised top-k gates. Kept line-for-line:

* gate network: ``mlp`` = Linear(d, d) -> Tanh -> Linear(d, E, bias=False)   (SUT default
  ``mlp_router=True``); ``linear`` = Linear(d, E, bias=False) (Switch / MoA form, variant top1)
* probabilities = softmax(logits); padded tokens get probability 0 (SUT ``skip_mask``)
* top-k, then gate normalisation: ``renorm`` = g / (sum g + 1e-6) (SUT), ``raw`` = the softmax
  probabilities themselves (Switch top-1; MoA code)
* expert-sorted dispatch through ``ParallelExperts`` (bias=True for FFN experts, as SUT's layer),
  activation, output experts, outputs scaled by the gate and summed back per token
* balancing ('lb' mode): ``switchloss`` * switch_loss (+ ``zloss`` * z_loss if set)

Additions for this study (not in the reference): random warm-up routing, hash routing
(variant ``hash``: fixed random token-id -> expert table, equal gates 1/k, no router; Roller et al.
2021), and a ``route_record`` for analysis.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.experts import ParallelExperts, compute_gating
from backbone.layers import ACTIVATIONS
from backbone.routing import random_topk, switch_loss, z_loss


class MoEFFN(nn.Module):
    def __init__(self, d_model, expert_hidden, num_experts, k, state, gate="mlp", gate_norm="renorm",
                 bias=True, activation="gelu", switchloss=0.01, zloss=0.0, routing="learned",
                 vocab_size=None, hash_seed=0):
        super().__init__()
        self.input_size, self.num_experts, self.k = d_model, num_experts, min(k, num_experts)
        self.experts = ParallelExperts(num_experts, d_model, expert_hidden, bias)
        self.output_experts = ParallelExperts(num_experts, expert_hidden, d_model, bias)
        self.activation = ACTIVATIONS[activation]
        self.gate_norm, self.routing = gate_norm, routing
        self.switchloss, self.zloss = switchloss, zloss
        if routing == "hash":
            self.w_gate = None  # no learned router
        elif gate == "mlp":
            self.w_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(),
                                        nn.Linear(d_model, num_experts, bias=False))
        else:
            self.w_gate = nn.Linear(d_model, num_experts, bias=False)
        if routing == "hash":
            g = torch.Generator().manual_seed(hash_seed)
            table = torch.stack([torch.randperm(num_experts, generator=g)[: self.k] for _ in range(vocab_size)])
            self.register_buffer("hash_table", table, persistent=True)
        self.state = [state]  # list: keep out of the module tree
        self.num_units = num_experts
        self.route_record = None

    def forward(self, x, key_pad=None, state=None):
        state = self.state[0]
        B, T, d = x.shape
        x = x.reshape(-1, d)
        valid = None if key_pad is None else (~key_pad).reshape(-1)

        if self.w_gate is None:  # hash: uniform 'probs' only feed compute_gating's bookkeeping
            logits = x.new_zeros(x.size(0), self.num_experts)
        else:
            logits = self.w_gate(x)
        probs = torch.softmax(logits, dim=1)
        if valid is not None:
            probs = probs.masked_fill(~valid[:, None], 0)
            logits = logits.masked_fill(~valid[:, None], 0)
        if self.routing == "hash":
            top_k_indices = self.hash_table[state.token_ids.reshape(-1)]
            top_k_gates = torch.full(top_k_indices.shape, 1.0 / self.k, device=x.device, dtype=x.dtype)
            if valid is not None:
                top_k_gates = top_k_gates * valid[:, None].to(x.dtype)
        else:
            if state.random_active:
                top_k_indices = random_topk(x.size(0), self.num_experts, self.k, x.device, state.generator)
                top_k_gates = probs.gather(1, top_k_indices)
            else:
                top_k_gates, top_k_indices = probs.topk(self.k, dim=1)
            if self.gate_norm == "renorm":
                top_k_gates = top_k_gates / (top_k_gates.sum(dim=1, keepdim=True) + 1e-6)

        batch_gates, batch_index, expert_size, gates, index_sorted = compute_gating(
            self.k, probs, top_k_gates, top_k_indices)
        expert_inputs = x[batch_index]
        h = self.activation(self.experts(expert_inputs, expert_size))
        expert_outputs = self.output_experts(h, expert_size) * batch_gates[:, None]
        y = torch.zeros((B * T, d), dtype=expert_outputs.dtype, device=x.device).index_add(0, batch_index, expert_outputs)

        aux = x.new_zeros(())
        if state.lb and self.routing != "hash":
            aux = aux + self.switchloss * switch_loss(probs, expert_size, self.num_experts)
            if self.zloss:
                aux = aux + self.zloss * z_loss(logits if valid is None else logits[valid])
        if state.record:
            rec = top_k_indices.clone()
            if valid is not None:
                rec[~valid] = -1
            self.route_record = rec.view(B, T, self.k)
        return y.view(B, T, d), aux
