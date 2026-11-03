"""MoMHA: SUT's mixture of multi-head attention -- experts are groups of heads.

Reference code: SUT@192cbd3 ``sut_layer/moa_attention.py`` (``MultiheadAttention``) with
``sut_layer/parallel_linear/parallel_experts/moe.py`` (``MoE.top_k_gating`` / ``map`` / ``reduce``).
Kept as in the reference:

* each expert owns a query projection d -> H*head_dim and an output projection H*head_dim -> d
  (``ParallelExperts``, bias=False); K and V are shared: Linear(d, H*head_dim, bias=False), Xavier
  uniform gain 1/sqrt(2), i.e. H shared key/value heads used by every expert's matching head;
* gate: SUT's MLP router Linear(d, d) -> Tanh -> Linear(d, E, bias=False) on the token's input,
  softmax, padded tokens' probabilities set to 0, top-k, gates renormalised g / (sum g + 1e-6);
* q scaled by head_dim^-0.5; expert e, head h attends with its own query against shared head h;
  ``reduce`` applies each selected expert's output projection, scaled by its gate, summed;
* balancing ('lb'): SUT's switchloss term, weight 0.01.

Variant ``top1``: with k = 1 SUT's renormalisation makes every gate exactly 1, so the router would
get no gradient; that variant therefore uses the raw probability as the gate (Switch convention).

Changes for this study: RoPE replaces SUT's learned relative-position logits (one position scheme
for all structures); no MIM loss (the supervisor asked for no extra loss unless in 'lb' mode);
random warm-up routing, ``route_record``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from backbone.experts import ParallelExperts, compute_gating
from backbone.layers import causal_mask
from backbone.routing import random_topk, switch_loss


class SUTGate(nn.Module):
    def __init__(self, input_size, head_size, num_experts, k, gate_norm="renorm"):
        super().__init__()
        self.gate_norm = gate_norm
        self.num_experts, self.input_size, self.head_size = num_experts, input_size, head_size
        self.experts = ParallelExperts(num_experts, input_size, head_size, bias=False)
        self.output_experts = ParallelExperts(num_experts, head_size, input_size, bias=False)
        self.k = min(k, num_experts)
        self.w_gate = nn.Sequential(nn.Linear(input_size, input_size), nn.Tanh(),
                                    nn.Linear(input_size, num_experts, bias=False))

    def top_k_gating(self, x, skip_mask, state):
        logits = self.w_gate(x)
        probs = torch.softmax(logits, dim=1)
        if skip_mask is not None:
            probs = torch.masked_fill(probs, skip_mask, 0)
        if state.random_active:
            top_k_indices = random_topk(x.size(0), self.num_experts, self.k, x.device, state.generator)
            top_k_gates = probs.gather(1, top_k_indices)
        else:
            top_k_gates, top_k_indices = probs.topk(self.k, dim=1)
        if self.gate_norm == "renorm":
            top_k_gates = top_k_gates / (top_k_gates.sum(dim=1, keepdim=True) + 1e-6)
        (self.batch_gates, self.batch_index, self.expert_size, gates,
         self.index_sorted_experts) = compute_gating(self.k, probs, top_k_gates, top_k_indices)
        self.top_k_indices = top_k_indices
        return logits, probs

    def map(self, x, skip_mask, state):
        bsz, length, emb = x.size()
        x = x.reshape(-1, emb)
        if skip_mask is not None:
            skip_mask = skip_mask.reshape(-1, 1)
        logits, probs = self.top_k_gating(x, skip_mask, state)
        expert_outputs = self.experts(x[self.batch_index], self.expert_size)
        y = torch.zeros((bsz * length * self.k, self.head_size), dtype=expert_outputs.dtype, device=x.device)
        y = y.index_add(0, self.index_sorted_experts, expert_outputs)
        return y.view(bsz, length, self.k, -1), logits, probs

    def reduce(self, x):
        bsz, length, k, emb = x.size()
        x = x.reshape(-1, emb)
        expert_outputs = self.output_experts(x[self.index_sorted_experts], self.expert_size)
        expert_outputs = expert_outputs * self.batch_gates[:, None]
        y = torch.zeros((bsz * length, self.input_size), dtype=expert_outputs.dtype, device=x.device)
        return y.index_add(0, self.batch_index, expert_outputs).view(bsz, length, self.input_size)


class MoMHAAttention(nn.Module):
    def __init__(self, embed_dim, head_dim, heads_per_expert, num_expert, k, rope, state, switchloss=0.01,
                 gate_norm="renorm"):
        super().__init__()
        self.num_heads, self.head_dim, self.top_k = heads_per_expert, head_dim, k
        self.expert_dim = heads_per_expert * head_dim
        self.scaling = head_dim ** -0.5
        self.k_proj = nn.Linear(embed_dim, self.expert_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.expert_dim, bias=False)
        self.q_proj = SUTGate(embed_dim, self.expert_dim, num_expert, k, gate_norm)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        self.rope = rope
        self.switchloss = switchloss
        self.state = [state]
        self.num_units = num_expert
        self.route_record = None

    def forward(self, x, key_pad=None, state=None):
        state = self.state[0]
        B, T, _ = x.shape
        q, logits, probs = self.q_proj.map(x, key_pad, state)          # B, T, k, H*hd
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q * self.scaling
        q = q.reshape(B, T, self.top_k, self.num_heads, self.head_dim).permute(0, 2, 3, 1, 4)  # B,k,H,T,hd
        k = k.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)               # B,H,T,hd
        v = v.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k = self.rope(q), self.rope(k)
        w = torch.einsum("bkhie,bhje->bkhij", q, k)
        if key_pad is not None:
            w = w.masked_fill(key_pad[:, None, None, None, :], float("-inf"))
        if state.causal:
            w = w.masked_fill(causal_mask(T, T, x.device), float("-inf"))
        p = torch.softmax(w.float(), dim=-1).type_as(w)
        attn = torch.einsum("bkhij,bhje->bkhie", p, v)                   # B,k,H,T,hd
        attn = attn.permute(0, 3, 1, 2, 4).reshape(B, T, self.top_k, self.expert_dim)
        out = self.q_proj.reduce(attn)

        aux = x.new_zeros(())
        if state.lb:
            aux = aux + self.switchloss * switch_loss(probs, self.q_proj.expert_size, self.q_proj.num_experts)
        if state.record:
            rec = self.q_proj.top_k_indices.clone()
            if key_pad is not None:
                rec[key_pad.reshape(-1)] = -1
            self.route_record = rec.view(B, T, self.top_k)
        return out, aux
