"""MoA: attention experts that own only their query and output projections.

Reference code: MoA@96c33ca ``moa_layer/moa_attention.py`` (``MultiheadAttention``, self-attention
path) with ``moa_layer/parallel_linear/moe.py`` (``MoE.top_k_gating`` / ``map`` / ``reduce``).
Kept as in the reference:

* ``k_proj``, ``v_proj``: ONE shared key/value head, Linear(d, head_dim, bias=False), Xavier
  uniform with gain 1/sqrt(2);
* ``q_proj``: a MoE with E experts mapping d -> head_dim (``ParallelExperts``, no bias) and output
  experts head_dim -> d; the number of selected experts plays the role of the number of heads;
* gate: ``w_gate`` = zero-initialised parameter (d x E) -- so at the start every token sees equal
  probabilities -- ``logits = x @ w_gate``, softmax, padded tokens' probabilities set to 0,
  ``topk``; the code uses the RAW top-k probabilities as gates (no renormalisation);
* q is scaled by head_dim^-0.5, attention per selected expert against the shared K/V, then
  ``reduce``: output expert of each slot, multiplied by its gate, summed per token;
* balancing ('lb'): switchloss 0.01 * switch_loss + zloss 0.001 * z_loss (MoA ``run.sh``).

Variant ``sut`` (10 Nov, unified router): SUT/MoMHA MLP router with default random init, selected gates
renormalised to sum to 1 (with gradient), switch loss 0.01 only (no z-loss).
Variants: ``gate_norm: renorm_detached`` follows the paper text (gates renormalised over the
selected experts with the denominator detached); ``k`` = 8 of 8 is the paper's base model.
Changes for this study: RoPE on q/k (MoA used sinusoidal absolute positions), noisy gating off
(the reference run uses ``--gating normal``), random warm-up routing, ``route_record``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from backbone.experts import ParallelExperts, compute_gating
from backbone.layers import causal_mask
from backbone.routing import random_topk, switch_loss, z_loss


class MoAGate(nn.Module):
    """The q-projection MoE of MoA (map / reduce)."""

    def __init__(self, input_size, head_size, num_experts, k, gate_norm="raw", router="linear"):
        super().__init__()
        self.num_experts, self.input_size, self.head_size = num_experts, input_size, head_size
        self.router = router
        self.experts = ParallelExperts(num_experts, input_size, head_size, bias=False)
        self.output_experts = ParallelExperts(num_experts, head_size, input_size, bias=False)
        self.k = min(k, num_experts)
        self.gate_norm = gate_norm
        if router == "mlp":      # unified (SUT) router: Linear(d, d) -> Tanh -> Linear(d, E), PyTorch default init
            self.w_gate_mlp = nn.Sequential(nn.Linear(input_size, input_size), nn.Tanh(),
                                            nn.Linear(input_size, num_experts, bias=False))
        else:
            self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

    def top_k_gating(self, x, skip_mask, state):
        logits = self.w_gate_mlp(x) if self.router == "mlp" else x @ self.w_gate
        probs = torch.softmax(logits, dim=1)
        if skip_mask is not None:
            probs = torch.masked_fill(probs, skip_mask, 0)
        if state.random_active:
            top_k_indices = random_topk(x.size(0), self.num_experts, self.k, x.device, state.generator)
            top_k_gates = probs.gather(1, top_k_indices)
        else:
            top_k_gates, top_k_indices = probs.topk(self.k, dim=1)
        if self.gate_norm == "renorm_detached":
            top_k_gates = top_k_gates / top_k_gates.sum(dim=1, keepdim=True).detach().clamp_min(1e-9)
        elif self.gate_norm == "renorm":        # as MoMHA / SUT: selected gates sum to 1, with gradient
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


class MoAAttention(nn.Module):
    def __init__(self, embed_dim, head_dim, num_expert, k, rope, state, gate_norm="raw",
                 switchloss=0.01, zloss=0.001, router="linear"):
        super().__init__()
        self.head_dim, self.k = head_dim, k
        self.scaling = head_dim ** -0.5
        self.k_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.q_proj = MoAGate(embed_dim, head_dim, num_expert, k, gate_norm, router)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        self.rope = rope
        self.switchloss, self.zloss = switchloss, zloss
        self.num_units = num_expert
        self.state = [state]
        self.route_record = None

    def forward(self, x, key_pad=None, state=None):
        state = self.state[0]
        B, T, _ = x.shape
        q, logits, probs = self.q_proj.map(x, key_pad, state)        # B, T, k, hd
        k = self.k_proj(x)                                            # B, T, hd
        v = self.v_proj(x)
        q = q * self.scaling
        q = self.rope(q.transpose(1, 2))                              # B, k, T, hd
        k = self.rope(k)
        w = torch.einsum("bkie,bje->bkij", q, k)
        if key_pad is not None:
            w = w.masked_fill(key_pad[:, None, None, :], float("-inf"))
        if state.causal:
            w = w.masked_fill(causal_mask(T, T, x.device), float("-inf"))
        p = torch.softmax(w.float(), dim=-1).type_as(w)
        attn = torch.einsum("bkij,bje->bike", p, v)                  # B, T, k, hd
        out = self.q_proj.reduce(attn)

        aux = x.new_zeros(())
        if state.lb:
            aux = aux + self.switchloss * switch_loss(probs, self.q_proj.expert_size, self.q_proj.num_experts)
            valid = None if key_pad is None else (~key_pad).reshape(-1)
            if self.zloss:
                aux = aux + self.zloss * z_loss(logits if valid is None else logits[valid])
        if state.record:
            rec = self.q_proj.top_k_indices.clone()
            if key_pad is not None:
                rec[key_pad.reshape(-1)] = -1
            self.route_record = rec.view(B, T, self.k)
        return out, aux
