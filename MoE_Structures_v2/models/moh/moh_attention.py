"""MoH: attention heads as experts, weighted per token by a router.

Reference code: MoH@2cefd57 ``MoH-DiT/models.py`` class ``MoHAttention`` (identical in MoH-ViT).
Kept as in the reference:

* one fused ``qkv`` projection for all H heads, each head with its own Q, K, V slice, and one
  ``proj`` (H*head_dim -> d) acting as the per-head output projections;
* routed heads: ``wg`` = Linear(d, H - shared, bias=False) on the token's own input (query-aware);
  softmax, top-``routed_head`` mask, gates renormalised (clamped denominator) and multiplied by
  ``routed_head`` so the gates sum to the number of active heads;
* shared heads (variant ``shared2``): ``wg_1`` softmax x shared_head, then the two-stage weight
  ``wg_0`` softmax x 2 scales shared vs routed gates;
* attention is computed for ALL heads, then each head's output is multiplied by its gate
  (0 for unselected heads) before ``proj`` -- exactly as the reference (no compute saving);
* balancing ('lb'): l_aux = mean(me * ce) * E^2 over the routed heads, weight 0.01 (MoH default
  ``moh_loss_coeff``), averaged over the layers that report one (MoH train.py divides by count).

Changes for this study: RoPE on q/k (MoH-DiT has none), padded keys masked, the balancing loss is
computed on non-padded tokens, random warm-up routing, ``route_record``.
Variants ``block2`` / ``block4`` (added 9 Nov, not in the MoH paper): the 8 heads are split into groups of
2 / 4 and the router selects whole groups (2 of 4 groups / 1 of 2 groups = 4 active heads), so each routed
unit is a multi-head attention block as in MoMHA, while keeping MoH's per-head q/k/v, linear router, gate
scaling and load-balance loss.
Variant ``block2_sut`` (added 9 Nov): block2 with MoMHA's MLP router, gates summing to 1 and switch loss
(0.01 per layer), so it differs from MoMHA only in K/V: each head keeps its own K/V instead of 2 shared heads.

Reference quirk logged in docs/deviations.md: with shared_head == 1 the reference never creates
``wg_1`` but uses it, so shared_head must be 0 or >= 2.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.layers import masked_softmax_attention
from backbone.routing import moh_loss, random_topk, switch_loss


class MoHAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim, rope, state, shared_head=0, routed_head=4,
                 qkv_bias=False, lb_coef=0.01, head_group=1, router="linear", gate_scale="k", lb_type="moh"):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, head_dim * num_heads * 3, bias=qkv_bias)
        self.proj = nn.Linear(head_dim * num_heads, dim)
        self.shared_head, self.routed_head = shared_head, routed_head
        assert shared_head != 1, "reference MoHAttention requires shared_head == 0 or >= 2"
        # head_group > 1 (variants block2 / block4): the router picks groups of head_group heads, so each
        # routed unit is a small multi-head attention, as in MoMHA; one gate per group, shared by its heads.
        self.head_group = head_group
        if head_group > 1:
            assert shared_head == 0 and num_heads % head_group == 0 and routed_head % head_group == 0
        if routed_head > 0:
            n_units = (num_heads - shared_head) // head_group
            if router == "mlp":     # SUT / MoMHA router: Linear(d, d) -> Tanh -> Linear(d, E)
                self.wg = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, n_units, bias=False))
            else:
                self.wg = nn.Linear(dim, n_units, bias=False)
            if shared_head > 0:
                self.wg_0 = nn.Linear(dim, 2, bias=False)
        if shared_head > 1:
            self.wg_1 = nn.Linear(dim, shared_head, bias=False)
        self.rope = rope
        self.lb_coef = lb_coef
        # variant block2_sut: MoMHA's gate ("sum1": selected gates sum to 1, no x k) and switch loss
        assert gate_scale in ("k", "sum1") and lb_type in ("moh", "switch") and router in ("linear", "mlp")
        self.gate_scale, self.lb_type = gate_scale, lb_type
        self.state = [state]
        self.num_units = num_heads // head_group   # record holds head (or head-group) ids, shared heads first
        self.route_record = None

    def forward(self, x, key_pad=None, state=None):
        state = self.state[0]
        B, N, C = x.shape
        _x = x.reshape(B * N, C)
        valid = None if key_pad is None else (~key_pad).reshape(-1)
        aux = x.new_zeros(())

        if self.routed_head > 0:
            logits = self.wg(_x)
            gates = F.softmax(logits, dim=1)
            num_tokens, num_experts = gates.shape
            n_sel = self.routed_head // self.head_group
            if state.random_active:
                indices = random_topk(num_tokens, num_experts, n_sel, x.device, state.generator)
            else:
                _, indices = torch.topk(gates, k=n_sel, dim=1)
            mask = F.one_hot(indices, num_classes=num_experts).sum(dim=1)
            if self.training and state.lb:
                g, m = (gates, mask) if valid is None else (gates[valid], mask[valid])
                if self.lb_type == "switch":
                    aux = aux + self.lb_coef * switch_loss(g, m.sum(0), num_experts)
                else:
                    aux = aux + self.lb_coef * moh_loss(g, m)
            routed_head_gates = gates * mask
            denom_s = torch.sum(routed_head_gates, dim=1, keepdim=True)
            denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
            if n_sel > 1:   # with one selected unit renormalising makes the gate exactly 1 (no router gradient):
                routed_head_gates = routed_head_gates / denom_s   # keep the raw probability, as MoMHA top1
            routed_head_gates = routed_head_gates.reshape(B, N, -1) * (n_sel if self.gate_scale == "k" else 1)
            if self.head_group > 1:     # one gate per group -> the same gate on each of its heads
                routed_head_gates = routed_head_gates.repeat_interleave(self.head_group, dim=2)
            if state.record:
                rec = indices.clone() + self.shared_head
                if valid is not None:
                    rec[~valid] = -1
                self.route_record = rec.view(B, N, -1)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.rope(q), self.rope(k)
        out = masked_softmax_attention(q, k, v, key_pad, self.scale, causal=state.causal)   # B, H, N, hd

        out = out.transpose(1, 2)  # B, N, H, hd
        if self.routed_head > 0:
            if self.shared_head > 0:
                shared_head_gates = F.softmax(self.wg_1(_x), dim=1).reshape(B, N, -1) * self.shared_head
                weight_0 = F.softmax(self.wg_0(_x), dim=1).reshape(B, N, 2) * 2
                shared_head_gates = torch.einsum("bn,bne->bne", weight_0[:, :, 0], shared_head_gates)
                routed_head_gates = torch.einsum("bn,bne->bne", weight_0[:, :, 1], routed_head_gates)
                masked_gates = torch.cat([shared_head_gates, routed_head_gates], dim=2)
            else:
                masked_gates = routed_head_gates
        else:
            masked_gates = F.softmax(self.wg_1(_x), dim=1).reshape(B, N, -1) * self.shared_head
        out = torch.einsum("bne,bned->bned", masked_gates, out).reshape(B, N, self.head_dim * self.num_heads)
        return self.proj(out), aux
