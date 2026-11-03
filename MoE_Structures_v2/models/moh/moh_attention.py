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
Reference quirk logged in docs/deviations.md: with shared_head == 1 the reference never creates
``wg_1`` but uses it, so shared_head must be 0 or >= 2.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.layers import masked_softmax_attention
from backbone.routing import moh_loss, random_topk


class MoHAttention(nn.Module):
    def __init__(self, dim, num_heads, head_dim, rope, state, shared_head=0, routed_head=4,
                 qkv_bias=False, lb_coef=0.01):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, head_dim * num_heads * 3, bias=qkv_bias)
        self.proj = nn.Linear(head_dim * num_heads, dim)
        self.shared_head, self.routed_head = shared_head, routed_head
        assert shared_head != 1, "reference MoHAttention requires shared_head == 0 or >= 2"
        if routed_head > 0:
            self.wg = nn.Linear(dim, num_heads - shared_head, bias=False)
            if shared_head > 0:
                self.wg_0 = nn.Linear(dim, 2, bias=False)
        if shared_head > 1:
            self.wg_1 = nn.Linear(dim, shared_head, bias=False)
        self.rope = rope
        self.lb_coef = lb_coef
        self.state = [state]
        self.num_units = num_heads     # record holds head ids (shared heads first)
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
            if state.random_active:
                indices = random_topk(num_tokens, num_experts, self.routed_head, x.device, state.generator)
            else:
                _, indices = torch.topk(gates, k=self.routed_head, dim=1)
            mask = F.one_hot(indices, num_classes=num_experts).sum(dim=1)
            if self.training and state.lb:
                g, m = (gates, mask) if valid is None else (gates[valid], mask[valid])
                aux = aux + self.lb_coef * moh_loss(g, m)
            routed_head_gates = gates * mask
            denom_s = torch.sum(routed_head_gates, dim=1, keepdim=True)
            denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
            routed_head_gates = routed_head_gates / denom_s
            routed_head_gates = routed_head_gates.reshape(B, N, -1) * self.routed_head
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
