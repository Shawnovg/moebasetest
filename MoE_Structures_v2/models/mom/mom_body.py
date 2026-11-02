"""MoM: Mixture-of-Modules -- per token, per step, pick which attention module and which FFN module run.

Reference code: Mixture-of-Modules@2ed1d28 ``mom/model.py`` (``MoTAttention``, ``MoTMLP``,
``MoTBlock``) and ``mom/utils.py`` (``TokenRouter``, ``GRUTokenRouter``,
``load_balancing_loss_func``). Kept as in the reference:

* one pool of ``num_attn`` attention modules and ``num_ffn`` FFN modules, applied for
  ``depth`` assembly steps (here 2 = the depth of every other structure);
* each module has its own LayerNorm (pre-LN inside the module, no block LayerNorm);
* attention step: qkv = sum over the token's chosen modules of c_attn_m(LN_m(x)) (scores NOT
  applied, ``weighted_qkv`` False); one multi-head attention on the summed q/k/v; output
  o = sum_m r_m * c_proj_m(attn) where r_m is the router's RAW score (``use_ste`` False);
* FFN step: f = sum_m r_m * FFN_m(LN_m(u));
* routers: 'naive' TokenRouter Linear(d, 256) -> ReLU -> Linear(256, M + 1) (default here) or the
  hand-written GRU router whose state is carried across the steps (variant ``gru``); separate
  routers for attention and FFN, shared over steps (one router module per type, as upstream);
* the last router class is SKIP (``enable_pause``): no module runs; a token that skips attention
  contributes q = k = v = 0 (it is still a zero key/value for other tokens) and gets no update;
* top-K choice over the scores, K = ``routing_top_k`` (1 by default, variant ``k2``);
* GPT-2 initialisation (normal 0.02, zero biases) of every MoM module;
* balancing ('lb'): ``load_balancing_loss_func`` on the concatenation of every step's scores,
  separately for attention and FFN, summed, weight ``router_aux_loss_coef`` 0.01.

Changes for this study (docs/deviations.md): trained from scratch (no donor model, no
bottom / anchor / top dense layers -- the supervisor asked for a structure-only comparison);
RoPE on q/k instead of GPT-2 learned positions; bidirectional attention with padding mask
instead of causal; random warm-up routing; ``route_record``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.layers import ACTIVATIONS, masked_softmax_attention
from backbone.routing import mom_loss


class TokenRouter(nn.Module):
    """'naive' router: two-layer MLP over the token's current hidden state."""

    def __init__(self, d, router_dim, n_classes):
        super().__init__()
        self.router = nn.Sequential(nn.Linear(d, router_dim), nn.ReLU(), nn.Linear(router_dim, n_classes))

    def forward(self, x, h=None):
        return self.router(x), None


class GRUTokenRouter(nn.Module):
    """'gru' router: GRU cell (hand-written as upstream) whose state follows the token across steps."""

    def __init__(self, d, router_dim, n_classes):
        super().__init__()
        self.router_dim = router_dim
        self.router = nn.Linear(router_dim, n_classes)
        self.h2h = nn.Linear(router_dim, 3 * router_dim)
        self.x2h = nn.Linear(d, 3 * router_dim)

    def forward(self, x, h=None):
        if h is None:
            h = x.new_zeros(x.shape[0], self.router_dim)
        i_r, i_i, i_n = self.x2h(x).chunk(3, -1)
        h_r, h_i, h_n = self.h2h(h).chunk(3, -1)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        h = newgate + inputgate * (h - newgate)
        return self.router(h), h


ROUTERS = {"naive": TokenRouter, "gru": GRUTokenRouter}


def _choose(scores, k, state, n_real):
    """Top-k over the scores (or random during warm-up). Returns gates (raw scores) and ids, N x k."""
    if state.random_active:
        ids = torch.rand(scores.shape, device=scores.device, generator=state.generator).topk(k, -1).indices
        return scores.gather(1, ids), ids
    return torch.topk(scores, k, dim=-1)


class MoMBody(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, d_ff, rope, state, num_attn=4, num_ffn=4, depth=2,
                 routing_top_k=1, enable_pause=True, router_type="naive", router_dim=256,
                 aux_coef=0.01, activation="gelu", eps=1e-5, init_std=0.02):
        super().__init__()
        assert num_attn == num_ffn or router_type == "gru", "naive router outputs num_ffn + 1 classes"
        self.d, self.H, self.hd = d_model, n_heads, head_dim
        self.num_attn, self.num_ffn, self.depth = num_attn, num_ffn, depth
        self.enable_pause = enable_pause
        n_classes = (max(num_attn, num_ffn) if router_type == "gru" else num_ffn) + (1 if enable_pause else 0)
        self.n_classes = n_classes
        self.k = min(routing_top_k, n_classes)
        self.aux_coef = aux_coef
        inner = n_heads * head_dim
        self.attn_ln = nn.ModuleList([nn.LayerNorm(d_model, eps=eps) for _ in range(num_attn)])
        self.c_attn = nn.ModuleList([nn.Linear(d_model, 3 * inner) for _ in range(num_attn)])
        self.attn_proj = nn.ModuleList([nn.Linear(inner, d_model) for _ in range(num_attn)])
        self.ffn_ln = nn.ModuleList([nn.LayerNorm(d_model, eps=eps) for _ in range(num_ffn)])
        self.c_fc = nn.ModuleList([nn.Linear(d_model, d_ff) for _ in range(num_ffn)])
        self.ffn_proj = nn.ModuleList([nn.Linear(d_ff, d_model) for _ in range(num_ffn)])
        self.attn_router = ROUTERS[router_type](d_model, router_dim, n_classes)
        self.ffn_router = ROUTERS[router_type](d_model, router_dim, n_classes)
        self.act = ACTIVATIONS[activation]
        self.rope = rope
        self.state = [state]
        self.num_units = n_classes      # record: B x T x steps x (attn, ffn) x k; SKIP = n_classes - 1
        self.route_record = None
        for m in self.modules():                         # GPT-2 _init_weights
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _attention(self, x, key_pad, state, h_router):
        B, T, d = x.shape
        xf = x.reshape(B * T, d)
        scores, h_router = self.attn_router(xf, h_router)
        gates, ids = _choose(scores, self.k, state, self.num_attn)
        qkv = xf.new_zeros(B * T, 3 * self.H * self.hd)
        for m in range(self.num_attn):
            rows, slot = torch.where(ids == m)
            if rows.numel() == 0:
                continue
            qkv = qkv.index_add(0, rows, self.c_attn[m](self.attn_ln[m](xf[rows])))
        q, k, v = qkv.view(B, T, 3, self.H, self.hd).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.rope(q), self.rope(k)
        a = masked_softmax_attention(q, k, v, key_pad, self.hd ** -0.5)
        a = a.transpose(1, 2).reshape(B * T, self.H * self.hd)
        out = xf.new_zeros(B * T, d)
        for m in range(self.num_attn):
            rows, slot = torch.where(ids == m)
            if rows.numel() == 0:
                continue
            out = out.index_add(0, rows, self.attn_proj[m](a[rows]) * gates[rows, slot, None])
        return out.view(B, T, d), scores, ids, h_router

    def _ffn(self, x, state, h_router):
        B, T, d = x.shape
        xf = x.reshape(B * T, d)
        scores, h_router = self.ffn_router(xf, h_router)
        gates, ids = _choose(scores, self.k, state, self.num_ffn)
        out = xf.new_zeros(B * T, d)
        for m in range(self.num_ffn):
            rows, slot = torch.where(ids == m)
            if rows.numel() == 0:
                continue
            y = self.ffn_proj[m](self.act(self.c_fc[m](self.ffn_ln[m](xf[rows]))))
            out = out.index_add(0, rows, y * gates[rows, slot, None])
        return out.view(B, T, d), scores, ids, h_router

    def forward(self, x, key_pad, state, step_hook=None):
        state = self.state[0]
        B, T, _ = x.shape
        h_a = h_f = None
        attn_scores, ffn_scores, records = [], [], []
        for step in range(self.depth):
            o, s_a, ids_a, h_a = self._attention(x, key_pad, state, h_a)
            x = x + o
            f, s_f, ids_f, h_f = self._ffn(x, state, h_f)
            x = x + f
            attn_scores.append(s_a)
            ffn_scores.append(s_f)
            records.append(torch.stack([ids_a.view(B, T, self.k), ids_f.view(B, T, self.k)], 2))  # B,T,2,k
            if step_hook is not None:
                x = step_hook(step, x)
        aux = x.new_zeros(())
        if state.lb:
            valid = None if key_pad is None else (~key_pad).reshape(-1)
            for scores in (attn_scores, ffn_scores):
                cat = torch.cat([s if valid is None else s[valid] for s in scores], 0)
                aux = aux + self.aux_coef * mom_loss(cat, self.n_classes)
        if state.record:
            rec = torch.stack(records, 2)                 # B,T,steps,2(attn/ffn),k ; SKIP = n_classes-1
            if key_pad is not None:
                rec[key_pad] = -1
            self.route_record = rec
        return x, aux
