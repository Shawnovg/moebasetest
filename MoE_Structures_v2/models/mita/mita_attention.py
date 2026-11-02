"""MiTA: Mixture-of-Top-k Attention -- key/value pairs as experts, chosen per query.

Reference code: QishuaiWen/MiTA@95460f7 ``mita/mita_attention.py`` (``MiTA_Attention.mita``,
router_topk = 1) and ``mita/_mita_attention.py`` (router_topk > 1), merged by
``mita/mixed_attention.py`` (``MixedAttention``, FlashAttention varlen + online-softmax merge).

What the reference computes, per head (kept exactly):

1. landmark queries ``router`` = average pooling of the queries over m equal windows;
2. each landmark scores all keys (router @ K^T) and keeps its top-``kv_topk`` keys and values:
   that set is one routed expert;
3. compressed shared expert: landmark values ``agent_value = SDPA(router, K, V)``; the landmarks,
   scaled by d^-0.5, serve as keys -- so the shared logits are q . router * d^-0.5 * d^-0.5
   (the reference passes ``router * scale`` and FlashAttention scales again);
4. each query is routed to the landmark(s) with the largest ``router @ q`` (argmax, s = 1;
   top-s in the s > 1 file);
5. ONE softmax over the concatenation [shared m entries ; the k pairs of each chosen expert]
   (MixedAttention merges the two FlashAttention calls with their log-sum-exp, which equals one
   joint softmax; with s > 1 a key chosen by two experts appears twice, as in the reference);
6. heads merged and projected by ``proj`` (bias=True); ``qkv`` bias False.

This file computes steps 1-6 with dense PyTorch ops instead of FlashAttention (CPU, no CUDA),
which gives the same numbers; tests/test_mita.py checks it against a per-query loop of the
same equations. MiTA has no learned router and no balancing loss.

Changes for this study (all logged in docs/deviations.md):
* 1-D sequences: windows are equal slices of each sequence's non-padded length (the reference
  pools a 2-D image grid with AdaptiveAvgPool2d and drops the last token);
* RoPE is applied to q and k before pooling (the vision models use absolute embeddings);
* padded keys can never be chosen unless a sequence is shorter than kv_topk, and are masked;
* 'lb' mode (adaptation, no reference): Switch loss on softmax(router @ q) over landmarks;
* 'warmup' mode: queries are sent to random landmarks during the warm-up;
* variant ``no_shared`` drops the compressed shared part (routed experts only).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def window_pool_weights(valid_len: torch.Tensor, T: int, m: int) -> torch.Tensor:
    """B x m x T averaging weights: window i covers [floor(i L/m), ceil((i+1) L/m)) of length L."""
    B = valid_len.shape[0]
    L = valid_len.float()[:, None]                                    # B x 1
    i = torch.arange(m, device=valid_len.device).float()[None, :]     # 1 x m
    start = torch.floor(i * L / m)
    end = torch.ceil((i + 1) * L / m)
    t = torch.arange(T, device=valid_len.device).float()[None, None, :]
    inside = (t >= start[..., None]) & (t < end[..., None])
    w = inside.float()
    return w / w.sum(-1, keepdim=True).clamp_min(1.0)


class MiTAAttention(nn.Module):
    def __init__(self, dim, num_heads, num_landmarks, kv_topk, rope, state, router_topk=1,
                 shared=True, qkv_bias=False, proj_bias=True, lb_coef=0.01):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.m, self.kv_topk, self.router_topk = num_landmarks, kv_topk, router_topk
        self.shared = shared
        self.rope = rope
        self.lb_coef = lb_coef
        self.state = [state]
        self.num_units = num_landmarks  # record: B x T x heads landmark (expert) id
        self.route_record = None
        self.last_gate = None

    def forward(self, x, key_pad=None, state=None):
        state = self.state[0]
        B, N, C = x.shape
        H, d, m, s = self.num_heads, self.head_dim, self.m, self.router_topk
        q, k, v = self.qkv(x).reshape(B, N, 3, H, d).permute(2, 0, 3, 1, 4).unbind(0)   # B,H,N,d
        q, k = self.rope(q), self.rope(k)

        valid_len = torch.full((B,), N, device=x.device) if key_pad is None else (~key_pad).sum(1)
        pool = window_pool_weights(valid_len, N, m)                    # B,m,N
        router = torch.einsum("bmn,bhnd->bhmd", pool, q)              # landmark queries B,H,m,d

        # routed experts: each landmark's top-k keys
        r_attn_key = router @ k.transpose(-2, -1)                     # B,H,m,N
        if key_pad is not None:
            r_attn_key = r_attn_key.masked_fill(key_pad[:, None, None, :], float("-inf"))
        kk = min(self.kv_topk, N)
        _, topk_key_idx = r_attn_key.topk(kk, dim=-1)                 # B,H,m,kk

        # shared expert: landmark values via full attention, landmarks as keys
        agent_value = F.scaled_dot_product_attention(
            router, k, v, attn_mask=None if key_pad is None else ~key_pad[:, None, None, :])

        # query -> expert assignment
        gate = router @ q.transpose(-2, -1)                           # B,H,m,N
        if state.random_active:
            # s distinct random landmarks per query (top-s of uniform noise)
            expert_idx = torch.rand(B, H, N, m, device=x.device, generator=state.generator).topk(s, -1).indices
        else:
            expert_idx = gate.topk(s, dim=-2).indices.permute(0, 1, 3, 2)   # B,H,N,s

        # gather each query's routed key/value pairs: B,H,N,s*kk
        sel = torch.gather(topk_key_idx.unsqueeze(2).expand(B, H, N, m, kk), 3,
                           expert_idx.unsqueeze(-1).expand(B, H, N, s, kk)).reshape(B, H, N, s * kk)
        base = (torch.arange(B * H, device=x.device) * N).view(B, H, 1, 1)
        flat = (sel + base).reshape(-1)
        k_sel = k.reshape(B * H * N, d).index_select(0, flat).view(B, H, N, -1, d)
        v_sel = v.reshape(B * H * N, d).index_select(0, flat).view(B, H, N, -1, d)
        routed_logits = torch.einsum("bhnd,bhnjd->bhnj", q, k_sel) * self.scale
        if key_pad is not None:
            sel_pad = torch.gather(key_pad[:, None, None, :].expand(B, H, N, N), 3, sel)
            routed_logits = routed_logits.masked_fill(sel_pad, float("-inf"))

        if self.shared:
            shared_logits = torch.einsum("bhnd,bhmd->bhnm", q, router * self.scale) * self.scale
            logits = torch.cat([shared_logits, routed_logits], dim=-1)
        else:
            logits = routed_logits
        p = torch.softmax(logits.float(), dim=-1).type_as(q)
        if self.shared:
            out = torch.einsum("bhnm,bhmd->bhnd", p[..., :m], agent_value) + \
                torch.einsum("bhnj,bhnjd->bhnd", p[..., m:], v_sel)
        else:
            out = torch.einsum("bhnj,bhnjd->bhnd", p, v_sel)
        out = self.proj(out.transpose(1, 2).reshape(B, N, C))

        aux = x.new_zeros(())
        if state.lb:  # adaptation: Switch loss on the landmark scores, per head, valid tokens only
            probs = torch.softmax(gate, dim=-2).permute(0, 1, 3, 2).reshape(B, H, N, m)
            onehot = F.one_hot(expert_idx[..., 0], m).float()
            if key_pad is not None:
                vm = (~key_pad)[:, None, :, None].float()
                f = (onehot * vm).sum(2) / vm.sum(2)
                P = (probs * vm).sum(2) / vm.sum(2)
            else:
                f, P = onehot.mean(2), probs.mean(2)
            aux = aux + self.lb_coef * (m * (f * P).sum(-1)).mean()
        if state.record:
            rec = expert_idx[..., 0].permute(0, 2, 1).clone()           # B,N,H landmark id per head
            if key_pad is not None:
                rec[key_pad] = -1
            self.route_record = rec
        return out, aux
