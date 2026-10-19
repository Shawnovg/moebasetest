"""Mixture of multi-head attention experts (MoMHA) used by SUT.

Port of SUT@192cbd3 ``sut_layer/moa_attention.py`` (class ``MultiheadAttention``) without fairseq.

What the released code does (and what this port keeps):

* Only the query projection and the output projection are experts: ``q_proj`` is a ``MoE`` whose
  ``map`` gives each selected expert's query (``expert_dim = heads_per_expert * head_dim``) and whose
  ``reduce`` applies that expert's output projection, gate-weighted.
* Keys and values are dense and shared by all experts: ``k_proj, v_proj: embed_dim -> expert_dim``.
  So expert e, head h attends with its own query but the shared K/V slice of head h.
* Self-attention adds learned relative-position logits ``rel_pos_emb[h, d, 2*64+1]`` (zero init),
  computed from the scaled query; offsets are clamped to [-63, 63]. It does not restrict attention.
* Cross-attention (decoder) has no relative-position term and uses ``v = v_proj(key)``.
* Tokens with ``skip_mask == 0`` (padding or halted) are never dispatched: their query is zero and
  their output is exactly zero.

Removed (not used by single-pass training/evaluation): incremental decoding state, ONNX tracing,
quant-noise wrappers (``q_noise`` defaults to 0 upstream), ``add_bias_kv``/``add_zero_attn``
(both False in SUT configs).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .moe import MoE


class MoMHA(nn.Module):
    def __init__(self, embed_dim, kdim=None, vdim=None, dropout=0.0, bias=False,
                 self_attention=False, encoder_decoder_attention=False,
                 num_expert=12, top_k=4, expert_dim=256, head_dim=64,
                 cvloss=0, switchloss=0, zloss=0, miloss=0, sample_topk=0, gating_dropout=0,
                 max_positions=64, mim_variant="code"):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.top_k = top_k
        self.dropout_module = nn.Dropout(dropout)
        assert expert_dim % head_dim == 0
        self.num_heads = expert_dim // head_dim
        self.head_dim = head_dim
        self.expert_dim = expert_dim
        self.scaling = self.head_dim ** -0.5
        self.max_positions = max_positions
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert not self.self_attention or self.qkv_same_dim

        self.k_proj = nn.Linear(self.kdim, self.expert_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, self.expert_dim, bias=bias)
        self.q_proj = MoE(embed_dim, self.expert_dim, num_expert, k=top_k,
                          cvloss=cvloss, switchloss=switchloss, zloss=zloss, miloss=miloss,
                          gating_dropout=gating_dropout, noisy_gating=False, acc_aux_loss=True,
                          bias=bias, mim_variant=mim_variant)
        if self.self_attention:
            self.rel_pos_emb = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim, self.max_positions * 2 + 1))
        self.sample_topk = sample_topk
        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
        if self.self_attention:
            nn.init.zeros_(self.rel_pos_emb)

    def rel_pos_logits(self, query, length, last=False):
        device = query.device
        idx = torch.arange(length, dtype=torch.long, device=device)
        if not last:
            idx_grid = idx[None, :] - idx[:, None]
        else:
            idx_grid = idx[None, :] - (length - 1)
        idx_grid = torch.clamp(idx_grid, min=1 - self.max_positions, max=self.max_positions - 1) + self.max_positions
        logits = torch.einsum("bkhid,hdj->bkhij", query, self.rel_pos_emb)
        return logits.gather(-1, idx_grid[None, None, None, :, :].expand(
            logits.size(0), logits.size(1), logits.size(2), -1, -1))

    def forward(self, query: Tensor, key: Optional[Tensor], value: Optional[Tensor],
                key_padding_mask: Optional[Tensor] = None, skip_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None, need_weights: bool = False):
        """query/key/value: T x B x C. key_padding_mask: B x S (True = pad). skip_mask: T x B (0 = skip).

        attn_mask: additive float T x S. Returns (attn T x B x C, weights or None, aux_loss).
        """
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        src_len = key.size(0)

        q, aux_loss = self.q_proj.map(query, skip_mask=skip_mask, sample_topk=self.sample_topk)
        k = self.k_proj(key)
        v = self.v_proj(key) if self.encoder_decoder_attention else self.v_proj(value)
        q = q * self.scaling

        q = q.reshape(tgt_len, bsz, self.top_k, self.num_heads, self.head_dim).permute(1, 2, 3, 0, 4)
        k = k.reshape(-1, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.reshape(-1, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        total_bsz = bsz * self.top_k * self.num_heads
        attn_weights = torch.einsum("bkhie,bhje->bkhij", q, k).reshape(total_bsz, tgt_len, src_len)
        if self.self_attention:
            attn_weights = attn_weights + self.rel_pos_logits(q, src_len).view(total_bsz, tgt_len, src_len)
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask.unsqueeze(0)
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.top_k, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(key_padding_mask[:, None, None, None, :].to(torch.bool), float("-inf"))
            attn_weights = attn_weights.view(total_bsz, tgt_len, src_len)

        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)
        attn = torch.einsum("bkhij,bhje->bkhie",
                            attn_probs.view(bsz, self.top_k, self.num_heads, tgt_len, src_len), v)
        attn = self.q_proj.reduce(attn.permute(3, 0, 1, 2, 4).reshape(tgt_len, bsz, self.top_k, self.expert_dim))
        weights = attn_weights.view(bsz, self.top_k, self.num_heads, tgt_len, src_len) if need_weights else None
        return attn, weights, aux_loss
