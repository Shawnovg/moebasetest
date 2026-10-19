"""Phase-1 donor: a vanilla GPT-2 (pre-LN) with HuggingFace GPT2LMHeadModel parameter names.

Upstream MoM initialises from HF GPT-2 checkpoints. Keeping HF's names
(``transformer.h.{i}.ln_1 / attn.c_attn / attn.c_proj / ln_2 / mlp.c_fc / mlp.c_proj``) lets the
upstream conversion map (convert.py) run unchanged. Block math equals ``MoTLayer``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .model import ACT, Conv1D, MoMConfig, gpt2_init_weights


class GPT2Attention(nn.Module):
    def __init__(self, config: MoMConfig):
        super().__init__()
        self.config = config
        n = config.n_positions
        self.register_buffer("bias", torch.tril(torch.ones((n, n), dtype=torch.bool)).view(1, 1, n, n), persistent=False)
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x, attention_mask=None):
        b, t, _ = x.shape
        q, k, v = self.c_attn(x).split(self.embed_dim, dim=2)
        q, k, v = (z.view(b, t, self.num_heads, self.head_dim).permute(0, 2, 1, 3) for z in (q, k, v))
        w = torch.matmul(q, k.transpose(-1, -2)) / torch.full([], v.size(-1) ** 0.5, dtype=q.dtype, device=q.device)
        if self.config.causal:
            mask = self.bias[:, :, :t, :t]
            w = torch.where(mask, w, torch.full([], torch.finfo(w.dtype).min, dtype=w.dtype, device=w.device))
        if attention_mask is not None:
            w = w + attention_mask
        w = self.attn_dropout(F.softmax(w, dim=-1))
        a = torch.matmul(w, v).permute(0, 2, 1, 3).contiguous().view(b, t, self.embed_dim)
        return self.resid_dropout(self.c_proj(a))


class GPT2MLP(nn.Module):
    def __init__(self, config: MoMConfig):
        super().__init__()
        inner = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.c_fc = Conv1D(inner, config.n_embd)
        self.c_proj = Conv1D(config.n_embd, inner)
        self.act = ACT[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class GPT2Block(nn.Module):
    def __init__(self, config: MoMConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask)
        return x + self.mlp(self.ln_2(x))


class GPT2Model(nn.Module):
    def __init__(self, config: MoMConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, attention_mask=None):
        b, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0)
        am = None
        if attention_mask is not None:
            am = attention_mask.view(b, -1)[:, None, None, :].to(self.wte.weight.dtype)
            am = (1.0 - am) * torch.finfo(self.wte.weight.dtype).min
        x = self.drop(self.wte(input_ids) + self.wpe(pos))
        for blk in self.h:
            x = blk(x, am)
        return self.ln_f(x)


class GPT2LMHead(nn.Module):
    """Dense donor for the causal reference path."""

    def __init__(self, config: MoMConfig):
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        gpt2_init_weights(self, config.initializer_range, config.n_layer)
        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = self.lm_head(self.transformer(input_ids, attention_mask))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[..., :-1, :].reshape(-1, logits.size(-1)), labels[..., 1:].reshape(-1))
        return {"loss": loss, "aux_loss": None, "logits": logits}


class GPT2Classifier(nn.Module):
    """Dense donor for the encoder adaptation (causal=False)."""

    def __init__(self, config: MoMConfig):
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.head = nn.Linear(config.n_embd, config.num_classes)
        gpt2_init_weights(self, config.initializer_range, config.n_layer)

    def forward(self, input_ids, attention_mask=None):
        return {"logits": self.head(self.transformer(input_ids, attention_mask)[:, 0]), "aux_loss": None}
