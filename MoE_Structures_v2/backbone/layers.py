"""Shared building blocks: rotary positions, softmax attention, dense attention and dense FFN.

Every attention variant (dense, MoH, MoA, MoMHA, MiTA, MoM modules) applies the same rotary
position embedding to its queries and keys, so position handling is identical across structures.
Attention is bidirectional over the prompt; padded keys are masked.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Rotary(nn.Module):
    """RoPE (Su et al.), rotate-half convention, applied to the last dim of q / k."""

    def __init__(self, head_dim: int, base: float = 10000.0, max_len: int = 512):
        super().__init__()
        assert head_dim % 2 == 0
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor, seq_dim: int = -2) -> torch.Tensor:
        """x: [..., T, D] with T on ``seq_dim``."""
        T = x.shape[seq_dim]
        shape = [1] * x.dim()
        shape[seq_dim] = T
        shape[-1] = x.shape[-1]
        cos = self.cos[:T].view(shape).to(x.dtype)
        sin = self.sin[:T].view(shape).to(x.dtype)
        return x * cos + self._rotate_half(x) * sin


def masked_softmax_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                             key_pad: Optional[torch.Tensor], scale: float,
                             return_probs: bool = False):
    """q: [B,H,T,D], k/v: [B,H,S,D], key_pad: [B,S] True = pad. Returns [B,H,T,D] (+ probs)."""
    logits = torch.einsum("bhtd,bhsd->bhts", q, k) * scale
    if key_pad is not None:
        logits = logits.masked_fill(key_pad[:, None, None, :], float("-inf"))
    probs = torch.softmax(logits.float(), dim=-1).type_as(q)
    out = torch.einsum("bhts,bhsd->bhtd", probs, v)
    return (out, probs) if return_probs else out


class DenseAttention(nn.Module):
    """Standard multi-head self-attention (the unit every attention variant is measured against)."""

    def __init__(self, d_model: int, n_heads: int, head_dim: int, rope: Rotary, bias: bool = False):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, head_dim
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, n_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, n_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=bias)
        self.rope = rope
        self.last_attn = None

    def forward(self, x: torch.Tensor, key_pad: Optional[torch.Tensor] = None, state=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q), self.rope(k)
        out = masked_softmax_attention(q, k, v, key_pad, self.head_dim ** -0.5)
        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out), x.new_zeros(())


class DenseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation: str = "gelu"):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = ACTIVATIONS[activation]

    def forward(self, x: torch.Tensor, key_pad: Optional[torch.Tensor] = None, state=None):
        return self.fc2(self.act(self.fc1(x))), x.new_zeros(())


ACTIVATIONS = {"gelu": F.gelu, "relu": F.relu, "gelu_tanh": lambda x: F.gelu(x, approximate="tanh")}
