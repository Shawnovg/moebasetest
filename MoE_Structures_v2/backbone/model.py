"""The shared 2-layer pre-LN encoder that every structure plugs into.

    tokens -> embedding -> [pre-LN block] x n_layers -> final LayerNorm -> output head (vocab)

A structure package (``models/<name>/__init__.py``) provides ONE of:

* ``build_sublayers(cfg, rope, state, layer_idx) -> (attention_module, ffn_module)``
  used by Dense, Dense-wide, MoE, MoH, MoA, MoMHA and MiTA: the block stays
  ``x + attn(LN(x))`` then ``x + ffn(LN(x))``, only the two sub-modules change;
* ``build_body(cfg, rope, state) -> nn.Module`` used by MoM, which replaces the whole layer stack.

Every sub-module is called as ``module(x, key_pad, state)`` and returns ``(output, aux_loss)``.
``step_hook(i, h) -> h`` (None in training) lets analysis code read or patch the residual stream
after the embeddings (i = -1) and after every layer / assembly step.
"""
from __future__ import annotations

import importlib
from typing import Dict, Optional

import torch
import torch.nn as nn

from .layers import Rotary
from .routing import RoutingState


class Block(nn.Module):
    def __init__(self, d_model: int, attn: nn.Module, ffn: nn.Module, eps: float = 1e-5):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=eps)
        self.attn = attn
        self.ln2 = nn.LayerNorm(d_model, eps=eps)
        self.ffn = ffn

    def forward(self, x, key_pad, state):
        a, aux1 = self.attn(self.ln1(x), key_pad, state)
        x = x + a
        f, aux2 = self.ffn(self.ln2(x), key_pad, state)
        return x + f, aux1 + aux2


def structure_package(name: str):
    return importlib.import_module(f"models.{name}")


class EncoderModel(nn.Module):
    def __init__(self, cfg: Dict, vocab_size: int):
        super().__init__()
        bb = cfg["backbone"]
        self.cfg = cfg
        d = bb["d_model"]
        self.embed = nn.Embedding(vocab_size, d)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.rope = Rotary(bb["head_dim"], base=bb.get("rope_base", 10000.0), max_len=bb.get("max_len", 512))
        self.state = RoutingState(balance=cfg["routing"]["balance"])
        pkg = structure_package(cfg["structure"])
        if hasattr(pkg, "build_body"):
            self.body = pkg.build_body(cfg, self.rope, self.state)
            self.blocks = None
        else:
            self.body = None
            self.blocks = nn.ModuleList()
            for i in range(bb["n_layers"]):
                attn, ffn = pkg.build_sublayers(cfg, self.rope, self.state, i)
                self.blocks.append(Block(d, attn, ffn, eps=bb.get("ln_eps", 1e-5)))
        self.final_ln = nn.LayerNorm(d, eps=bb.get("ln_eps", 1e-5))
        self.head = nn.Linear(d, vocab_size)
        self.step_hook = None

    def forward(self, tokens: torch.Tensor, pad_mask: Optional[torch.Tensor] = None,
                select: Optional[torch.Tensor] = None):
        """tokens, pad_mask: B x T (pad_mask True = padding). Returns logits B x T x V, aux loss.
        With ``select`` (B x T bool, the answer slots) only those positions are decoded: logits N_sel x V."""
        self.state.token_ids = tokens
        self.state.valid = None if pad_mask is None else ~pad_mask
        x = self.embed(tokens)
        if self.step_hook is not None:
            x = self.step_hook(-1, x)
        aux = x.new_zeros(())
        if self.body is not None:
            x, aux = self.body(x, pad_mask, self.state, self.step_hook)
        else:
            for i, blk in enumerate(self.blocks):
                x, a = blk(x, pad_mask, self.state)
                aux = aux + a
                if self.step_hook is not None:
                    x = self.step_hook(i, x)
        if select is not None:
            x = x[select]
        return self.head(self.final_ln(x)), aux

    def routed_modules(self):
        """(name, module) for every module that routes (has a ``route_record`` attribute)."""
        return [(n, m) for n, m in self.named_modules() if hasattr(m, "route_record")]


def count_parameters(model: nn.Module, exclude_embeddings: bool = True) -> int:
    n = 0
    for name, p in model.named_parameters():
        if exclude_embeddings and (name.startswith("embed.") or name.startswith("head.")):
            continue
        n += p.numel()
    return n
