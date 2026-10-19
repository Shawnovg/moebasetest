"""SUT encoder and decoder blocks.

Port of SUT@192cbd3 ``sut_layer/sut_transformer_layer.py``
(``TransformerEncoderLayerBase`` and ``TransformerDecoderLayerBase``) without fairseq.

Kept exactly:
* post-LN by default (``normalize_before=False`` in ``base_ende_architecture``);
* encoder FFN-expert activation = ``gelu`` followed by the *main* dropout (upstream passes
  ``self.dropout_module``), decoder FFN-expert activation = ``gelu`` then *activation* dropout;
* FFN experts have biases (``bias=True``), attention experts do not;
* both the attention gate and the FFN gate receive ``halt_mask`` (p_never_halt) as skip mask;
* boolean encoder attention masks become -1e8 additive masks.

Removed: incremental decoding, ``cross_self_attention``, ``scale_attn``/``scale_heads``/``scale_fc``/
``scale_resids`` options (all off in SUT configurations), activation checkpointing/FSDP wrappers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MoMHA
from .moe import MoE


@dataclass
class SUTBlockConfig:
    embed_dim: int = 128
    # attention experts
    attn_num_expert: int = 4
    attn_k: int = 2
    attn_expert_dim: int = 64  # = heads_per_expert * head_dim
    head_dim: int = 32
    # FFN experts
    ff_num_expert: int = 4
    ff_k: int = 2
    ff_expert_dim: int = 256
    # regularisation
    dropout: float = 0.1
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    gating_dropout: float = 0.0
    # routing objectives (coefficients)
    cvloss: float = 0.0
    switchloss: float = 0.0
    zloss: float = 0.0
    miloss: float = 0.0
    mim_variant: str = "code"
    sample_topk: int = 0
    normalize_before: bool = False
    max_rel_positions: int = 64
    activation_fn: str = "gelu"


def _act(name):
    return {"gelu": F.gelu, "relu": F.relu}[name]


class _Wrap(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class SUTEncoderLayer(nn.Module):
    def __init__(self, cfg: SUTBlockConfig):
        super().__init__()
        self.embed_dim = cfg.embed_dim
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout_module = nn.Dropout(cfg.dropout)
        self.normalize_before = cfg.normalize_before
        self.self_attn = MoMHA(
            cfg.embed_dim, dropout=cfg.attention_dropout, self_attention=True,
            num_expert=cfg.attn_num_expert, top_k=cfg.attn_k, expert_dim=cfg.attn_expert_dim,
            head_dim=cfg.head_dim, cvloss=cfg.cvloss, switchloss=cfg.switchloss, zloss=cfg.zloss,
            miloss=cfg.miloss, sample_topk=cfg.sample_topk, gating_dropout=cfg.gating_dropout,
            max_positions=cfg.max_rel_positions, mim_variant=cfg.mim_variant)
        self.moe = MoE(
            self.embed_dim, cfg.ff_expert_dim, cfg.ff_num_expert, cfg.ff_k,
            cvloss=cfg.cvloss, switchloss=cfg.switchloss, zloss=cfg.zloss, miloss=cfg.miloss,
            activation=nn.Sequential(_Wrap(_act(cfg.activation_fn)), self.dropout_module),
            noisy_gating=False, bias=True, acc_aux_loss=True, gating_dropout=cfg.gating_dropout,
            mim_variant=cfg.mim_variant)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x, self_attn_input, halt_mask, layer_idx,
                encoder_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        """x, self_attn_input: T x B x C; halt_mask: T x B float; encoder_padding_mask: B x T bool."""
        if self_attn_input is None:
            self_attn_input = x
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8 if x.dtype == torch.float32 else -1e4)

        residual = x
        if self.normalize_before:
            x_ = self.self_attn_layer_norm(x)
            self_attn_input = x_ if x is self_attn_input else self.self_attn_layer_norm(self_attn_input)
            x = x_
        x, _, self_aux_loss = self.self_attn(
            query=x, key=self_attn_input, value=self_attn_input,
            key_padding_mask=encoder_padding_mask, attn_mask=attn_mask, skip_mask=halt_mask)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x, mlp_aux_loss = self.moe(x, skip_mask=halt_mask)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, None, self_aux_loss + mlp_aux_loss


class SUTDecoderLayer(nn.Module):
    def __init__(self, cfg: SUTBlockConfig, no_encoder_attn: bool = False):
        super().__init__()
        self.embed_dim = cfg.embed_dim
        self.dropout_module = nn.Dropout(cfg.dropout)
        self.activation_dropout_module = nn.Dropout(cfg.activation_dropout)
        common = dict(num_expert=cfg.attn_num_expert, top_k=cfg.attn_k, expert_dim=cfg.attn_expert_dim,
                      head_dim=cfg.head_dim, cvloss=cfg.cvloss, switchloss=cfg.switchloss, zloss=cfg.zloss,
                      miloss=cfg.miloss, sample_topk=cfg.sample_topk, gating_dropout=cfg.gating_dropout,
                      mim_variant=cfg.mim_variant)
        self.self_attn = MoMHA(cfg.embed_dim, dropout=cfg.attention_dropout, self_attention=True,
                               max_positions=cfg.max_rel_positions, **common)
        self.normalize_before = cfg.normalize_before
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MoMHA(cfg.embed_dim, kdim=cfg.embed_dim, vdim=cfg.embed_dim,
                                      dropout=cfg.attention_dropout, encoder_decoder_attention=True, **common)
            self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.moe = MoE(
            self.embed_dim, cfg.ff_expert_dim, cfg.ff_num_expert, cfg.ff_k,
            cvloss=cfg.cvloss, switchloss=cfg.switchloss, zloss=cfg.zloss, miloss=cfg.miloss,
            activation=nn.Sequential(_Wrap(_act(cfg.activation_fn)), self.activation_dropout_module),
            noisy_gating=False, acc_aux_loss=True, gating_dropout=cfg.gating_dropout, bias=True,
            mim_variant=cfg.mim_variant)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x, self_attn_input, halt_mask, layer_idx, encoder_out=None, encoder_padding_mask=None,
                self_attn_mask=None, self_attn_padding_mask=None):
        if self_attn_input is None:
            self_attn_input = x
        residual = x
        if self.normalize_before:
            x_ = self.self_attn_layer_norm(x)
            self_attn_input = x_ if self_attn_input is x else self.self_attn_layer_norm(self_attn_input)
            x = x_
        y = self_attn_input
        x, attn, self_aux_loss = self.self_attn(
            query=x, key=y, value=y, key_padding_mask=self_attn_padding_mask,
            attn_mask=self_attn_mask, skip_mask=halt_mask)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        cross_aux_loss = 0.0
        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            x, attn, cross_aux_loss = self.encoder_attn(
                query=x, key=encoder_out, value=encoder_out,
                key_padding_mask=encoder_padding_mask, skip_mask=halt_mask)
            x = self.dropout_module(x)
            x = residual + x
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x, mlp_aux_loss = self.moe(x, skip_mask=halt_mask)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, attn, None, self_aux_loss + cross_aux_loss + mlp_aux_loss
