"""SUT encoder, decoder and the two task wrappers.

* ``SUTSeq2Seq``     - reference path: encoder-decoder as in SUT@192cbd3 transformer_encoder.py /
                       transformer_decoder.py / transformer_base.py (shared embeddings, tied output).
* ``SUTClassifier``  - controlled path (Level 3 adaptation): the same SUT encoder with a 7-way
                       readout at the <cls> position. With ``share_layers=False`` it builds untied
                       blocks, which gives the Dense / Switch-style / MoA-style families from the
                       same audited units (see configs/logic/*.yaml).

Embedding conventions copied from fairseq: token embedding ~ N(0, d^-1/2) with zero pad row,
scaled by sqrt(d); sinusoidal positions with fairseq's padding-aware position ids; embedding
dropout; pad positions zeroed before the encoder.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .halting import ACTWrapper
from .layers import SUTBlockConfig, SUTDecoderLayer, SUTEncoderLayer
from .moe import MoE


# ------------------------------------------------------------------------------------------ utils
def make_positions(tokens: torch.Tensor, padding_idx: int) -> torch.Tensor:
    mask = tokens.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


class SinusoidalPositionalEmbedding(nn.Module):
    """fairseq.modules.SinusoidalPositionalEmbedding (non-incremental path)."""

    def __init__(self, embedding_dim: int, padding_idx: int, init_size: int = 1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.register_buffer("weights", self.get_embedding(init_size, embedding_dim, padding_idx), persistent=False)

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, tokens):
        bsz, seq_len = tokens.shape
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.size(0):
            self.weights = self.get_embedding(max_pos, self.embedding_dim, self.padding_idx).to(self.weights)
        positions = make_positions(tokens, self.padding_idx)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()


def build_embedding(num_embeddings, embed_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embed_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embed_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def moe_modules(model: nn.Module) -> Dict[str, MoE]:
    out, seen = {}, set()
    for n, m in model.named_modules():
        if isinstance(m, MoE) and id(m) not in seen:
            seen.add(id(m))
            out[n] = m
    return out


def collect_moe_aux_loss(model: nn.Module):
    """Equivalent of the loop in sut_loss.py: sum of get_aux_loss_and_clear() over MoE modules."""
    total = 0.0
    for m in moe_modules(model).values():
        total = total + m.get_aux_loss_and_clear()
    return total


def clear_moe_statistics(model: nn.Module):
    for m in moe_modules(model).values():
        m.clear_aux_statistics()


# ------------------------------------------------------------------------------------------ config
@dataclass
class SUTConfig:
    vocab_size: int = 32
    padding_idx: int = 0
    block: SUTBlockConfig = field(default_factory=SUTBlockConfig)
    encoder_layers: int = 6  # number of block applications (steps)
    decoder_layers: int = 6
    share_layers: bool = True  # True = universal (one block applied repeatedly)
    halting: bool = True
    act_threshold: float = 0.999
    halting_dropout: float = 0.0
    actloss: float = 0.0
    no_scale_embedding: bool = False
    num_classes: int = 7  # classifier interface only

    @staticmethod
    def from_dict(d: dict) -> "SUTConfig":
        d = dict(d)
        block = SUTBlockConfig(**d.pop("block", {}))
        return SUTConfig(block=block, **d)

    def to_dict(self):
        return asdict(self)


# ------------------------------------------------------------------------------------------ encoder
class SUTEncoder(nn.Module):
    def __init__(self, cfg: SUTConfig, embed_tokens: nn.Embedding):
        super().__init__()
        b = cfg.block
        self.cfg = cfg
        self.padding_idx = cfg.padding_idx
        self.embed_tokens = embed_tokens
        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(b.embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(b.embed_dim, cfg.padding_idx)
        self.dropout_module = nn.Dropout(b.dropout)
        self.num_layers = cfg.encoder_layers
        if cfg.share_layers:
            self.universal_layer = SUTEncoderLayer(b)
            self.layers = None
        else:
            assert not cfg.halting, "halting is only defined for the shared (universal) block"
            self.layers = nn.ModuleList([SUTEncoderLayer(b) for _ in range(cfg.encoder_layers)])
        self.halting = cfg.halting
        if self.halting:
            self.halting_universal_layer = ACTWrapper(self.universal_layer, b.embed_dim,
                                                      threshold=cfg.act_threshold,
                                                      halting_dropout=cfg.halting_dropout)
        self.actloss = cfg.actloss
        self.layer_norm = nn.LayerNorm(b.embed_dim) if b.normalize_before else None
        # analysis hooks (None in normal use):
        #   step_hook(i, st) -> st, st = {"x", "soft", "act", "p_never_halt"} (T x B layout); i = -1 for the
        #   embeddings, 0.. for block steps. "act" is the ACT state (step, log_never_halt, acc_h, acc_depth)
        #   so a hook can patch the full halting state, not only the hidden vectors.
        #   run_steps: run only the first k block applications (depth ablation)
        self.step_hook = None
        self.run_steps = None

    def layer(self, i):
        return self.universal_layer if self.layers is None else self.layers[i]

    def forward(self, src_tokens: torch.Tensor) -> dict:
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = bool(encoder_padding_mask.any())
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        x = embed + self.embed_positions(src_tokens)
        x = self.dropout_module(x)
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))
        x = x.transpose(0, 1)

        soft_halt_x = None
        halt_mask = (1.0 - encoder_padding_mask.t().float()).contiguous()
        act_state = None
        act_loss = 0.0
        active_per_step: List[torch.Tensor] = []
        if self.step_hook is not None:
            x = self.step_hook(-1, {"x": x, "soft": None, "act": None, "p_never_halt": None})["x"]
        n_steps = self.num_layers if self.run_steps is None else min(self.run_steps, self.num_layers)
        for i in range(n_steps):
            pad = encoder_padding_mask if has_pads else None
            p_never_halt = None
            if self.halting:
                act_state, lr, soft_halt_x, act_loss, p_never_halt = self.halting_universal_layer(
                    act_state, x, soft_halt_x, halt_mask, i, encoder_padding_mask=pad)
                active_per_step.append((p_never_halt > 0).float().sum().detach())
            else:
                lr = self.layer(i)(x, soft_halt_x, halt_mask, i, encoder_padding_mask=pad)
                active_per_step.append(halt_mask.sum().detach())
            x = lr[0]
            if self.step_hook is not None:
                st = self.step_hook(i, {"x": x, "soft": soft_halt_x, "act": act_state, "p_never_halt": p_never_halt})
                x, soft_halt_x, act_state = st["x"], st["soft"], st["act"]
        total_encoder_loss = 0.0
        depth_tokens = None
        if self.halting:
            x = soft_halt_x
            total_encoder_loss = self.actloss * act_loss
            if act_state is not None and act_state[0] > 0:
                i_last, log_never_halt, _, acc_expect_depth = act_state
                p_last = log_never_halt.exp()
                p_last = p_last.masked_fill(p_last < (1 - self.halting_universal_layer.threshold), 0)
                # per-token expected halting index (same quantity as the ACT loss), T x B
                depth_tokens = (acc_expect_depth + p_last * i_last).detach()
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": encoder_padding_mask,  # B x T
            "encoder_loss": total_encoder_loss,
            "encoder_expected_halt": act_loss,
            "encoder_active_tokens": torch.stack(active_per_step) if active_per_step else None,
            "expected_depth_tokens": depth_tokens,
        }


# ------------------------------------------------------------------------------------------ decoder
class SUTDecoder(nn.Module):
    def __init__(self, cfg: SUTConfig, embed_tokens: nn.Embedding):
        super().__init__()
        b = cfg.block
        self.cfg = cfg
        self.padding_idx = cfg.padding_idx
        self.embed_tokens = embed_tokens
        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(b.embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(b.embed_dim, cfg.padding_idx)
        self.dropout_module = nn.Dropout(b.dropout)
        self.num_layers = cfg.decoder_layers
        assert cfg.share_layers, "the reference decoder is universal"
        self.universal_layer = SUTDecoderLayer(b)
        self.halting = cfg.halting
        if self.halting:
            self.halting_universal_layer = ACTWrapper(self.universal_layer, b.embed_dim,
                                                      threshold=cfg.act_threshold,
                                                      halting_dropout=cfg.halting_dropout)
        self.actloss = cfg.actloss
        self.layer_norm = nn.LayerNorm(b.embed_dim) if b.normalize_before else None
        # share_all_embeddings: output projection tied to the (shared) embedding
        self.output_projection = nn.Linear(embed_tokens.weight.shape[1], embed_tokens.weight.shape[0], bias=False)
        self.output_projection.weight = embed_tokens.weight

    @staticmethod
    def future_mask(t: torch.Tensor) -> torch.Tensor:
        dim = t.size(0)
        return torch.triu(torch.full((dim, dim), float("-inf"), device=t.device, dtype=t.dtype), 1)

    def forward(self, prev_output_tokens, encoder_out: dict):
        enc = encoder_out["encoder_out"]
        padding_mask = encoder_out["encoder_padding_mask"]
        positions = self.embed_positions(prev_output_tokens)
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        x = x + positions
        x = self.dropout_module(x)
        x = x.transpose(0, 1)
        self_attn_padding_mask = None
        if prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)
        halt_mask = (1.0 - prev_output_tokens.eq(self.padding_idx).t().float()).contiguous()
        soft_halt_x, act_state, act_loss = None, None, 0.0
        for idx in range(self.num_layers):
            self_attn_mask = self.future_mask(x)
            kw = dict(encoder_out=enc, encoder_padding_mask=padding_mask,
                      self_attn_mask=self_attn_mask, self_attn_padding_mask=self_attn_padding_mask)
            if self.halting:
                act_state, lr, soft_halt_x, act_loss, _ = self.halting_universal_layer(
                    act_state, x, soft_halt_x, halt_mask, idx, **kw)
            else:
                lr = self.universal_layer(x, soft_halt_x, halt_mask, idx, **kw)
            x = lr[0]
        total_decoder_loss = 0.0
        if self.halting:
            x = soft_halt_x
            total_decoder_loss = self.actloss * act_loss
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        x = x.transpose(0, 1)
        logits = self.output_projection(x)
        extra = {
            "total_aux_loss": encoder_out["encoder_loss"] + total_decoder_loss,
            "encoder_expected_halt": encoder_out["encoder_expected_halt"],
            "decoder_expected_halt": act_loss,
        }
        return logits, extra


# ------------------------------------------------------------------------------------------ wrappers
class SUTSeq2Seq(nn.Module):
    """Reference interface (encoder-decoder, shared embeddings, tied output)."""

    def __init__(self, cfg: SUTConfig):
        super().__init__()
        self.cfg = cfg
        emb = build_embedding(cfg.vocab_size, cfg.block.embed_dim, cfg.padding_idx)
        self.encoder = SUTEncoder(cfg, emb)
        self.decoder = SUTDecoder(cfg, emb)

    def forward(self, src_tokens, prev_output_tokens):
        enc = self.encoder(src_tokens)
        logits, extra = self.decoder(prev_output_tokens, enc)
        extra["encoder_active_tokens"] = enc["encoder_active_tokens"]
        extra["expected_depth_tokens"] = enc["expected_depth_tokens"]
        return logits, extra


class SUTClassifier(nn.Module):
    """Controlled interface: SUT encoder + linear readout at position 0 (<cls>)."""

    def __init__(self, cfg: SUTConfig):
        super().__init__()
        self.cfg = cfg
        emb = build_embedding(cfg.vocab_size, cfg.block.embed_dim, cfg.padding_idx)
        self.encoder = SUTEncoder(cfg, emb)
        self.head = nn.Linear(cfg.block.embed_dim, cfg.num_classes)

    def forward(self, tokens):
        enc = self.encoder(tokens)
        h = enc["encoder_out"][0]  # T x B x C -> <cls> state, B x C
        logits = self.head(h)
        extra = {
            "total_aux_loss": enc["encoder_loss"],
            "encoder_expected_halt": enc["encoder_expected_halt"],
            "encoder_active_tokens": enc["encoder_active_tokens"],
            "expected_depth_tokens": enc["expected_depth_tokens"],
        }
        return logits, extra


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    """Exact port of sut_loss.label_smoothed_nll_loss (sum reduction)."""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss
