"""MoM routers and auxiliary loss.

Port of Mixture-of-Modules@2ed1d28 ``mom/utils.py``: TokenRouter ('naive'), GRUTokenRouter ('gru'),
GRUdevTokenRouter ('dev'), STE, load_balancing_loss_func. Router type strings work exactly as
upstream: substring tests for 'naive' / 'dev' / 'gru', plus 'softmax' (temperature-scaled softmax
output) and 'random' (perturbation gate).

Facts from the code worth knowing:
* Output size: TokenRouter has num_ffn + 1 classes (the +1 is SKIP when ``enable_pause``);
  GRU routers have max(num_ffn, num_attn) + 1. The last class is SKIP: no module runs.
* The GRU state is per token (rows = batch*length), starts at zero at the beginning of every
  MoTBlock, and is carried across the assembly steps of that block. Attention and FFN have
  separate routers with separate states.
* GRU update: h' = n + z * (h - n) with n = tanh(W_in x + r * (W_hn h)). This is torch.nn.GRUCell's
  equation, implemented by hand.
* load_balancing_loss_func hard-codes top_k = 1 and applies softmax to the router *scores*. For
  'softmax' routers the scores are already probabilities, so the loss uses a double softmax
  (faithfully kept, logged as finding M5).
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs.clamp_(-1, 1)


class TokenRouter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        k = cfg.num_ffn + (1 if cfg.enable_pause else 0)
        self.hidden_dim = cfg.n_embd
        self.router_dim = cfg.router_dim
        self.enable_perturb = "random" in cfg.router_type
        self.softmax_output = "softmax" in cfg.router_type
        self.router = nn.Sequential(nn.Linear(self.hidden_dim, self.router_dim), nn.ReLU(), nn.Linear(self.router_dim, k))

    def forward(self, x):
        routing_score = self.router(x)
        if self.softmax_output:
            routing_score = F.softmax(routing_score / self.config.router_softmax_temperature, dim=-1)
        return routing_score


class GRUTokenRouter(nn.Module):
    dev = False

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        k = (cfg.num_ffn if self.dev else max(cfg.num_ffn, cfg.num_attn)) + (1 if cfg.enable_pause else 0)
        self.hidden_dim = cfg.n_embd
        self.router_dim = cfg.router_dim
        self.router = nn.Linear(self.router_dim, k)
        self.enable_perturb = "random" in cfg.router_type
        self.softmax_output = "softmax" in cfg.router_type
        n_gates = 4 if self.enable_perturb else 3
        self.h2h = nn.Linear(self.router_dim, n_gates * self.router_dim)
        self.x2h = nn.Linear(self.hidden_dim, n_gates * self.router_dim)

    def forward(self, x, h):
        bl, _ = x.shape
        if h is None:
            h = torch.zeros((bl, self.router_dim), dtype=x.dtype, device=x.device)
        gate_x = self.x2h(x)
        gate_h = self.h2h(h)
        if self.enable_perturb:
            i_r, i_i, i_n, i_p = gate_x.chunk(4, -1)
            h_r, h_i, h_n, h_p = gate_h.chunk(4, -1)
        else:
            i_r, i_i, i_n = gate_x.chunk(3, -1)
            h_r, h_i, h_n = gate_h.chunk(3, -1)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n) if self.dev else torch.tanh(i_n + (resetgate * h_n))
        h = newgate + inputgate * (h - newgate)
        if self.enable_perturb:
            perturbgate = torch.sigmoid(i_p + h_p)
            noise = torch.rand(h.shape, dtype=h.dtype, device=h.device)
            h = perturbgate * (h - noise) + noise
        routing_score = self.router(h)
        if self.softmax_output:
            routing_score = F.softmax(routing_score / self.config.router_softmax_temperature, dim=-1)
        elif self.dev and "sigmoid" in self.config.router_type:
            # upstream calls F.sigmoid(x, dim=-1), which raises TypeError; we apply sigmoid (finding M7)
            routing_score = torch.sigmoid(routing_score)
        return routing_score, h


class GRUdevTokenRouter(GRUTokenRouter):
    dev = True


def get_router(cfg):
    if "naive" in cfg.router_type:
        return TokenRouter(cfg)
    if "dev" in cfg.router_type:
        return GRUdevTokenRouter(cfg)
    if "gru" in cfg.router_type:
        return GRUTokenRouter(cfg)
    raise NotImplementedError(cfg.router_type)


def load_balancing_loss_func(gate_logits, num_modules=None, attention_mask=None):
    """Exact port (Switch/Mixtral style, top_k fixed to 1)."""
    top_k = 1
    if gate_logits is None:
        return 0
    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([g.to(compute_device) for g in gate_logits], dim=0)
    else:
        compute_device = gate_logits.device
        concatenated_gate_logits = gate_logits
    routing_weights = F.softmax(concatenated_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = F.one_hot(selected_experts, num_modules)
    if attention_mask is None:
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)
        expert_attention_mask = (attention_mask[None, :, :, None, None]
                                 .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_modules))
                                 .reshape(-1, top_k, num_modules).to(compute_device))
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(expert_attention_mask, dim=0)
        router_per_expert_attention_mask = (attention_mask[None, :, :, None]
                                            .expand((num_hidden_layers, batch_size, sequence_length, num_modules))
                                            .reshape(-1, num_modules).to(compute_device))
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0)
    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_modules
