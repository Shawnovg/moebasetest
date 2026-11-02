"""Experiment-wide routing controls and the balancing objectives of each reference paper.

Every routed module owns a pointer to the model's ``RoutingState`` and asks it three things:

* ``state.balance``       'none' | 'lb' | 'warmup'. Under 'lb' the module adds ITS paper's balancing
                          loss (weights in the structure's config); otherwise it adds nothing.
* ``state.random_active`` True during the warm-up phase of mode 'warmup': the module replaces the
                          router's choice of units with a uniformly random choice, but takes the gate
                          values from the router's probabilities at those random units, so the router
                          keeps receiving gradient from the task loss.
* ``state.token_ids``     B x T token ids (hash-routing variants) and ``state.valid`` B x T bool.

Balancing objectives (each an exact port of the released code it names):

* ``switch_loss``  SUT / MoA ``compute_switchloss``: E * sum_i norm(sum_t p_ti) * norm(count_i)
* ``z_loss``       SUT / MoA ``compute_zloss``: mean_t (log sum_i exp(logit_ti))^2
* ``moh_loss``     MoH ``MoHAttention``: mean_i(me_i * ce_i) * E^2, me = mean router prob,
                   ce = fraction of tokens selecting i
* ``mom_loss``     MoM ``load_balancing_loss_func`` (Switch/Mixtral form, top_k hard-coded to 1,
                   softmax applied to the router's raw scores)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class RoutingState:
    balance: str = "none"          # none | lb | warmup
    random_active: bool = False
    token_ids: Optional[torch.Tensor] = None
    valid: Optional[torch.Tensor] = None
    record: bool = False           # store per-token routing decisions for analysis
    generator: Optional[torch.Generator] = field(default=None, repr=False)

    @property
    def lb(self) -> bool:
        return self.balance == "lb"


def random_topk(n_rows: int, n_units: int, k: int, device, generator=None) -> torch.Tensor:
    """k distinct uniformly random unit indices per row (warm-up routing)."""
    r = torch.rand(n_rows, n_units, device=device, generator=generator)
    return r.topk(k, dim=1).indices


def switch_loss(probs: torch.Tensor, expert_size: torch.Tensor, num_experts: int) -> torch.Tensor:
    loss = F.normalize(probs.sum(0), p=1, dim=0) * F.normalize(expert_size.float(), p=1, dim=0)
    return loss.sum() * num_experts


def z_loss(logits: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.log(torch.exp(logits).sum(dim=1)) ** 2)


def moh_loss(gates: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    num_experts = gates.shape[1]
    me = gates.mean(dim=0)
    ce = mask.float().mean(dim=0)
    return torch.mean(me * ce) * num_experts * num_experts


def mom_loss(scores: torch.Tensor, num_modules: int) -> torch.Tensor:
    """scores: N x num_modules raw router outputs of valid tokens (one router, one step)."""
    routing_weights = F.softmax(scores, dim=-1)
    _, selected = torch.topk(routing_weights, 1, dim=-1)
    expert_mask = F.one_hot(selected, num_modules)                  # N x 1 x M
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)       # 1 x M
    router_prob_per_expert = torch.mean(routing_weights, dim=0)      # M
    return torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0)) * num_modules
