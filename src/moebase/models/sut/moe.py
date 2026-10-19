"""Sparse mixture-of-experts unit used by SUT for both FFN experts and attention experts.

Port of SUT@192cbd3 ``sut_layer/parallel_linear/parallel_experts/{moe.py,parallel_experts.py}``.
Line-for-line equivalent except for the items listed in docs/03_deviation_log.md:

* D-SUT-1  ParallelLinear's hand-written autograd is replaced by the same split/mm loop under
           ordinary autograd (identical forward and gradients in FP32; upstream additionally casts
           to FP16 under autocast via ``custom_fwd(cast_inputs=torch.float16)``).
* D-SUT-2  ``mim_variant='code'`` (default, faithful) reproduces the released sign, which equals
           H(E|X) + H(E). ``mim_variant='paper'`` gives -MI = H(E|X) - H(E) as written in the
           paper's prose. See docs/02_source_audit.md section S3.
* D-SUT-3  ``clear_aux_statistics()`` added so evaluation passes do not accumulate statistics
           (upstream only clears inside the training criterion, so statistics grow during validation).
* Diagnostics: the last ``top_k_indices`` and ``expert_size`` are stored for routing analysis.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_gating(k: int, probs: torch.Tensor, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor):
    zeros = torch.zeros_like(probs)
    gates = zeros.scatter(1, top_k_indices, top_k_gates)
    top_k_gates = top_k_gates.flatten()
    top_k_experts = top_k_indices.flatten()
    nonzeros = top_k_gates.nonzero().squeeze(-1)
    top_k_experts_nonzero = top_k_experts[nonzeros]
    _, _index_sorted_experts = top_k_experts_nonzero.sort(0)
    expert_size = (gates > 0).long().sum(0)
    index_sorted_experts = nonzeros[_index_sorted_experts]
    batch_index = index_sorted_experts.div(k, rounding_mode="trunc")
    batch_gates = top_k_gates[index_sorted_experts]
    return batch_gates, batch_index, expert_size, gates, index_sorted_experts


class ParallelExperts(nn.Module):
    """E independent linear maps applied to expert-sorted rows."""

    def __init__(self, num_experts: int, input_size: int, output_size: int, bias: bool = False):
        super().__init__()
        self.w = nn.Parameter(torch.empty(num_experts, input_size, output_size))
        self.b = nn.Parameter(torch.zeros(num_experts, output_size)) if bias else None
        self.reset_parameters()

    def extra_repr(self):
        return "num_experts={}, input_size={}, output_size={}".format(*self.w.shape)

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.w, -1.0 / self.w.size(1), 1.0 / self.w.size(1))
        if self.b is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.b, -bound, bound)

    def forward(self, inputs: torch.Tensor, expert_size: torch.Tensor) -> torch.Tensor:
        sizes: List[int] = expert_size.tolist()
        chunks = inputs.split(sizes, dim=0)
        outs = []
        for i, chunk in enumerate(chunks):
            o = chunk @ self.w[i]
            if self.b is not None:
                o = o + self.b[i]
            outs.append(o)
        return torch.cat(outs, dim=0)


class MoE(nn.Module):
    """Sparsely gated mixture of experts (upstream class ``MoE``)."""

    def __init__(self, input_size, head_size, num_experts, k,
                 cvloss=0, switchloss=0, zloss=0, miloss=0,
                 bias=False, gating_dropout=0, activation=None, noisy_gating=False,
                 acc_aux_loss=False, mlp_router=True, hidden_size=None,
                 mim_variant: str = "code"):
        super().__init__()
        assert mim_variant in ("code", "paper")
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.input_size = input_size
        self.head_size = head_size
        self.bias = bias
        self.experts = ParallelExperts(num_experts, input_size, head_size, bias)
        if hidden_size is None:
            hidden_size = head_size
        self.output_experts = ParallelExperts(num_experts, hidden_size, input_size, bias)
        self.k = min(k, self.num_experts)
        self.cvloss, self.switchloss, self.zloss, self.miloss = cvloss, switchloss, zloss, miloss
        self.mim_variant = mim_variant
        self.activation = activation
        self.acc_aux_loss = acc_aux_loss
        if self.acc_aux_loss:
            self.init_aux_statistics()

        if mlp_router:
            self.w_gate = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.Tanh(),
                nn.Dropout(gating_dropout),
                nn.Linear(input_size, 2 * num_experts if noisy_gating else num_experts, bias=False),
            )
            out_weight = self.w_gate[-1].weight
        else:
            self.w_gate = nn.Linear(input_size, 2 * num_experts if noisy_gating else num_experts, bias=False)
            out_weight = self.w_gate.weight
        if self.noisy_gating:
            nn.init.zeros_(out_weight)
        # diagnostics (not used by the computation)
        self.last_top_k_indices: Optional[torch.Tensor] = None
        self.last_expert_size: Optional[torch.Tensor] = None
        self.usage_counts: Optional[torch.Tensor] = None  # summed expert_size over calls since reset

    def reset_usage(self):
        self.usage_counts = None

    def extra_repr(self):
        return "k={}, cvloss={}, switchloss={}, zloss={}, miloss={}({}), noisy_gating={}".format(
            self.k, self.cvloss, self.switchloss, self.zloss, self.miloss, self.mim_variant, self.noisy_gating)

    # ---------------------------------------------------------------- auxiliary objectives
    @staticmethod
    def cv_squared(x):
        eps = 1e-10
        if x.shape[0] == 1:
            return 0
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def init_aux_statistics(self):
        self.acc_probs = 0.0
        self.acc_gates = 0.0
        self.acc_freq = 0.0
        self.acc_lsesq = 0.0
        self.acc_count = 0
        self.probs = []
        self.log_probs = []
        self.masks = []

    clear_aux_statistics = init_aux_statistics

    def update_aux_statistics(self, logits, probs, gates, skip_mask=None):
        lsesq = torch.log(torch.exp(logits).sum(dim=-1)) ** 2
        log_probs = torch.log_softmax(logits, dim=-1)
        self.acc_count = self.acc_count + logits.size(0)
        self.acc_probs = self.acc_probs + probs.sum(0)
        self.acc_gates = self.acc_gates + gates.sum(0)
        self.acc_freq = self.acc_freq + (gates > 0).float().sum(0)
        self.acc_lsesq = self.acc_lsesq + lsesq.sum()
        self.probs.append(probs)
        self.log_probs.append(log_probs)
        self.masks.append(skip_mask)

    def _mi_terms(self, prob, log_prob, p_x):
        p_e = (p_x * prob).sum(0)
        H_e = (p_e * p_e.log()).sum()  # = -H(E)
        neg_H_e_given_x = (p_x * prob * log_prob).sum()  # = -H(E|X)
        if self.mim_variant == "code":
            return -(neg_H_e_given_x + H_e)  # upstream: = H(E|X) + H(E)
        return -neg_H_e_given_x + H_e  # paper: -MI = H(E|X) - H(E)

    def get_aux_loss_and_clear(self):
        if self.acc_count == 0:
            self.init_aux_statistics()
            return 0.0
        cvloss = self.cv_squared(F.normalize(self.acc_gates, p=1, dim=0))
        switchloss = self.num_experts * (
            F.normalize(self.acc_probs, p=1, dim=0) * F.normalize(self.acc_freq, p=1, dim=0)).sum()
        zloss = self.acc_lsesq / self.acc_count
        prob = torch.cat(self.probs, dim=0)
        log_prob = torch.cat(self.log_probs, dim=0)
        mask = torch.cat(self.masks, dim=0)
        p_x = mask / (mask.sum() + 1e-12)
        miloss = self._mi_terms(prob, log_prob, p_x)
        loss = (self.cvloss * cvloss + self.switchloss * switchloss
                + self.zloss * zloss + self.miloss * miloss)
        self.init_aux_statistics()
        return loss

    def compute_cvloss(self, probs):
        return self.cv_squared(F.normalize(probs.sum(0), p=1, dim=0))

    def compute_switchloss(self, probs, freqs):
        loss = F.normalize(probs.sum(0), p=1, dim=0) * F.normalize(freqs.float(), p=1, dim=0)
        return loss.sum() * self.num_experts

    def compute_zloss(self, logits):
        return torch.mean(torch.log(torch.exp(logits).sum(dim=1)) ** 2)

    def compute_miloss(self, logits, prob, mask):
        log_prob = torch.log_softmax(logits, dim=-1)
        p_x = 1 / logits.shape[0] if mask is None else mask / (mask.sum() + 1e-12)
        return self._mi_terms(prob, log_prob, p_x)

    # ---------------------------------------------------------------- gating
    def top_k_gating(self, x, skip_mask=None, sample_topk=0, noise_epsilon=1e-2, return_indices=False):
        clean_logits = self.w_gate(x)
        if self.noisy_gating:
            clean_logits, raw_noise_stddev = clean_logits.chunk(2, dim=-1)
            if self.training:
                noise_stddev = torch.sigmoid(raw_noise_stddev) + noise_epsilon
                logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            else:
                logits = clean_logits
        else:
            logits = clean_logits

        probs = torch.softmax(logits, dim=1)
        if skip_mask is not None:
            probs = torch.masked_fill(probs, (skip_mask == 0), 0)
            logits = torch.masked_fill(logits, (skip_mask == 0), 0)

        if self.training and (sample_topk > 0):
            assert sample_topk <= self.k
            _, top_km1_indices = probs.topk(self.k - sample_topk, dim=1)
            masked_probs = probs + 1e-6
            masked_probs[torch.arange(probs.size(0)).unsqueeze(1), top_km1_indices] = 0
            k_indices = torch.multinomial(masked_probs, sample_topk)
            top_k_indices = torch.cat([top_km1_indices, k_indices], dim=-1)
            top_k_gates = torch.gather(probs, 1, top_k_indices)
        else:
            top_k_gates, top_k_indices = probs.topk(self.k, dim=1)
        top_k_gates = top_k_gates / (top_k_gates.sum(dim=1, keepdim=True) + 1e-6)

        batch_gates, batch_index, expert_size, gates, index_sorted_experts = \
            compute_gating(self.k, probs, top_k_gates, top_k_indices)

        self.expert_size = expert_size
        self.index_sorted_experts = index_sorted_experts
        self.batch_index = batch_index
        self.batch_gates = batch_gates
        self.gates = gates.detach()
        self.last_top_k_indices = top_k_indices.detach()
        self.last_expert_size = expert_size.detach()
        uc = expert_size.detach().float()
        self.usage_counts = uc if self.usage_counts is None else self.usage_counts + uc

        loss = 0.0
        if self.acc_aux_loss:
            self.update_aux_statistics(logits, probs, gates, skip_mask)
        else:
            loss += self.cvloss * self.compute_cvloss(gates)
            loss += self.switchloss * self.compute_switchloss(probs, self.expert_size)
            loss += self.zloss * self.compute_zloss(logits)
            loss += self.miloss * self.compute_miloss(logits, probs, skip_mask)
        if return_indices:
            return loss, top_k_indices
        return loss

    # ---------------------------------------------------------------- FFN path
    def forward(self, x, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        bsz, length, emb_size = x.size()
        if skip_mask is not None:
            assert x.size()[:-1] == skip_mask.size(), "Skip mask should be same shape as `x`"
            skip_mask = skip_mask.flatten()[:, None]
        x = x.reshape(-1, emb_size)
        loss = self.top_k_gating(x, skip_mask, sample_topk=sample_topk)

        expert_inputs = x[self.batch_index]
        h = self.experts(expert_inputs, self.expert_size)
        h = self.activation(h)
        expert_outputs = self.output_experts(h, self.expert_size)
        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]
        zeros = torch.zeros((bsz * length, self.input_size), dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        return y.view(bsz, length, self.input_size), loss

    # ---------------------------------------------------------------- attention path (q / o projections)
    def map(self, x, skip_mask=None, sample_topk=0, return_indices=False):
        if skip_mask is not None:
            assert x.size()[:-1] == skip_mask.size(), "Skip mask should be same shape as `x`"
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.reshape(-1, 1)
        loss = self.top_k_gating(x, skip_mask, sample_topk=sample_topk, return_indices=return_indices)
        if return_indices:
            loss, idxs = loss
            idxs = idxs.view(bsz, length, -1)
        expert_inputs = x[self.batch_index]
        expert_outputs = self.experts(expert_inputs, self.expert_size)
        zeros = torch.zeros((bsz * length * self.k, self.head_size), dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.index_sorted_experts, expert_outputs)
        y = y.view(bsz, length, self.k, -1)
        if return_indices:
            return y, idxs, loss
        return y, loss

    def reduce(self, x, multiply_by_gates=True):
        bsz, length, k, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        expert_inputs = x[self.index_sorted_experts]
        expert_outputs = self.output_experts(expert_inputs, self.expert_size)
        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]
        zeros = torch.zeros((bsz * length, self.input_size), dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        return y.view(bsz, length, self.input_size)
