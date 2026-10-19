"""Stick-breaking dynamic halting used by SUT.

Exact port of SUT@192cbd3 ``halting.py::ACTWrapper`` (the class the encoder and decoder actually
instantiate). ``StickBreakingACT`` in the same upstream file is unused by the released model and is
not ported.

Behaviour of the released code, documented because the paper's notation leaves it open:

* Step 0 runs the block on every non-pad token without a gate (``p_never_halt = pad_mask``).
* From step 1 on, the gate reads the previous *raw* block output ``prev_h`` (not the soft-halted
  state), log-space stick breaking: ``log_halt = log_never_halt + log_softmax(gate(prev_h))``.
  ``p = exp(log_halt[...,1])`` is the probability of halting *at the previous output*.
* ``acc_h += p * prev_h``; a token whose survival ``exp(log_never_halt) < 1 - threshold`` gets
  ``p_never_halt = 0``. It is then not dispatched to any expert (true sparse skip) and its key/value
  input (``self_attn_input``) is frozen. Queries still use the raw state ``x``.
* Halted tokens remain visible to other tokens as keys/values through the frozen soft state.
* At the depth cap, the remaining survival mass is assigned to the last block output:
  ``soft = acc_h + p_never_halt * curr_h``; the model output is this soft state.
* ACT loss: ``mean_over_tokens(acc_expect_depth + p_never_halt * i)``, where halting at the output
  of step j counts depth j (0-based).
* The gate's last layer is zero-initialised, so the first gate gives p = 0.5.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ACTWrapper(nn.Module):
    def __init__(self, mod: nn.Module, embed_dim: int, threshold: float = 0.999, halting_dropout: float = 0.0):
        super().__init__()
        self._gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(halting_dropout),
            nn.Linear(embed_dim, 2, bias=False),
        )
        nn.init.zeros_(self._gate[-1].weight)
        self.threshold = threshold
        self.mod = mod

    def gate(self, h):
        return F.log_softmax(self._gate(h), dim=-1)

    @staticmethod
    def update_halting(log_g, log_never_halt):
        log_halt = log_never_halt[..., None] + log_g
        log_never_halt = log_halt[..., 0]
        p = torch.exp(log_halt[..., 1])
        return p, log_never_halt

    def forward(self, prev_act_state, prev_h, self_attn_input, pad_mask, layer_idx, **kwargs):
        if prev_act_state is None:
            log_never_halt = acc_expect_depth = torch.zeros_like(prev_h[..., 0])
            acc_h = torch.zeros_like(prev_h)
            i = 0
            p_never_halt = pad_mask
        else:
            (i, log_never_halt, acc_h, acc_expect_depth) = prev_act_state
            log_g = self.gate(prev_h)
            p, log_never_halt = self.update_halting(log_g, log_never_halt)
            acc_h = acc_h + p[..., None] * prev_h
            acc_expect_depth = acc_expect_depth + i * p
            p_never_halt = log_never_halt.exp()
            p_never_halt = p_never_halt.masked_fill((p_never_halt < (1 - self.threshold)), 0) * pad_mask
            p_never_halt = p_never_halt.contiguous()
            i = i + 1

        curr_act_state = (i, log_never_halt, acc_h, acc_expect_depth)
        outputs = self.mod(prev_h, self_attn_input, p_never_halt, layer_idx, **kwargs)
        curr_h = outputs[0]
        if prev_act_state is not None:
            self_attn_input = torch.where(
                p_never_halt[..., None] < (1 - self.threshold),
                self_attn_input,
                (acc_h + p_never_halt[..., None] * curr_h).type_as(self_attn_input),
            )
            act_loss = (acc_expect_depth + p_never_halt * i) * pad_mask
            act_loss = act_loss.sum() / pad_mask.sum()
        else:
            self_attn_input = curr_h
            act_loss = 0
        # p_never_halt is returned for diagnostics (active-token counts per step)
        return curr_act_state, outputs, self_attn_input, act_loss, p_never_halt
