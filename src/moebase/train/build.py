"""Build models, optimisers and schedules from a run configuration."""
from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn

from ..data.logic import Vocab
from ..models.mom.donor import GPT2Classifier, GPT2LMHead
from ..models.mom.model import MoMCausalLM, MoMClassifier, MoMConfig
from ..models.sut.model import SUTClassifier, SUTConfig, SUTSeq2Seq

KINDS = {
    "sut_classifier": ("classifier", SUTConfig, SUTClassifier),
    "sut_seq2seq": ("seq2seq", SUTConfig, SUTSeq2Seq),
    "mom_causal": ("causal", MoMConfig, MoMCausalLM),
    "mom_classifier": ("classifier", MoMConfig, MoMClassifier),
    "gpt2_causal": ("causal", MoMConfig, GPT2LMHead),
    "gpt2_classifier": ("classifier", MoMConfig, GPT2Classifier),
}


def build_model(model_cfg: dict, vocab: Vocab) -> Tuple[nn.Module, str, object]:
    kind = model_cfg["kind"]
    interface, cfg_cls, model_cls = KINDS[kind]
    params = dict(model_cfg.get("params", {}))
    params["vocab_size"] = len(vocab)
    if cfg_cls is SUTConfig:
        params["padding_idx"] = vocab.pad
        cfg = SUTConfig.from_dict(params)
    else:
        if kind.endswith("classifier"):
            params["causal"] = False
        cfg = MoMConfig(**params)
    return model_cls(cfg), interface, cfg


def param_groups(model: nn.Module, weight_decay: float, hf_style: bool):
    """hf_style: no decay for biases and LayerNorm weights (HF Trainer convention)."""
    if not hf_style or weight_decay == 0:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": weight_decay}]
    no_decay = set()
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            if pn.endswith("bias") or isinstance(m, nn.LayerNorm):
                no_decay.add(id(p))
    decay = [p for p in model.parameters() if p.requires_grad and id(p) not in no_decay]
    nd = [p for p in model.parameters() if p.requires_grad and id(p) in no_decay]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": nd, "weight_decay": 0.0}]


def build_optimizer(model: nn.Module, o: dict):
    typ = o.get("type", "adamw")
    if typ == "adam_fairseq":  # SUT scripts: --optimizer adam --adam-betas (0.9,0.98) --adam-eps 1e-9
        return torch.optim.AdamW(param_groups(model, o.get("weight_decay", 0.0), False), lr=o["lr"],
                                 betas=tuple(o.get("betas", (0.9, 0.98))), eps=o.get("eps", 1e-9))
    if typ == "adamw":
        return torch.optim.AdamW(param_groups(model, o.get("weight_decay", 0.0), o.get("hf_no_decay", True)),
                                 lr=o["lr"], betas=tuple(o.get("betas", (0.9, 0.999))), eps=o.get("eps", 1e-8))
    raise ValueError(typ)


def lr_lambda_factory(o: dict, max_steps: int):
    sched = o.get("schedule", "cosine")
    lr = o["lr"]
    warm = int(o.get("warmup_steps", 0) or round(o.get("warmup_ratio", 0.0) * max_steps))

    if sched == "inverse_sqrt":  # fairseq inverse_sqrt with warmup_init_lr
        init = o.get("warmup_init_lr", 1e-7)

        def f(step):
            s = step + 1
            if s < warm:
                return (init + s * (lr - init) / warm) / lr
            return math.sqrt(warm) / math.sqrt(s) if warm > 0 else 1.0
        return f
    if sched == "cosine":  # transformers.get_cosine_schedule_with_warmup
        def f(step):
            if step < warm:
                return step / max(1, warm)
            progress = (step - warm) / max(1, max_steps - warm)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return f
    if sched == "constant":
        return lambda step: min(1.0, (step + 1) / max(1, warm)) if warm else 1.0
    raise ValueError(sched)
