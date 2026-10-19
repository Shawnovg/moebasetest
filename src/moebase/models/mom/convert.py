"""Phase 1 -> phase 2 transfer: dense donor layers become MoM modules.

Exact port of Mixture-of-Modules@2ed1d28 ``mom/utils.py::get_layer_map / get_name_map /
init_from_vanilla``, taking a state dict instead of a file path, and *cloning* tensors (upstream
assigns ``p.data = state_dict[...]``, sharing storage with the loaded checkpoint).

Mapping (vanilla layer index -> MoM position), consumed in order:
    bottom layers 1:1 -> MoTLayer; for each chunk, the next max(num_ffn, num_attn) vanilla layers
    become module 0..N-1 of that MoTBlock (attention *and* FFN of a donor layer go to the same
    module index); then one vanilla layer -> the anchor MoTLayer; then top layers.
    wte and wpe are copied. ln_f is *not* in the map, so the final LayerNorm keeps its fresh
    initialisation (finding M4). Routers are trained from scratch.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .model import MoMConfig


def get_layer_map(config: MoMConfig) -> Dict[str, str]:
    layer_map = {}
    l_new, l_old = 0, 0
    modules_per_chunk = max(config.num_ffn, config.num_attn)
    for _ in range(config.num_bottom_layers):
        layer_map[f"{l_old}"] = f"{l_new}"
        l_new += 1
        l_old += 1
    for _ in range(config.chunks):
        for j in range(modules_per_chunk):
            layer_map[f"{l_old}"] = f"{l_new}-{j}"
            l_old += 1
        l_new += 1
        if config.require_anchor_layer:
            layer_map[f"{l_old}"] = f"{l_new}"
            l_new += 1
            l_old += 1
    for _ in range(config.num_top_layers):
        layer_map[f"{l_old}"] = f"{l_new}"
        l_new += 1
        l_old += 1
    return layer_map


def get_name_map(state_dict, config: MoMConfig) -> Dict[str, str]:
    """Map MoM parameter name -> vanilla (donor) parameter name."""
    name_map = {}
    if "transformer" in list(state_dict.keys())[0]:
        add_transformer_prefix = False
        name_map["transformer.wte.weight"] = "transformer.wte.weight"
        name_map["transformer.wpe.weight"] = "transformer.wpe.weight"
    else:
        add_transformer_prefix = True
        name_map["transformer.wte.weight"] = "wte.weight"
        name_map["transformer.wpe.weight"] = "wpe.weight"
    layer_map = get_layer_map(config)
    for name_old in state_dict.keys():
        for layer_old, layer_new in layer_map.items():
            prefix = f"h.{layer_old}."
            if prefix in name_old:
                name_new = name_old.replace(".ln_1.", ".mot_attn.attention_layernorm.")
                name_new = name_new.replace(".ln_2.", ".mot_ffn.ffn_layernorm.")
                name_new = name_new.replace(".attn.", ".mot_attn.")
                name_new = name_new.replace(".mlp.", ".mot_ffn.")
                if "-" not in layer_new:
                    name_new = name_new.replace(f"h.{layer_old}.", f"h.{layer_new}.")
                else:
                    l, m = layer_new.split("-")
                    name_new = name_new.replace(f"h.{layer_old}.", f"h.{l}.")
                    name_new = name_new.replace(".weight", f".{m}.weight")
                    name_new = name_new.replace(".bias", f".{m}.bias")
                if add_transformer_prefix:
                    name_new = f"transformer.{name_new}"
                name_map[name_new] = name_old
                break
    return name_map


@torch.no_grad()
def init_from_vanilla(model: nn.Module, state_dict, config: MoMConfig):
    """Copy donor tensors into MoM. Returns (model, report) where report lists what was/wasn't set."""
    name_map = get_name_map(state_dict, config)
    copied, not_copied = [], []
    for n, p in model.named_parameters():
        if n in name_map:
            src = state_dict[name_map[n]]
            assert p.shape == src.shape, (n, p.shape, src.shape)
            p.data = src.detach().clone().to(p.device, p.dtype)
            copied.append((n, name_map[n]))
        else:
            not_copied.append(n)
    # keep the LM head tied to wte, as HF does after loading
    if hasattr(model, "lm_head") and hasattr(model, "transformer"):
        model.lm_head.weight = model.transformer.wte.weight
    donor_used = {v for _, v in copied}
    donor_unused = [k for k in state_dict if k not in donor_used and not k.endswith(".attn.bias")]
    return model, {"copied": copied, "not_copied": not_copied, "donor_unused": donor_unused}
