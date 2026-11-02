"""MoMHA structure: multi-head attention experts sharing K/V heads (swap A) + dense FFN."""
from backbone.layers import DenseFFN

from .momha_attention import MoMHAAttention


def build_sublayers(cfg, rope, state, layer_idx):
    bb, p = cfg["backbone"], cfg["structure_params"]
    attn = MoMHAAttention(bb["d_model"], bb["head_dim"], p["heads_per_expert"], p["num_experts"], p["k"],
                          rope, state, switchloss=p.get("switchloss", 0.01),
                          gate_norm=p.get("gate_norm", "renorm"))
    return attn, DenseFFN(bb["d_model"], bb["d_ff"], bb.get("activation", "gelu"))
