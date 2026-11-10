"""MoA structure: single-head attention experts sharing one K/V head (swap A) + dense FFN."""
from backbone.layers import DenseFFN

from .moa_attention import MoAAttention


def build_sublayers(cfg, rope, state, layer_idx):
    bb, p = cfg["backbone"], cfg["structure_params"]
    attn = MoAAttention(bb["d_model"], bb["head_dim"], p["num_experts"], p["k"], rope, state,
                        gate_norm=p.get("gate_norm", "raw"), switchloss=p.get("switchloss", 0.01),
                        zloss=p.get("zloss", 0.001), router=p.get("router", "linear"))
    return attn, DenseFFN(bb["d_model"], bb["d_ff"], bb.get("activation", "gelu"))
