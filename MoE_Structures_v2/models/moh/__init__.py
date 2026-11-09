"""MoH structure: head-routed attention (swap A) + dense FFN. See moh_attention.py."""
from backbone.layers import DenseFFN

from .moh_attention import MoHAttention


def build_sublayers(cfg, rope, state, layer_idx):
    bb, p = cfg["backbone"], cfg["structure_params"]
    attn = MoHAttention(bb["d_model"], p["num_heads"], bb["head_dim"], rope, state,
                        shared_head=p.get("shared_head", 0), routed_head=p["routed_head"],
                        qkv_bias=p.get("qkv_bias", False), head_group=p.get("head_group", 1),
                        router=p.get("router", "linear"), gate_scale=p.get("gate_scale", "k"),
                        lb_type=p.get("lb_type", "moh"),
                        # MoH train.py averages the per-layer losses (coef / n_layers); MoMHA's switch loss
                        # is 0.01 in every layer, so the block2_sut variant keeps the full coefficient
                        lb_coef=p.get("lb_coef", 0.01) / (1 if p.get("lb_type") == "switch" else bb["n_layers"]))
    return attn, DenseFFN(bb["d_model"], bb["d_ff"], bb.get("activation", "gelu"))
