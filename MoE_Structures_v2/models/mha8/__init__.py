"""MHA-8: 8 query heads x 32 with their own keys/values (MoH with all 8 heads and no router).
Dense-attention control for the routed-attention structures, as Dense-wide is for MoE; FFN unchanged."""
from backbone.layers import DenseFFN, GroupedAttention


def build_sublayers(cfg, rope, state, layer_idx):
    bb = cfg["backbone"]
    p = cfg.get("structure_params", {})
    return (GroupedAttention(bb["d_model"], p.get("n_heads", 8), p.get("n_kv_heads", 8), bb["head_dim"], rope),
            DenseFFN(bb["d_model"], bb["d_ff"], bb.get("activation", "gelu")))
