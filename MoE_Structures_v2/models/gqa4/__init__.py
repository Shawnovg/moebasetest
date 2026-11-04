"""GQA-4: 4 query heads x 32 sharing TWO key/value head(s), no router. Matches the ACTIVE attention width of MoMHA
(4 query heads per token, same key/value sharing); FFN unchanged."""
from backbone.layers import DenseFFN, GroupedAttention


def build_sublayers(cfg, rope, state, layer_idx):
    bb = cfg["backbone"]
    p = cfg.get("structure_params", {})
    return (GroupedAttention(bb["d_model"], p.get("n_heads", 4), p.get("n_kv_heads", 2), bb["head_dim"], rope),
            DenseFFN(bb["d_model"], bb["d_ff"], bb.get("activation", "gelu")))
