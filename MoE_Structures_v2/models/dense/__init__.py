"""Dense baseline: standard multi-head attention + dense FFN (the shared backbone, unchanged)."""
from backbone.layers import DenseAttention, DenseFFN


def build_sublayers(cfg, rope, state, layer_idx):
    bb = cfg["backbone"]
    p = cfg.get("structure_params", {})
    d_ff = p.get("d_ff", bb["d_ff"])
    return (DenseAttention(bb["d_model"], bb["n_heads"], bb["head_dim"], rope, out_scale=p.get("out_scale", 1.0)),
            DenseFFN(bb["d_model"], d_ff, bb.get("activation", "gelu")))
