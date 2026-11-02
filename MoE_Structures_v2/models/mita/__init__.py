"""MiTA structure: Mixture-of-Top-k attention (swap A) + dense FFN. See mita_attention.py."""
from backbone.layers import DenseFFN

from .mita_attention import MiTAAttention


def build_sublayers(cfg, rope, state, layer_idx):
    bb, p = cfg["backbone"], cfg["structure_params"]
    attn = MiTAAttention(bb["n_heads"] * bb["head_dim"], bb["n_heads"], p["num_landmarks"], p["kv_topk"],
                         rope, state, router_topk=p.get("router_topk", 1), shared=p.get("shared", True),
                         lb_coef=p.get("lb_coef", 0.01))
    return attn, DenseFFN(bb["d_model"], bb["d_ff"], bb.get("activation", "gelu"))
