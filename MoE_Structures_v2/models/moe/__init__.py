"""MoE structure: dense attention + FFN experts (swap B). See moe_ffn.py and configs/."""
from backbone.layers import DenseAttention

from .moe_ffn import MoEFFN


def build_sublayers(cfg, rope, state, layer_idx):
    bb, p = cfg["backbone"], cfg["structure_params"]
    attn = DenseAttention(bb["d_model"], bb["n_heads"], bb["head_dim"], rope)
    ffn = MoEFFN(bb["d_model"], p["expert_hidden"], p["num_experts"], p["k"], state,
                 gate=p.get("gate", "mlp"), gate_norm=p.get("gate_norm", "renorm"), bias=p.get("bias", True),
                 activation=bb.get("activation", "gelu"), switchloss=p.get("switchloss", 0.01),
                 zloss=p.get("zloss", 0.0), routing=p.get("routing", "learned"),
                 vocab_size=cfg.get("vocab_size"), hash_seed=1000 + layer_idx)
    return attn, ffn
