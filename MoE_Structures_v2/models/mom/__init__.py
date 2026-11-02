"""MoM structure: the whole 2-layer stack replaced by a routed pool of modules (swap A + B + depth)."""
from .mom_body import MoMBody


def build_body(cfg, rope, state):
    bb, p = cfg["backbone"], cfg["structure_params"]
    return MoMBody(bb["d_model"], bb["n_heads"], bb["head_dim"], p.get("d_ff", bb["d_ff"]), rope, state,
                   num_attn=p.get("num_attn", 4), num_ffn=p.get("num_ffn", 4),
                   depth=p.get("depth", bb["n_layers"]), routing_top_k=p.get("k", 1),
                   enable_pause=p.get("enable_pause", True), router_type=p.get("router_type", "naive"),
                   router_dim=p.get("router_dim", 256), aux_coef=p.get("aux_coef", 0.01),
                   activation=bb.get("activation", "gelu"), eps=bb.get("ln_eps", 1e-5))
