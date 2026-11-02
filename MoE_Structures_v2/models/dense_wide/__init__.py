"""Dense-wide control: the Dense model with FFN width 2,048, i.e. the same FFN parameter count as
MoE (8 experts x 256) but every token uses all of it. Separates "more parameters" from "routing"."""
from models.dense import build_sublayers  # noqa: F401  (d_ff comes from structure_params)
