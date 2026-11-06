# Follow-up after the 5 Nov supervisor feedback (runs finished 7 Nov 2026)

Interpretations taken:
1. p_stay 0.5 → new dataset `dyn_p50` (dyn 0.95 and dyn_fast 0.80 kept).
2. Multi-head attention on MoH → MoH `shared2`: 2 always-on shared heads + 2 of 6 routed (MoH paper's main form), same 4 active heads.
3. Prediction figure for a normal Transformer → Dense (MHA-4), MHA-8 next to MoH (`pred_compare_dyn.png`, `dyn_pred_dyn_dense.png`).
4. mod → per-pair diagnosis + Dense/MoH 30k steps, weight decay 1.0 (seed 0).
5. No MoE → attention-only comparisons: MoH vs MHA-4/MHA-8, MoA vs MQA-4/MQA-8, MoMHA vs GQA-4/GQA-8.

Routed models use the load-balance loss (0.01); 3 seeds unless stated.

## Excess over the Bayes floor (x1e-3, positions >= 8)
Floors: dyn 0.00989, dyn_fast 0.01889, dyn_p50 0.02942.

| Model | dyn | dyn_fast | dyn_p50 |
|---|---|---|---|
| Dense (MHA-4) | 0.192 | 0.205 | 0.088 |
| MHA-8 | 0.193 | 0.205 | 0.103 |
| MoH | 0.157 | 0.168 | 0.086 |
| MQA-4 | 0.188 | 0.206 | 0.084 |
| MQA-8 | 0.194 | 0.215 | 0.095 |
| MoA | 0.176 | 0.179 | 0.093 |
| GQA-4 | 0.193 | 0.198 | 0.082 |
| GQA-8 | 0.194 | 0.216 | 0.097 |
| MoMHA | 0.167 | 0.184 | 0.090 |

Routing gain holds at p_stay 0.95/0.80, vanishes at 0.50.

## MoH with shared heads
| Model | dyn | dyn_fast | dyn_p50 | fact | mod |
|---|---|---|---|---|---|
| Dense | 0.192 | 0.205 | 0.088 | 0.54 | 0.30 |
| MHA-8 | 0.193 | 0.205 | 0.103 | 0.80 | 0.30 |
| MoH (4 of 8) | 0.157 | 0.168 | 0.086 | 0.46 | 0.28 |
| MoH (2 shared + 2 of 6) | 0.157 | 0.175 | 0.080 | 0.36 | 0.26 |

## Predictions (dyn, test sequence 0)
MSE: Dense 0.0101, MHA-8 0.0102, MoH 0.0102, Bayes 0.0101. Routing gain ≈ 0.035e-3 MSE ≈ 0.3 % of the error.

## mod
3k steps: held-out accuracy ≈ 0.9 only for + and × when (b, a) was trained; otherwise chance (0.00–0.03). Memorisation + commutativity.
30k steps, wd 1.0 (seed 0): Dense test 1.000, MoH 0.9999; 1.00 on every group including a − b → the rule is learned (grokking). Steps and wd changed together; one seed.

Tables: reports/tables/followup_dyn3.csv, followup_shared.csv, followup_mod_swap.csv, dyn_reference.csv.
