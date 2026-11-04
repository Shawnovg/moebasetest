# Switching-dynamics results — 2026-11-04

## Setup

The task is next-state prediction on a 4-D rotating-coordinates time series. Three transition matrices are chosen by a hidden 3-state Markov regime. The generator is Twilight's, unchanged.

| Dataset | Stay probability | Regime switches about every |
|---|---|---|
| `dyn` | 0.95 | 20 steps |
| `dyn_fast` | 0.80 | 5 steps |

- **Sizes:** 50k training sequences of 64 steps, 1k validation, 2k test. Data is in `E:\data\MoE_Structures_v2\dyn*`.
- **Structures:** Dense, Dense-wide, MoE, MoH (no shared heads), MoA, MoMHA. MoM and MiTA are excluded.
- **Balancing:** none (no load-balance loss, no random warm-up).
- **Model changes:** causal attention, linear input and output, MSE loss.
- **Seeds:** 3 per configuration.

## Reference predictors (MSE, positions t ≥ 8)

| Predictor | dyn | dyn_fast |
|---|---|---|
| Noise floor (knows the future regime) | 0.0064 | 0.0064 |
| Bayes floor (knows the last regime) | 0.00989 | 0.01889 |
| Matrix that best fits the last transition | 0.0106 | 0.0215 |

## Excess over the Bayes floor (×1e-3, mean ± sd, 3 seeds)

| Structure | dyn | dyn_fast |
|---|---|---|
| Dense | 0.192 ± 0.006 | 0.205 ± 0.008 |
| Dense-wide | 0.179 ± 0.009 | 0.185 ± 0.002 |
| MoE | 0.180 ± 0.008 | 0.188 ± 0.022 |
| MoH | 0.162 ± 0.003 | 0.175 ± 0.010 |
| MoA | 0.183 ± 0.008 | 0.183 ± 0.006 |
| MoMHA | 0.169 ± 0.001 | 0.177 ± 0.009 |

### Variants on dyn

| Variant | Excess (×1e-3) |
|---|---|
| MoH k6 | 0.160 |
| MoH k8 | 0.162 |
| MoA k6 | 0.184 |
| MoA k8 | 0.195 |
| MoA renorm_detached | 0.177 |
| MoE top1 | 0.219 ± 0.026 (layer-0 collapse) |
| MoMHA top1 | 0.168 ± 0.015 |

## Findings

- **All structures land within about 2% of the Bayes floor.** The differences between structures are only 0.1–0.3% of the total error, but the ranking is consistent across both datasets and every seed.
- **MoH beats MoA at every matched k** (4, 6, 8), so MoH's advantage comes from its architecture, not from head selection.
- **Most of MoH's gain is in steady state** (8 or more steps into a regime), not at regime switches.
- **First-layer attention routers carry zero regime information**, because they see only the current state x[t].
- **The MoE FFN router in layer 2 is the most regime-specific** (normalised regime MI 0.55), yet MoE gains the least. Regime specialisation does not track accuracy.
- **The task is near-saturated for this backbone.** A harder variant would be needed to separate the structures more clearly, for example more regimes, more noise, higher dimension or a smaller model.

Technical report section: "Switching-dynamics study" in the MoE_Structures_v2 technical report. Git commit: 288237b on `structures-v2`.

## Update (4 Nov evening): load-balance loss and router-free attention controls

### Routed models retrained with the load-balance loss (excess ×1e-3, 3 seeds)

| | dyn none | dyn LB | dyn_fast none | dyn_fast LB |
|---|---|---|---|---|
| MoH | 0.162 | **0.157** | 0.175 | **0.168** |
| MoMHA | 0.169 | 0.167 | 0.177 | 0.184 |
| MoA | 0.183 | 0.176 | 0.183 | 0.179 |
| MoE | 0.180 | 0.181 | 0.188 | 0.193 |

MoH stays best. MoE top-1 is now worse than top-2 (0.229 vs 0.181), consistent with the earlier top-1 advantage being a router-collapse artefact. Layer-2 regime MI (dyn): MoE 0.69, MoH 0.27, MoMHA 0.24, MoA 0.09; attention layer 1 still 0.

### Router-free attention controls (all 8 query heads on every token, no router)

MHA-8 (8 own K/V) ↔ MoH, MQA-8 (1 shared K/V) ↔ MoA, GQA-8 (2 shared K/V) ↔ MoMHA. 3 seeds each.

| | dyn | dyn_fast | fact acc | mod acc |
|---|---|---|---|---|
| Dense | 0.192 | 0.205 | 0.54 | 0.30 |
| MHA-8 | 0.193 | 0.205 | **0.80** | 0.30 |
| MQA-8 | 0.194 | 0.215 | 0.66 | 0.28 |
| GQA-8 | 0.194 | 0.216 | 0.69 | 0.28 |
| MoH (none / LB) | 0.162 / 0.157 | 0.175 / 0.168 | 0.52 / 0.46 | 0.28 / 0.28 |

- Dynamics: all three controls are no better than Dense; every routed variant beats its control. The gain comes from per-token head selection, not from having 8 heads.
- fact: the controls are far better than the routed models (MHA-8 0.80 vs MoH 0.52; Dense-wide 0.98). Sparse routing in attention hurts memorisation; extra always-on attention parameters help.
- mod: no structure beats Dense (0.30); routing is neutral to harmful.

Table: reports/tables/attn_baselines.csv. Figure: reports/slides/figures/attn_baselines.png (script reports/slides/make_attn_baselines.py).

### Active-width controls: MQA-4 and GQA-4 (4 query heads, no router; MHA-4 = Dense)

Without the router, MoH / MoA / MoMHA activate exactly the attention weights of MHA-4 / MQA-4 / GQA-4 (65.5k / 41.0k / 49.2k per layer).

| | dyn | dyn_fast | fact acc | mod acc |
|---|---|---|---|---|
| MHA-4 (= Dense) | 0.192 | 0.205 | 0.54 | 0.30 |
| MQA-4 | 0.188 | 0.206 | 0.51 | 0.21 |
| GQA-4 | 0.193 | 0.198 | 0.59 | 0.24 |

- Dynamics: every 4-head control sits at Dense level; every routed model beats both its 4-head and 8-head control. Routing, not width, gives the gain.
- fact: at matched active width routing is no better (MoH 0.52 vs 0.54) or worse (MoA 0.35–0.46 vs 0.51; MoMHA 0.34–0.45 vs 0.59); the 8-head controls are best.
- mod: routed ≈ their 4-head controls; nothing beats Dense.
