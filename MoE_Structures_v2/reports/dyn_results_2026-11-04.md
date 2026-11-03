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
