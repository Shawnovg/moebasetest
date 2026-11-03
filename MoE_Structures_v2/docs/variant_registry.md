# Variant registry: every configuration that is run

Each row is one YAML in `models/<structure>/configs/`. Every routed variant is run under the three
balancing modes (`none`, `lb`, `warmup`) except `moe/hash` (no router: `none` only).
"Source" says where the choice comes from: the released code (code), the paper text (paper), or
this study (ours).

| Structure | Variant | What changes | Units chosen / available | Source |
|---|---|---|---|---|
| Dense | default | 4 heads x 32, FFN 512 | - | baseline |
| Dense-wide | default | FFN 2,048 (= MoE's total FFN parameters, all active) | - | ours (control) |
| MoE | default | 8 FFN experts x 256, MLP router, renormalised top-2 | 2 / 8 | code (SUT) |
| MoE | top1 | linear router, raw-probability gate, top-1 | 1 / 8 | Switch Transformer |
| MoE | hash | fixed token-id -> 2 experts, gate 1/2, no router | 2 / 8 | Roller et al. 2021 |
| MoH | default | 8 heads x 32, router keeps 4 heads, renormalised x4 | 4 / 8 | code (MoH, routed only) |
| MoH | shared2 | 2 always-on heads + 2 of 6 routed, two-stage weights | 2+2 / 8 | code + paper (main form) |
| MoH | k6 | router keeps 6 of 8 heads | 6 / 8 | ours |
| MoH | k8 | all 8 heads (gate-weighted): no selection; matches MoA k8 | 8 / 8 | ours (dyn study) |
| MoA | default | 8 single-head query experts, shared K/V head, raw top-4 gates, zero-init gate | 4 / 8 | code (MoA) |
| MoA | renorm_detached | gates renormalised over top-4 (denominator detached) | 4 / 8 | paper |
| MoA | k6 | 6 of 8 query experts; matches MoH k6 | 6 / 8 | ours (dyn study) |
| MoA | k8 | all 8 experts active (gate-weighted) | 8 / 8 | ours (no selection) |
| MoMHA | default | 4 experts x 2 heads, shared K/V heads, MLP router, renormalised top-2 | 2 / 4 | code (SUT) |
| MoMHA | top1 | 4 experts x 4 heads, raw gate, top-1 | 1 / 4 | ours (raw gate: F1) |
| MoM | default | pool 4 attention + 4 FFN modules + SKIP, 2 steps, top-1, MLP router | 1 / 5 per step and type | code (MoM, naive router) |
| MoM | k2 | top-2 modules per step | 2 / 5 | code option |
| MoM | noskip | no SKIP class | 1 / 4 | code option (`enable_pause=False`) |
| MoM | gru | GRU router carried across steps | 1 / 5 | code / paper default |
| MiTA | default | m = 8 landmarks, top-8 keys per expert, 1 expert per query, shared part | 1 / 8 experts | code (MiTA, scaled down) |
| MiTA | m4k4 | m = k = 4 | 1 / 4 | ours (sparser) |
| MiTA | m16k16 | m = k = 16 | 1 / 16 | ours (denser) |
| MiTA | s2 | 2 experts per query | 2 / 8 | code (`_mita_attention.py`) |
| MiTA | noshared | routed experts only | 1 / 8 | ours (ablation) |

## Balancing modes

| Mode | What it does | Per structure |
|---|---|---|
| none | task loss only | all |
| lb | + the reference's own balancing loss | MoE / MoMHA: Switch 0.01; MoA: Switch 0.01 + z 0.001; MoH: MoH loss 0.01 averaged over layers; MoM: `load_balancing_loss_func` 0.01; MiTA: Switch-form 0.01 (ours, no reference loss) |
| warmup | first 10 % of steps route uniformly at random (gates still from the router) | all routed |

## Dynamics study (datasets dyn, dyn_fast)

Balancing mode `none` only (no balancing loss, no random warm-up), seeds 0-2.
* `dyn_main`: Dense, Dense-wide, MoE, MoH, MoA, MoMHA default configs on dyn and dyn_fast.
* `dyn_variants` (dyn): MoE top1; MoH k6, k8; MoA k6, k8, renorm_detached; MoMHA top1.
  Not run: MoE hash (routes by token id; the dynamics input has no tokens), MoH shared2 (shared heads, excluded
  by design), all MoM and MiTA configs. MoH k8 and MoA k6 were added so MoH and MoA are compared at the
  same k (4, 6, 8); they were not run on the token tasks.
