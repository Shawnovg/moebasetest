# Mechanistic analysis of the pilot checkpoints

Notebook: `notebooks/MoE_Analysis.ipynb` (built by `notebooks/_build_analysis_notebook.py`; executed copy:
`notebooks/MoE_Analysis_executed.ipynb`). Each compute unit caches to `runs/analysis/<profile>/`; the small
result tables are copied to `reports/analysis_results/`. Figures: `reports/figures/analysis/` (one per figure
unit) and `reports/slides/figures/` (slide versions). Slide sources: `reports/slides/deck/project/`; the equations
are rendered from LaTeX by `scripts/render_latex.py` (`reports/slides/equations_slides.json`).

Run it with `python scripts/run_notebook.py notebooks/MoE_Analysis.ipynb` (about 70 min on 2 CPU cores
for the `quick` profile; `MOE_ANALYSIS_PROFILE=full` uses larger samples).

## What was measured (seed-0 pilot checkpoints, length regime)

| Model | test acc | label probe (last step) | region probe | 90 % of region gain by step | acc after 2 steps | SAE best \|r\| with region bits | routing entropy |
|---|---|---|---|---|---|---|---|
| Dense | 0.751 | 0.594 | 0.753 | 5 | 0.334 | 0.52 | – |
| UT (fixed depth) | 0.573 | 0.281 | 0.559 | 4 | 0.413 | – | – |
| Switch | 0.664 | 0.465 | 0.677 | 3 | 0.490 | 0.36 | 0.996 |
| MoA | 0.610 | 0.352 | 0.602 | 5 | 0.500 | – | 0.072 |
| SUT fixed-depth | 0.627 | 0.388 | 0.628 | 4 | 0.416 | 0.31 | 0.246 |
| SUT (released MIM sign) | 0.562 | 0.224 | 0.532 | 2 | 0.550 | 0.23 | 0.090 |
| SUT, MIM paper sign | 0.601 | 0.327 | 0.596 | 3 | 0.525 | – | 0.861 |
| SUT, 3x budget | 0.691 | 0.496 | 0.689 | 3 | 0.514 | 0.32 | 0.053 |
| MoM donor (dense) | 0.547 | 0.160 | 0.500 | – | – | – | – |
| MoM | 0.545 | 0.156 | 0.502 | – | – | 0.16 | 0.977 |

Probes report balanced accuracy (chance: 1/7 for the label, 0.5 for the region bits). Random-init controls
stay at chance for the label and region bits at every step.

## Findings

1. **Dense computes the relation over its whole depth.** Region bits become linearly decodable from layer 2
   and keep improving to layer 6. Patching shows the edited token's information stays local through layer 2
   and moves into `<cls>` at layers 3–4 (recovery 0.52 → 0.94). Truncating Dense to 2 layers drops it to 0.33.
2. **SUT (released settings) is effectively a one-to-two-step model.** Its probes plateau after step 1
   (region 0.53); running only one step costs 2 points (0.540 vs 0.560). The halting-threshold sweep changes
   nothing, because the learned halting distribution already puts nearly all mass on steps 0–2 (mean expected
   halting step 1.45; paper sign 0.97; 3x budget 2.69).
3. **The released MIM sign collapses routing.** Normalised expert entropy is 0.09 (paper sign: 0.86, +4 points
   accuracy). Because the MIM weighting `p_x` is the not-halted probability without a stop-gradient, the MIM
   gradient also reaches the halting gate.
4. **Experts are used when depth is available.** Random routing costs SUT fixed-depth 50 points and SUT 3x 20
   points, but the released SUT only 3 points.
5. **MoM has nothing to route.** Its GPT-2-style donor never learned the task in the pilot budget (0.547 vs
   majority 0.541). MoM's probes are at chance at every step, its routers are near-uniform and ignore token
   type (I(E;type)/H(E) ≈ 0.03), and forcing SKIP for the whole 4-step block changes accuracy by 0.05 points.
6. **No model computes explicit truth tables.** Clause truth tables are about as decodable from random
   features (0.66) as from trained models (0.66–0.71).

## Caveats

CPU pilot, one seed per model for the analyses, 3,000 training steps (MoM 10,000). The original paper
figures could not be downloaded in this environment; the slide schematic is redrawn after SUT Fig. 1/3 and
MoM Fig. 1.

Suggested GPU follow-ups:
- SUT with the paper MIM sign and `.detach()` on the MIM weighting.
- A MoM donor trained until it learns the task.
- Probes and patching on 3–5 seeds.
