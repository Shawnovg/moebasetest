# MoE_Structures_v2

Structure-only comparison of routed Transformer layers (MoE, MoH, MoA, MoMHA, MoM, MiTA) against a Dense
baseline on synthetic tasks, with two balancing strategies (each paper's load-balance loss vs random routing
for the first 10 % of steps).

## Quick start

```bash
pip install torch numpy pandas pyyaml matplotlib nbformat          # CPU or CUDA torch
python -m data_gen.build_all                                         # writes E:\data\MoE_Structures_v2\<dataset>\*.npz
python -m experiments.run --structure moe --dataset fact --balance lb --seed 0      # ENTRY: one trial
python -m experiments.sweep --phase main --datasets fact --workers 2                # many trials (resumable)
python -m analysis.report_tables                                     # tables + figures into reports/
python tools/run_notebook.py notebooks/04_main_sweep.ipynb         # or open the notebooks in Jupyter
python -m pytest tests                                               # parity tests against the reference code
```

Paths: `backbone/paths.py` (data root defaults to `E:\data\MoE_Structures_v2` on Windows; override with the
environment variable `MOEV2_DATA`, runs with `MOEV2_RUNS`).

## Where things are

* `configs/base.yaml` — backbone and training settings shared by every structure.
* `models/<structure>/` — the routed module, `configs/default.yaml` + one YAML per variant, an analysis notebook.
* `backbone/` — shared model, data, training loop, routing state and balancing losses.
* `experiments/` — `run.py` (one trial), `plan.py` (which trials each phase contains), `sweep.py`.
* `analysis/` — result collection, routing statistics, figures, structure schematics.
* `docs/code_map.md` — directory tree and the full call chain; `docs/deviations.md` — every place we differ
  from the reference code and the issues we found in it; `docs/variant_registry.md` — all variants.
* `references/README.md` — the reference repositories and commits.
