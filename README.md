# MoE_Base_Test: small faithful SUT and MoM on logical inference

Small, audited ports of **Sparse Universal Transformer** (Tan et al., EMNLP 2023) and **Mixture-of-Modules**
(Gong et al., 2024), plus dense, UT, Switch-style and MoA-style controls built from the same units, for the
propositional-logic inference benchmark (Bowman et al. 2015, Ordered-Memory release).

* `docs/01_plan_review.md`: review of the research plan, with changes and their reasons (read this first)
* `docs/02_source_audit.md`: paper → upstream line → local module, and upstream findings
* `docs/03_deviation_log.md`: every intentional difference from upstream

## Setup (Windows or Linux)

```bash
pip install -r requirements.txt            # torch, numpy, pyyaml, pytest
python scripts/fetch_upstream.py           # clones SUT, MoM, Ordered-Memory at pinned commits -> third_party/
python scripts/prepare_logic_data.py       # audited splits + manifest -> data/logic-v1/  (~1 min)
python -m pytest                           # unit, parity and smoke tests
python -m pytest -m slow                   # tiny-set learning checks
```

Environment overrides: `MOE_BASE_DATA_DIR`, `MOE_BASE_RUNS_DIR`, `MOE_BASE_THIRD_PARTY`, `MOE_BASE_LOGIC_RAW`.

## Running

```bash
python scripts/profile_models.py                                   # params, MACs, latency per config (pilot)
python scripts/train.py configs/logic/sut_common6.yaml --smoke     # 2 minutes on CPU
python scripts/train.py configs/logic/sut_common6.yaml --seed 0 --no-test     # screening (validation only)
python scripts/train.py configs/logic/sut_ref_seq2seq.yaml --seed 0
python scripts/run_mom_two_phase.py --seed 0                       # donor -> MoM, plus both controls
python scripts/train.py configs/logic/dense6.yaml --set data.regime=sysC     # A/B/C regimes: sysA|sysB|sysC
```

Each run writes `runs/<name>_s<seed>_<cfghash>/`: `record.json` (config, code hash, upstream commits,
environment), `log.jsonl`, `best.pt` (lowest validation Brier score), and `result.json`. The result holds
validation and test metrics (accuracy, macro-F1, Brier, NLL, ECE, confusion matrix, per-length accuracy,
accuracy without order-swapped near-duplicates) and routing usage.

## Configs (`configs/logic/`)

| Config | Family | Fidelity |
|---|---|---|
| `dense6`, `ut_fixed6`, `ut_sb6`, `switch6`, `moa6` | controls | L3 common interface, audited SUT units |
| `sut_common6` (+ `_fixed6`, `_e1_6`, `_mim_paper`, `_lb`, `_act`) | SUT and ablations | L3 common interface |
| `sut_ref_seq2seq` | SUT | L2 units, seq2seq interface, paper geometry scaled |
| `mom_donor6_causal` → `mom_ref6_causal` (+ `_nobal`, `_k2`), `mom_scratch6_causal`, `donor6_continue_causal` | MoM two-phase and controls | L2 units, causal interface |
| `mom_donor6_cls` → `mom_ref6_cls` | MoM | L3 encoder adaptation |

Common scale: width 128, 4 heads × 32 (or the equivalent expert geometry), FFN 512, six block applications.
Freeze final values only after `profile_models.py` and a validation pilot.

## Layout

```
src/moebase/data/logic.py        parser, truth-table checker, splits, manifest, token interfaces (pure Python)
src/moebase/models/sut/          moe.py, attention.py, halting.py, layers.py, model.py
src/moebase/models/mom/          routers.py, model.py, donor.py, convert.py
src/moebase/train/               config, build (models/optimisers/schedules), loop, metrics, data_utils
src/moebase/compute.py           MAC estimates and parameter counts
tests/                           test_data, test_sut_units, test_mom_units, test_parity_upstream, test_smoke_configs
notebooks/                       11_audit_and_data, 14_train_and_compare
```

## Status (18 Oct 2026)

* **Tests:** all 66 pass on PyTorch 2.5 (CPU): unit, smoke, tiny-set learning, and 10 parity tests that match outputs
  and gradients of the authors' SUT and MoM code. For MoM parity without `transformers` installed, the tests use a
  minimal stand-in (`tests/upstream_loader.py::_stub_transformers`).
* **CPU pilot:** executed in `notebooks/MoE_Experiment_executed.ipynb` (31 runs). Results are in `runs/pilot/`
  (`results_long.csv`, `curves.csv`, per-run `result.json`; checkpoints not copied) and figures in `reports/figures/`.
  The technical report is a Claude doc linked from the session; `docs/01_plan_review.md` lists the plan changes.
* **To reproduce or scale up:** open `notebooks/MoE_Experiment.ipynb`, set `PROFILE = "full"` (GPU) and run all cells.
  Finished runs are skipped.

## Mechanistic analysis (27 Oct 2026)

`notebooks/MoE_Analysis.ipynb` looks inside the trained pilot checkpoints:
* linear probing of every step (label, region bits, truth tables, syntax; random-init controls);
* activation patching with minimal pairs;
* a sparse autoencoder on the `<cls>` readout;
* routing statistics;
* the authors' kinds of ablation: depth truncation, halting-threshold sweep, expert knockout, random routing, and MoM forced SKIP / module knockout.

Every unit defines its own functions and caches its results, and each figure has its own unit. Findings: `reports/analysis_summary.md`. Slides: `reports/slides/` (deck sources, figures, LaTeX equations).

The notebook needs the seed-0 `best.pt` checkpoints under `runs/pilot/length/`. They are not in git; the local copy on the D: drive includes them.

The models expose a `step_hook` (SUT encoder, MoM/GPT-2 transformer). It is `None` in training, and lets the notebook capture or patch the hidden state after every block application.
