# Code map: how one trial runs through the project

## Directory layout

```
MoE_Structures_v2/
├── backbone/            shared code used by every structure
│   ├── paths.py         where data (E:\data\MoE_Structures_v2) and runs live
│   ├── config.py        YAML loading: configs/base.yaml + models/<s>/configs/<variant>.yaml + overrides
│   ├── layers.py        RoPE, masked softmax attention, DenseAttention, DenseFFN
│   ├── experts.py       SUT ParallelExperts + compute_gating (used by MoE, MoMHA, MoA)
│   ├── routing.py       RoutingState (balance mode, warm-up flag) + every paper's balancing loss
│   ├── model.py         EncoderModel: embedding -> 2 pre-LN blocks (or MoM body) -> head
│   ├── data.py          Dataset: .npz splits -> padded batches; DynamicsDataset: state sequences
│   ├── metrics.py       evaluate(): accuracy, per-task accuracy, routing usage
│   ├── metrics_dyn.py   evaluate_dynamics(): MSE, error by regime age, rollout, routing per regime
│   └── train.py         train_run() / train_dynamics_run(): the training loops, write runs/<name>/
├── models/<structure>/  one directory per structure
│   ├── __init__.py      build_sublayers(...) (or build_body for MoM): what the backbone calls
│   ├── <module>.py      the routed module ported from the reference code
│   ├── configs/*.yaml   default.yaml + one YAML per variant
│   └── <s>_analysis.ipynb   per-structure notebook (configs, schematic, results, routing)
├── data_gen/            vocab.py, tasks.py (generators), build_all.py (writes the token datasets),
│                        dynamics.py (switching-dynamics generator, writes dyn / dyn_fast)
├── experiments/         run.py (ENTRY: one trial), plan.py (phases), sweep.py (parallel runner)
├── analysis/            collect.py, routing.py, figures.py, schematics.py, report_tables.py,
│                        dyn_reference.py (no-learning reference predictors), dynamics.py (dyn tables/figures)
├── notebooks/           00-07 notebooks (07 = dynamics study) + _build_notebooks.py (generates all notebooks)
├── tests/               test_parity.py (vs reference code), test_mita.py, ref_loader.py
├── tools/run_notebook.py   execute a notebook without Jupyter
├── configs/base.yaml    shared backbone + training settings
├── docs/                deviations.md, variant_registry.md, this file
├── references/README.md reference repositories and commits
├── runs/                one folder per trial (config.json, history.jsonl, result.json, model.pt)
└── reports/             figures/, tables/
```

## Call chain of one trial

```
python -m experiments.run --structure moa --variant default --dataset mqar --balance warmup --seed 0
│
├─ experiments/run.py  main()
│    └─ backbone/config.py  load_run_config("moa", "default", {dataset, routing.balance, train.seed})
│         reads configs/base.yaml, then models/moa/configs/default.yaml (base: inheritance for variants)
│
└─ backbone/train.py  train_run(cfg)
     ├─ backbone/data.py   load_vocab(), Dataset("mqar", "train"/"valid"/"test")  <- DATA_ROOT/mqar/*.npz
     ├─ backbone/model.py  EncoderModel(cfg, vocab)
     │    ├─ structure_package("moa")  -> models/moa/__init__.py
     │    │    └─ build_sublayers(cfg, rope, state, i) for i in 0, 1
     │    │         -> models/moa/moa_attention.py  MoAAttention  (MoAGate -> backbone/experts.py)
     │    │         -> backbone/layers.py           DenseFFN
     │    └─ Block(ln1, attn, ln2, ffn) x 2,  final LayerNorm, head
     ├─ loop over train.steps:
     │    state.random_active = step < warmup_frac * steps     (balance == "warmup")
     │    logits, aux = model(tokens, pad_mask, select=answer slots)
     │        Block.forward -> MoAAttention.forward(x, key_pad, state)
     │            routing: MoAGate.top_k_gating (random units during warm-up)
     │            aux: backbone/routing.py switch_loss + z_loss   (only if balance == "lb")
     │    loss = cross_entropy(logits, answers) + aux ; AdamW step
     │    every eval_every steps: backbone/metrics.py evaluate(valid) -> history.jsonl
     └─ end: evaluate(valid), evaluate(test, routing=True) -> result.json ; model.pt ; config.json
```

MoM differs only in `EncoderModel`: `models/mom/__init__.py` provides `build_body`, so the two blocks
are replaced by `models/mom/mom_body.py` MoMBody (pool of modules applied for 2 assembly steps).

## Sweeps and analysis

```
python -m experiments.sweep --phase screen|main|variants [--datasets ...] --workers 2
    experiments/plan.py phase()  -> list of trials -> one `experiments.run` subprocess per trial (skips finished)
python -m analysis.report_tables
    analysis/collect.py  load_results / load_history / load_routing  (reads runs/*/result.json, history.jsonl)
    analysis/routing.py  collapse and task-specialisation tables
    analysis/figures.py  every result figure;  analysis/schematics.py  the structure diagrams
    -> reports/tables/*.csv, reports/figures/*.png
```

## The dynamics study (datasets dyn / dyn_fast)

```
python -m data_gen.dynamics                      -> DATA_ROOT/dyn, DATA_ROOT/dyn_fast (states + hidden regimes)
python -m experiments.run --structure moh --dataset dyn --seed 0
    backbone/config.py  load_run_config(...) -> apply_task(): dataset dyn* => cfg["task"] = "dynamics", state_dim 4
    backbone/train.py   train_run(cfg) -> train_dynamics_run(cfg)
        backbone/data.py      DynamicsDataset (normalised with the training mean/std from manifest.json)
        backbone/model.py     EncoderModel: embed = Linear(4 -> 128), head = Linear(128 -> 4),
                              RoutingState(causal=True) -> every attention module masks future keys
                              (layers.masked_softmax_attention / MoA / MoMHA: layers.causal_mask)
        loss = MSE(predicted next state, true next state) at every position
        backbone/metrics_dyn.py  evaluate_dynamics(): MSE (original units), by regime age, 16-step rollout,
                                 unit picks per hidden regime -> result.json
python -m experiments.sweep --phase dyn_main | dyn_variants      (experiments/plan.py)
python -m analysis.dyn_reference    reference predictors (Bayes floor, oracles, persistence)
python -m analysis.dynamics         reports/tables/dyn_*.csv, reports/figures/dyn_*.png
notebooks/07_dynamics.ipynb         the same, step by step
```
MoM and MiTA are not wired for the dynamics task (EncoderModel raises an error): they are excluded from
this study and their modules have no causal mask.
