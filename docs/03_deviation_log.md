# Deviation log

Every intentional difference between this project and the pinned upstream code. A deviation either
leaves numerics unchanged (checked by a parity test) or is a labelled condition.

| ID | Where | Upstream | Here | Numerics | Why |
|---|---|---|---|---|---|
| D-SUT-1 | `ParallelExperts` | custom autograd Function, FP16 cast under CUDA autocast | split/mm loop under autograd; FP32 unless you enable AMP yourself | identical in FP32 (`test_moe_ffn_parity` grads) | no fairseq/AMP coupling |
| D-SUT-2 | MIM | released sign only | `mim_variant: code` (default) or `paper` | default identical | review R1 |
| D-SUT-3 | MoE statistics | cleared only in training criterion | also cleared before/after/within evaluation; `get_aux_loss_and_clear` returns 0 when empty | training identical | upstream leaks memory in validation |
| D-SUT-4 | MoMHA | incremental decoding, ONNX, quant-noise, bias_kv/zero_attn | removed | identical for single-pass forward (`test_attention_parity`) | unused here |
| D-SUT-5 | Layers | cross_self_attention, scale_attn/heads/fc/resids, checkpoint/FSDP | removed (all off in SUT configs) | identical (`test_*_layer_parity`) | unused |
| D-SUT-6 | Halting | returns 4-tuple; threshold from env var | returns also `p_never_halt`; threshold from config | identical (`test_halting_parity`) | diagnostics |
| D-SUT-7 | Encoder | — | `share_layers: false` builds untied blocks (halting disallowed) | n/a (new condition) | Dense/Switch/MoA families from audited units |
| D-SUT-8 | Task | WMT seq2seq | logic seq2seq: `p1 #SEP# p2 <eos>` → `REL <eos>`; accuracy at position 0 over 7 label tokens; invalid-rate logged | — | SUT paper's logic setup |
| D-SUT-9 | Classifier | — | `<cls>` readout of the encoder output (soft-halted state) | — | controlled interface (L3) |
| D-MoM-1 | Model | HF PreTrainedModel, flash-attn, KV cache, gradient checkpointing | plain nn.Module, eager attention, no cache | identical forward (`test_mom_block_parity`) | portability; cache is buggy upstream (M6) |
| D-MoM-2 | `qkv_states[idx] += x` | in-place advanced indexing | `index_add` | identical (indices unique per module) | autograd-safe |
| D-MoM-3 | Dev router sigmoid | `F.sigmoid(x, dim=-1)` (TypeError) | `torch.sigmoid(x)` | n/a | upstream bug M7 |
| D-MoM-4 | Transfer | `p.data = state_dict[...]` (shared storage) | clone; returns a report of copied / not copied | identical values | safety |
| D-MoM-5 | Init order | HF `apply` post-order | reversed module list (same post-order property; different RNG order) | same distribution, different draws | no HF dependency |
| D-MoM-6 | `causal: false` | — | bidirectional attention + `<cls>` head | n/a (labelled L3 adaptation) | common interface |
| D-MoM-7 | Task | OpenWebText LM, all-token loss | logic serialised as `<eos> p1 #SEP# p2 #REL# REL`, loss on the relation token only | — | task adaptation |
| D-DATA-1 | Valid split | OM loader every 10th of train0..6 | same, applied after A/B/C filtering | — | not specified for A/B/C upstream |
| D-DATA-2 | A/B/C tests | "examples unseen in training", aggregation unspecified | all matching test0..12 examples, split by length ≤6 / 7–12 | — | review E1 |
| D-FAM-1 | MoA | MoA authors' code | MoMHA with 1 head/expert, untied layers | — | same unit as SUT; record K/V width (review R6) |
| D-FAM-2 | Switch | Switch top-1 router (linear), capacity factor | SUT MoE (MLP router, no capacity dropping), switch loss 0.01 | — | same unit as SUT |

| D-TRAIN-1 | Batching | fairseq batches by length (max-tokens) | fixed batch size with length bucketing (sort within windows of 50 batches, shuffle batch order) | — | 1.6× faster on CPU |
| D-TRAIN-2 | Budget | SUT: 450 epochs; MoM: 10k–20k steps | CPU pilot: 3,000 steps (≈1.6 epochs), see the technical report | — | hardware |
| D-TRAIN-3 | MoM encoder adaptation | AdamW(0.9, 0.95), wd 0.1, warm-up ratio 0.1 | common-panel optimiser (AdamW(0.9, 0.98), wd 0.01, warm-up 300) | — | equal treatment in the controlled panel |

## Open items (not implemented yet)

* Original UT ACT halting (Dehghani et al.) as a separate condition from stick-breaking.
* MoH (shared + routed heads).
* Dynamics transfer task: wire the existing Twilight four-dimensional switching generator into
  `train/data_utils.py` (continuous embedding, causal mask, last-position regression head).
* Frozen/cyclic routing controls (deterministic input-dependent router with matched top-k).
* The GPT-2-style donor (encoder adaptation) did not learn within 3,000 steps, so the MoM two-phase comparison is
  still open. First check: donor with sinusoidal + relative positions (the SUT-unit Dense learned), or a longer donor run.
