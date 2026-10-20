# Source audit: paper → upstream code → local module

Pinned commits (also in `src/moebase/paths.py`):

| Repo | Commit | Licence |
|---|---|---|
| shawntan/SUT | `192cbd3ed567e45e6ec6d33eb1ea9c3b331bc2b0` (2023-10-23) | no LICENSE file; files carry the fairseq MIT header |
| gzhch/Mixture-of-Modules | `2ed1d28c320c0a8d114b203b2c68675ffa489d45` (2024-10-03) | no LICENSE file; HF Apache-2.0 headers |
| yikangshen/Ordered-Memory | `2a0d3a22fb70216993a7faf25de3515f42e10431` (2019-11-20) | MIT |

## SUT

| Mechanism | Upstream | Local | Parity test |
|---|---|---|---|
| Expert linear maps | `sut_layer/parallel_linear/parallel_experts/parallel_experts.py` (ParallelLinear, ParallelExperts) | `models/sut/moe.py::ParallelExperts` | via `test_moe_ffn_parity` grads |
| Top-k gating, dispatch, aux losses | `.../moe.py::MoE` (compute_gating, top_k_gating, map/reduce, get_aux_loss_and_clear) | `models/sut/moe.py::MoE` | `test_moe_ffn_parity` |
| MoMHA (expert q/o, shared K/V, rel-pos) | `sut_layer/moa_attention.py::MultiheadAttention` | `models/sut/attention.py::MoMHA` | `test_attention_parity[self/cross]` |
| Encoder/decoder block (post-LN) | `sut_layer/sut_transformer_layer.py` | `models/sut/layers.py` | `test_encoder_layer_parity`, `test_decoder_layer_parity` |
| Stick-breaking halting | `halting.py::ACTWrapper` (StickBreakingACT unused) | `models/sut/halting.py` | `test_halting_parity` |
| Universal encoder/decoder loop | `transformer_encoder.py` L240–300, `transformer_decoder.py` L345–400 | `models/sut/model.py` | behavioural tests |
| Criterion | `sut_loss.py` (label smoothing, `+ total_aux_loss + Σ get_aux_loss_and_clear`) | `train/loop.py::forward_loss` | — |
| Embeddings | fairseq Embedding N(0,d^-½), sinusoidal positions, scale √d | `models/sut/model.py` | — |

### Findings (S = SUT)

* **S1** Only q and o are experts; K/V are dense with `expert_dim` width (`moa_attention.py` L78–90).
* **S2** Encoder FFN-expert activation = gelu → *main* dropout (L75–77); decoder uses activation dropout (L316).
* **S3** MIM sign: `moe.py` L169–171 and L199–201 compute H(E|X)+H(E) (see review R1).
* **S4** Aux losses accumulate over all steps and both MoE pools (`acc_aux_loss=True`) and are cleared only
  in the training criterion. Validation passes therefore grow the statistic lists (fixed locally, D-SUT-3).
* **S5** Effective aux weight = coefficient / sample_size (fairseq gradient normalisation; review R2).
* **S6** `params_de_base.sh` / `params_de_big.sh` use `ARCHS=moa_transformer_base`, `LOSS=moa_ce_loss`,
  which are not registered in the repo; only `params_de.sh` (`sut_transformer_base`, `sut_ce_loss`) matches.
* **S7** Dead/buggy diagnostics: `transformer_encoder.py` L286 reads `moe.top_k_indices` (never set);
  `transformer_decoder.py` L389 `total += total + ...` (harmless, starts at 0); `halting.py __main__`
  calls a non-existent `layer.init`.
* **S8** `ParallelLinear` forces FP16 under CUDA autocast (`custom_fwd(cast_inputs=torch.float16)`).
* **S9** `ACT_THRESHOLD` is read from an environment variable at import (default 0.999).
* **S10** Relative-position window: 64 learned offsets per head, offsets clamped to ±63; no locality mask.

## MoM

| Mechanism | Upstream | Local | Parity test |
|---|---|---|---|
| Routers (naive MLP, GRU, GRU-dev), STE | `mom/utils.py` | `models/mom/routers.py` | inside `test_mom_block_parity` |
| Attention assembly | `mom/model.py::MoTAttention` L84–435 | `models/mom/model.py::MoTAttention` | `test_mom_block_parity` |
| FFN assembly | `MoTMLP` L437–548 | `MoTMLP` | idem |
| Assembly loop, GRU state reset | `MoTBlock` L551–613 (state reset L582) | `MoTBlock` | idem + `test_block_is_deterministic...` |
| Anchor / plain layers | `MoTLayer` L616–679 | `MoTLayer` | `test_hand_route_equals_repeated_donor_layer` |
| Layout | `GPT2MoTModel.__init__` | `GPT2MoTModel` | `test_layout_counts` |
| Loss | `GPT2MoTLMHeadModel.forward` + `load_balancing_loss_func` (top_k=1 hard-coded) | `MoMCausalLM`, `mom_aux_loss` | `test_lm_loss_only_on_target_and_aux_present` |
| Two-phase transfer | `mom/utils.py::get_layer_map/get_name_map/init_from_vanilla` | `models/mom/convert.py` | `test_transfer_copies_donor_tensors_exactly` |
| Training objective | `train.py::Trainer.compute_loss` = `loss + coef * aux` | `train/loop.py` | — |
| Init | HF `_init_weights` (N(0,0.02); direct `c_proj.weight` scaled by 1/√(2·n_layer)) | `gpt2_init_weights` | `test_gpt2_init_scales_only_direct_c_proj` |

### Findings (M = MoM)

* **M1** `pretrain.sh` calls `pretrain.py`, not in the repository.
* **M2** `model.py` L59 imports `.configuration_gpt_mot`; the shipped file is `config.py`.
* **M3** `train.py` L113 `assert 'mot' in mot_args.method:` is a syntax error; `train.py` also never copies
  the layout fields (chunks, num_attn, …) from `MoTArguments` into the config (they come from a
  structure JSON that is not released).
* **M4** `ln_f` is not in the transfer map; the final LayerNorm keeps its fresh initialisation.
* **M5** `load_balancing_loss_func` softmaxes router scores; for `*softmax*` routers this is a double softmax.
* **M6** The KV cache in `MoTBlock` reuses one `layer_past` for all depths (incorrect for generation;
  unused in training). Not ported.
* **M7** `GRUdevTokenRouter` sigmoid branch calls `F.sigmoid(x, dim=-1)` (TypeError). Local port applies
  `torch.sigmoid`.
* **M8** Per-module c_proj weights (`c_proj.{m}.weight`) do not receive GPT-2's scaled init; only anchors do
  (irrelevant after transfer, relevant for from-scratch MoM).
* **M9** No dropout on the routed FFN path; dropout only on the anchor FFN and attention output.
* **M10** Load-balancing loss is applied with `router_aux_loss_coef=0.01` in all `mot*` script branches,
  contrary to the paper's "no balance loss".

## Logical inference data (Ordered Memory)

Verified by `scripts/prepare_logic_data.py` (written to the manifest) and `tests/test_data.py`:
all 249,069 labels match exhaustive truth tables over 2^6 worlds; 0 degenerate clauses; 0 file-index
mismatches; A/B/C training sizes equal Ordered Memory Table 2; no exact train/test pair overlap; order-swapped
near-duplicates flagged per record (`swap_dup`).
