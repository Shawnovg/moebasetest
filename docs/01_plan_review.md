# Review of the research plan (v1, 16 Oct 2026)

Reviewed against the released code of SUT (`shawntan/SUT@192cbd3`), MoM
(`gzhch/Mixture-of-Modules@2ed1d28`) and the Ordered-Memory logic data (`yikangshen/Ordered-Memory@2a0d3a2`),
read line by line on 17 Oct 2026. The plan's structure is sound: the separation of reference and controlled
paths, the fidelity levels, validation-only selection, E=1 and fixed-routing controls, and the cost panels all
hold up. The changes below come from facts that are only visible in the code or the data. Each one is already
implemented in this project.

## A. Findings that change the experiment

**R1. SUT's released MIM loss has the opposite sign on the marginal-entropy term.**
`moe.py` lines 169–171 compute `H_e = Σ p_e log p_e` (that is, −H(E)) and return
`miloss = -(neg_H_e_given_x + H_e)` = **H(E|X) + H(E)**. The paper's prose describes maximizing
MI = H(E) − H(E|X). With the shipped coefficient `MILOSS=0.1`, training therefore *penalizes* balanced
usage. In a numerical check, uniform routing scores 2.77, balanced confident routing scores 1.39, and full
collapse scores 0, which is the best value.
*Change:* the reference configs use the released sign (`mim_variant: code`). `sut_common6_mim_paper.yaml`
runs the −MI version as a named ablation, and `tests/test_sut_units.py::test_mim_sign_released_vs_paper`
pins down both behaviors. The plan's comparison "SUT with MIM versus load balancing" therefore needs three
arms: code-MIM, paper-MIM and LB. Report expert-usage entropy for every arm.

**R2. fairseq divides the auxiliary losses by the token count.** In `sut_loss.py` the MoE losses are
added to the *summed* cross-entropy, and the fairseq trainer then scales gradients by 1/`sample_size`. So
the effective MIM weight is 0.1/(target tokens in the batch); for WMT that is about 0.1/8000.
*Change:* `loss.aux_reduction: fairseq` reproduces this and is the default. `mean` is the labelled
alternative. Coefficients from different reductions are not comparable, so record which one was used.

**R3. The planned MoM layout is not legal under the released code.** `GPT2MoTModel` builds
`bottom layers + chunks × (MoTBlock + anchor MoTLayer) + top layers`, and `get_layer_map` uses
`max(N_attn, N_ffn)` donor layers per chunk plus one per anchor. "Six donors → six modules × six steps"
would need eight donor layers (one bottom, six modules, one anchor), or the anchor removed, and it would give
eight block applications, not six.
*Change:* `mom_ref6_causal.yaml` uses **[1-4-1]**: donor layer 0 becomes the bottom layer, donor layers
1–4 become modules 0–3, and donor layer 5 becomes the anchor. Settings are K=1 with SKIP and H=4. That gives
1+4+1 = **6 block applications**, the same execution depth as the donor and the common panel.
`MoMConfig.donor_layers()` and `block_applications()` check this. The paper's MoM-small is
[1-1-4-1-4-1] with 12 donor layers.

**R4. Order-swapped near-duplicates.** No exact pair leaks, but **1,223 validation pairs (9.0%) and
697 test_iid pairs (2.9%)** are (B, A) versions of training pairs (A, B). Their label follows from the
training label: symmetric relations keep it, and < / > swap. These are near-duplicates.
*Change:* each record now carries `swap_dup`, and evaluation also reports `acc_excl_swap_dup`. Length and
composition test sets contain none.

**R5. Learned absolute positions confound length generalization.** Training pairs are at most 69
tokens long, but test_len reaches 117. MoM and its GPT-2 donor use learned `wpe` vectors, so positions past
about 72 are never trained. SUT uses sinusoidal positions plus relative logits clamped to ±63.
SUT-vs-MoM differences on L1 are therefore partly a positional-encoding effect.
*Change:* the controlled panel builds Dense, UT, Switch-style, MoA-style and SUT from the same audited SUT
units, so they share one positional scheme. MoM reference results on test_len must carry this caveat.
Possible follow-up: a MoM variant with sinusoidal positions, labelled as a deviation.

**R6. MoA-style versus grouped experts also changes K/V width.** In SUT's MoMHA only the query and output
projections are experts. K/V are dense and shared, with `expert_dim / head_dim` heads. One head per expert
(MoA) therefore means **one shared K/V head**, which is multi-query attention. Two heads per expert means
two K/V heads. The plan's "128 active head dims on both sides" matches query width but not K/V width.
*Change:* record K/V width with the head-grouping comparison. Add a third arm if you want to separate the
two effects.

**R7. Halting does not remove attention-score work.** The routers and the K/V projections run on every
token, and the `q·k` scores are computed densely for all queries, including halted ones, whose query is zero.
Only expert projections and FFN experts are skipped. `compute.py` charges what is actually executed, so do
not expect latency to follow active-token counts. MoM's Python loop over modules adds dispatch overhead
at this scale.

## B. Questions in the plan answered from code and data

| Plan question | Answer (source) |
|---|---|
| Operator count per clause or per pair? | Per clause: file index = `min(max(ops(lhs), ops(rhs)), 12)`, where ops counts and/or/not (generator `get_len`; verified for every released line) |
| Contradictions and tautologies? | Rejected by the generator (`sat == worlds or empty → continue`); 0 degenerate clauses found |
| Label spellings | `< > = ^ \| # v`, loader order `list('<>=^|#v')`; all 249,238 labels verified against exhaustive truth tables |
| A/B/C exclusions | Serialized-string patterns `( and ( not a ) )`, `( and ( not `, plus `( or ( not `. The filtered training sets are **128,969 / 87,948 / 51,896**, exactly Ordered Memory Table 2 |
| Validation split | Ordered-Memory loader: every 10th example of train0..6 |
| Class balance | `#` is about 54% (majority baseline 54%). The 48–53% "Transformer" row in SUT Table 5 sits at or below the majority rate |
| Halting index and surviving mass at the depth cap | Gate reads the previous raw output; remaining mass goes to the last state; ACT loss counts halting at output j as depth j. See `halting.py` docstring |
| Halted tokens as keys/values | Remain visible through the frozen soft state; queries use the raw state |
| SKIP semantics (MoM) | Extra router class; selecting it runs no module (identity via the residual). The token can compute again at a later step. Its q/k/v are zero for that step, so it is still visible as a zero key |
| GRU state | Per token, reset at each MoTBlock, separate for attention and FFN |
| Module assembly | One attention computation on the **summed** Q/K/V projections of the selected modules (unweighted unless `weighted_qkv`) and the score-weighted sum of their output projections |

## C. Paper/code disagreements and the chosen default

| Item | Paper | Released code | Reference default | Ablation config |
|---|---|---|---|---|
| SUT MIM sign | −MI | H(E\|X)+H(E) | code | `sut_common6_mim_paper` |
| SUT aux scaling | n/a | ÷ sample_size | fairseq | `aux_reduction: mean` |
| MoM balance loss | none ("no positive effect") | coef 0.01 in all `mot` scripts | 0.01 | `mom_ref6_causal_nobal` |
| MoM router output | raw scores (Eq. 8) | `router_type` string; scripts pass temperature 0.2 | `gru` (raw) | set `router_type: gru_softmax` |
| MoM `ln_f` on transfer | "initialize modules from pre-trained" | not in the name map: stays at fresh init | not copied | — |

## D. Reproducibility status of the released code

* **MoM cannot run as released.** `pretrain.sh` calls `pretrain.py`, which is absent. `train.py` line 113
  is a syntax error (`assert ... :`). `model.py` imports `.configuration_gpt_mot`, which does not exist (the
  file is `config.py`). The `dev` router's sigmoid branch calls `F.sigmoid(x, dim=-1)`, which raises a
  TypeError. Level 1 (running the reference) is therefore impossible without patches. This project runs
  Level 2 instead: `tests/test_parity_upstream.py` imports the upstream classes, aliasing the missing
  module, and compares outputs and gradients.
* **SUT needs fairseq.** Two of the three scripts reference `moa_transformer_base` / `moa_ce_loss`, which
  are not registered in the repository. The parity tests load the *upstream* layer files with minimal
  stand-ins for the fairseq utilities they import, so the comparison is against the real SUT code.
* **No LICENSE file** in either SUT or MoM. Nothing is vendored: `scripts/fetch_upstream.py` clones the
  pinned commits into `third_party/`.

## E. Smaller improvements

1. *Controlled C2 and intersection sets:* define them as the matched composition test examples split by
   length (≤6 versus 7–12) instead of generating new data. This needs no new generator and adds no leakage
   risk (`test_comp_seenlen`, `test_comp_newlen`).
2. *SUT reference geometry:* use the paper's logic-task experts (attention E=12, k=4, 2 heads × 32;
   FFD E=12, k=4, D=128), scaled to width 128 and 6+6 steps. The Table 7 learning rate is illegible, so tune
   it on validation.
3. *MoM router width:* the upstream default `router_dim` of 256 is larger than the 128 model width. It is
   kept for the reference path. Screen 64 as a labelled variant.
4. *MoM budget:* the phase-2 warm-up is 10% of phase-2 steps (paper). The scratch control gets donor plus
   phase-2 steps.
5. *Not yet implemented:* the original UT ACT halting rule and MoH. Both are listed as open items in the
   deviation log.
