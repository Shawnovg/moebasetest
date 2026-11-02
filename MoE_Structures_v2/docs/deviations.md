# Deviations from the reference code, and things found in it

Every structure is ported from its released code (commits in `references/README.md`). This file
lists each place where this study differs from that code, and why, plus oddities found in the
reference code itself. Parity tests (`tests/test_parity.py`, `tests/test_mita.py`) check that, with
positions switched off, each module computes the same output as the reference.

## Changes common to all structures

| # | Change | Reason |
|---|---|---|
| C1 | One position scheme for all: RoPE on q and k in every attention (SUT/MoA: learned relative logits or fairseq sinusoidal; MoH/MiTA: ViT absolute embeddings; MoM: GPT-2 learned positions). | Position handling must not differ between the compared structures. |
| C2 | Bidirectional encoder with answer slots (`<ans>` tokens) instead of causal LM / classification. | MiTA is defined without a causal mask (its landmarks pool over the whole sequence); one interface for all. |
| C3 | Padding mask added to every attention and to every router statistic (balancing losses count real tokens only, except where the reference itself counts padding — see MoE/MoMHA). | Variable-length batches. |
| C4 | No dropout, no auxiliary losses unless `routing.balance = lb`. | Supervisor: structure-only comparison, no extra loss by default. |
| C5 | Random warm-up routing (`balance = warmup`): during the first 10 % of steps each routed module picks uniformly random units; the gate values are still the router's probabilities at those units so the router keeps learning. | Requested by the user as the alternative to a balancing loss. Not in any reference. |
| C6 | 2 layers, d_model 128, 4 heads x 32 everywhere; reference sizes are much larger. | Small synthetic-task study on CPU. |

## Per structure

### MoE (SUT `parallel_experts/moe.py`)
* Kept: MLP router Linear-Tanh-Linear, softmax, top-k, renormalised gates `g / (sum g + 1e-6)`, ParallelExperts init, Switch loss on all tokens with padded rows' probabilities zeroed (as the reference: padded tokens still count in `expert_size`).
* Variant `top1` uses a linear router and the raw probability as the gate (Switch Transformer convention). **Finding F1:** with SUT's renormalisation and k = 1 every gate equals 1, so the router receives no gradient from the task loss; SUT itself never uses k = 1.
* Variant `hash` (Roller et al. 2021) is not from SUT: fixed random token-id -> 2 experts, gate 1/2, no router, so only `balance = none` is run.

### MoMHA (SUT `moa_attention.py` MultiheadAttention)
* Kept: expert = query projection + output projection of `heads_per_expert` heads; K/V shared; MLP router, renormalised gates, Switch loss.
* Not kept: SUT's learned relative-position bias (C1) and its MI loss (C4).
* Variant `top1`: raw gate (F1).

### MoA (MoA `moa_layer/moa_attention.py`, `parallel_linear/moe.py`)
* Kept: zero-initialised `w_gate` parameter (d x E), single shared K/V head, per-expert W_q / W_o, Switch 0.01 + z-loss 0.001 (run.sh).
* **Finding F2:** the paper describes gates renormalised over the top-k; the released code uses the raw top-k softmax probabilities (no renormalisation). The default follows the code; variant `renorm_detached` follows the paper text.
* **Finding F3:** with a zero-initialised gate, all routing probabilities are exactly uniform at step 0 and top-k is decided by index order (ties) — every token starts with the same experts.
* Noisy gating off (MultiheadAttention default).

### MoH (MoH `MoH-DiT/models.py` MoHAttention)
* Kept: all heads computed, router `wg` softmax, top-k mask, renormalised x routed_head, shared heads with `wg_1` and two-stage `wg_0` (variant `shared2`), the load-balance loss `mean(me*ce)*E^2` averaged over layers (train.py), computed only in training.
* **Finding F4:** with `shared_head == 1` the reference creates no `wg_1` but uses it (AttributeError); we assert `shared_head != 1`.
* Default here is routed-only (0 shared, 4 of 8) so that MoH and MoA select from the same number of units; the paper's main configuration (shared + routed) is variant `shared2`.
* MoH uses 8 heads x 32 = 2x the attention parameters of Dense, with 4 heads active per token.

### MoM (Mixture-of-Modules `mom/model.py`, `mom/utils.py`)
* Kept: per-module LayerNorm, summed qkv of chosen modules (scores not applied to qkv), raw-score weighting of attention output and FFN output, SKIP class, separate attention / FFN routers shared over steps, GPT-2 init, `load_balancing_loss_func` (top-1 Switch form, softmax over the raw scores, all steps concatenated per type, weight 0.01).
* Changed: trained from scratch, no donor initialisation, no bottom / anchor / top dense layers (supervisor: same depth and structure-only comparison) — the reference builds MoM from a pretrained dense model, so this is the largest change. Default router is the 'naive' MLP (TokenRouter); the paper's GRU router is variant `gru`.
* **Finding F5:** the router's raw score multiplies the module output. Scores start near 0 (init std 0.02) and can be negative, so at the start every module contributes ~2-5 % of its output and the sign can flip. In the reference this is cushioned by the donor initialisation; from scratch it slows learning and makes collapse to one module (often SKIP) likely. See the technical report.
* **Finding F6:** a token that picks SKIP for attention still appears to other tokens as a zero key / zero value (logit 0), i.e. it still receives attention weight.
* **Finding F7 (pilot v1):** `load_balancing_loss_func` applies softmax to scores that are already probabilities for 'softmax' routers (double softmax); not relevant to the 'naive' default.

### MiTA (MiTA `mita/mita_attention.py`, `_mita_attention.py`, `mixed_attention.py`)
* Kept: landmarks = average-pooled queries; each landmark's top-k keys form one expert; compressed shared expert (landmark keys, `agent_value = SDPA(router, K, V)` values); query routed by argmax `router . q` (top-s for s > 1); one joint softmax over shared + routed entries (the reference merges two FlashAttention calls by log-sum-exp, which is the same); qkv bias False, proj bias True.
* Changed: dense PyTorch implementation instead of FlashAttention varlen kernels (same numbers, tested against a per-query loop); 1-D windows over each sequence's real length instead of AdaptiveAvgPool2d over an image grid; m = k = 8 instead of 25 (sequences are 20-70 tokens, not 196 patches).
* **Finding F8:** the shared keys are passed as `router * scale` and FlashAttention scales again, so shared logits are `q . r * d^-1` while routed logits are `q . k * d^-1/2`. Kept as in the code.
* **Finding F9:** MiTA has no learned router and no balancing loss: the expert choice is a parameter-free argmax. For `balance = lb` we add a Switch-style loss on softmax over landmark scores (our adaptation; gradients reach the queries). `warmup` sends queries to random landmarks.
* With s = 2 a key chosen by both experts appears twice in the softmax, as in the reference.
* The reference ViT drops the CLS token from pooling; there is no CLS here.

## Findings from the experiments (see the technical report for numbers and figures)

| # | Finding | Where it shows |
|---|---|---|
| E1 | **MoM trained from scratch collapses to SKIP.** Without balancing, every token picks SKIP for attention in both assembly steps (entropy 0) and mostly SKIP for the FFN, so the model is close to an embedding-only classifier (fact 0.009, mod 0.014). The load-balance loss spreads the picks (entropy 0.85-0.96) but accuracy does not recover (0.013), because of F5: outputs are scaled by raw router scores that start near 0. `noskip`, `gru` do not fix it; `k2` trains on some seeds only (0.01-0.34). This confirms that the reference relies on donor initialisation. | `routing_fact_mom_*.png`, variants_fact_mom.png |
| E2 | **MoE without balancing collapses** to 2 dominant experts per layer (min entropy 0.46; the top 2 of 8 take 75-90 % of the picks). The SUT Switch loss keeps routing uniform and is the only balancing that helps clearly (+0.21 on fact). Random warm-up routing only delays the collapse: after the warm-up the router returns to a few experts (entropy 0.56). | collapse_fact.png, balance_delta_fact.png |
| E3 | **MiTA is balanced by construction**: landmarks are pooled from consecutive positions, so the argmax over landmarks spreads queries evenly (entropy ~1.0 under all three modes). Both balancing methods change nothing. | collapse_fact.png |
| E4 | **MiTA's top-k key selection loses information** on position-exact tasks: it is the only structure below ceiling on copy (0.965) and rev (0.952). Accuracy rises strongly with the number of keys a query can see (m4k4 0.06-0.08, default m8k8 0.21, m16k16 0.44-0.46 on fact) and the compressed shared expert is essential (noshared 0.09). | screening, variants_fact_mita.png |
| E5 | **The fact task measures active capacity.** Dense-wide (2048-wide FFN, all active) reaches 0.985 while MoE with the same total FFN parameters but 512 active reaches 0.44-0.65; variants that activate more units win (MoA k8 0.79, MoH k6 0.67). Accuracy is still rising at step 3000 for every structure except Dense-wide, so fact also measures learning speed under the fixed budget. | main_fact.png, curves_fact_*.png |
| E6 | **mod (held-out pairs) depends on the step budget**: MoMHA with warm-up reaches 0.24-0.53 valid accuracy with the loss still falling at step 3000 (delayed, not broken). Seen-pair accuracy is 0.84-0.97 for every routed structure except MoM. | curves_mod_*.png |
| E7 | **MoA has a high seed variance** (fact: sd 0.10-0.12; mod lb: test 0.01-0.30 across seeds). Consistent with F3 (zero-initialised gate, ties broken by index at step 0). The paper-text gate normalisation (`renorm_detached`) is worse than the released code (0.17-0.21 vs 0.35-0.46). | main_fact.png, variants_fact_moa.png |
| E8 | **Single-expert top-1 variants behave differently by module type.** MoE top1 with a raw gate (Switch form) beats the default k=2 without balancing (0.58 vs 0.44), while MoMHA top1 (one 4-head expert) drops to 0.05-0.14. | variants_fact_moe.png, variants_fact_momha.png |
| E9 | **Dataset calibration.** copy / rev are at ceiling, mqar at floor (~0.12 for all, 2 layers cannot do multi-query recall in 3000 steps), logic at the majority class (0.55-0.58), and the 6-task mix only learns copy and rev (each task gets 1/6 of the data). Only fact and mod separate the structures; they were used for the main and variant sweeps. Confirmed with a second screening seed (seeds 0-1 on all 7 datasets); the only seed disagreement is MoM on rev (1.00 on seed 0, 0.02 on seed 1: the loss never leaves chance, ln 64 = 4.16), which shows MoM's collapse is decided early and depends on the seed. Dense-wide on mix is also unstable (0.26 / 0.12, slow to learn copy and rev with 1/6 of the data). | screening_heatmap.png, screening_separation.png |
| E10 | MoH `shared2` with the paper's two-stage weights is worse than routed-only (0.30-0.37 vs 0.43-0.52). The usage vector lists the 2 shared heads as never-routed units; the analysis excludes them (`analysis/collect.py`). | variants_fact_moh.png |
