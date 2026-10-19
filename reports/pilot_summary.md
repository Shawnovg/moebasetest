# CPU pilot results (18 Oct 2026)

Full report: https://claude.ai/code/artifact/b9589783-f2e8-42cb-bb51-c54721ee6ed7
Program: D:\HarunoKen\MoE_Base_Test\notebooks\MoE_Experiment.ipynb (executed copy: MoE_Experiment_executed.ipynb)

Setup: logic inference (Ordered-Memory release), width 128, 6 block applications, 3,000 steps, batch 64,
AdamW lr 1e-3, 2 seeds (1 for extras), CPU (2 cores). 66 tests pass incl. 10 parity tests vs upstream SUT/MoM.

| Model | Valid | Test 0-6 | Test 7-12 |
|---|---|---|---|
| Dense | 0.747 | 0.736 | 0.571 |
| Switch | 0.690 | 0.682 | 0.543 |
| MoA | 0.632 | 0.621 | 0.512 |
| UT fixed | 0.609 | 0.602 | 0.518 |
| SUT fixed-depth | 0.609 | 0.594 | 0.510 |
| SUT (released MIM) | 0.557 | 0.551 | 0.511 |
| SUT (paper MIM) | 0.586 | 0.574 | 0.512 |
| SUT 3x budget (9k steps) | 0.709 | 0.691 | 0.533 |
| MoM / donor (all arms) | 0.55-0.57 | ~0.54 | ~0.51 |
| Majority class | 0.538 | 0.541 | 0.512 |

Findings: released SUT MIM sign collapses routing (entropy 0.50-0.76 vs 0.93-0.97 with paper sign);
load balancing makes SUT halt immediately (depth 0.19); GPT-2-style MoM donor did not learn in 3,000 steps,
so the MoM comparison is open; Dense shows a 10-point composition gap on split B at seen lengths.
Next: train to convergence on GPU, paper MIM sign default, fix donor (positional scheme) and gate phase 2.
