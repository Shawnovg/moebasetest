# Reference code

Each structure is ported from the released code at these commits. Clone them next to the project
(or set `MOEV2_REF1` / `MOEV2_REF2`) to run `tests/test_parity.py`.

| Structure | Paper | Repository | Commit | Files ported |
|---|---|---|---|---|
| MoE, MoMHA | Tan et al. 2023, *Sparse Universal Transformer* | https://github.com/shawntan/SUT | 192cbd3 | `sut_layer/parallel_linear/parallel_experts/moe.py`, `sut_layer/moa_attention.py` |
| MoA | Zhang et al. 2022, *Mixture of Attention Heads* | https://github.com/yikangshen/MoA | 96c33ca | `moa_layer/moa_attention.py`, `moa_layer/parallel_linear/moe.py`, `run.sh` |
| MoH | Jin et al. 2024, *MoH: Multi-Head Attention as Mixture-of-Head Attention* | https://github.com/SkyworkAI/MoH | 2cefd57 | `MoH-DiT/models.py` (MoHAttention), `MoH-DiT/train.py` (loss averaging) |
| MoM | Gong et al. 2024, *Mixture-of-Modules* | https://github.com/gzhch/Mixture-of-Modules | 2ed1d28 | `mom/model.py`, `mom/utils.py` |
| MiTA | *Mixture-of-Top-k Attention: Efficient Attention via Scalable Fast Weights* | https://github.com/QishuaiWen/MiTA | 95460f7 | `mita/mita_attention.py`, `mita/_mita_attention.py`, `mita/mixed_attention.py` |
| logic data | Bowman et al. 2015 (via Ordered-Memory) | https://github.com/yikangshen/Ordered-Memory | 2a0d3a2 | `data/propositionallogic` (pilot v1 `logic-v1`, 'length' split) |

Expected layout for the tests: `$MOEV2_REF1/{SUT,Mixture-of-Modules}`, `$MOEV2_REF2/{MoA,MoH,MiTA}`.
The MoM parity test compares against pilot v1's MoM port (itself parity-tested against the
upstream code) and needs `MOEV2_PILOT` pointing at the MoE_Base_Test project.
