"""Parity of each routed module with the released reference code.

Positional encoding is switched off (identity RoPE) because the references use other position
schemes, there is no padding, routing is learned (no warm-up) and dropout is 0. Weights are copied
from the reference module into ours; outputs (and balancing losses where defined identically) must
agree to float32 precision.

* MoE FFN        vs SUT parallel_experts.MoE.forward                    (MLP gate, renormalised top-k)
* MoMHA gate     vs SUT MoE.map / reduce                                (the query/output experts)
* MoA attention  vs MoA moa_layer.MultiheadAttention (self-attention)   (+ switch & z losses)
* MoH attention  vs MoH-DiT MoHAttention                                (routed-only and shared2 forms)
* MoM body       vs pilot v1's MoTBlock port (itself parity-tested against upstream MoM)
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import ref_loader as R  # noqa: E402
from backbone.routing import RoutingState  # noqa: E402

IDENT = lambda x, seq_dim=-2: x  # noqa: E731


def close(a, b, tol=1e-5):
    err = (a - b).abs().max().item()
    assert err < tol, err


def test_moe_ffn():
    from models.moe.moe_ffn import MoEFFN
    pe = R.sut_moe()
    torch.manual_seed(0)
    ref = pe.MoE(32, 48, 6, 2, switchloss=0.01, bias=True, activation=nn.GELU(), mlp_router=True)
    st = RoutingState(balance="lb")
    ours = MoEFFN(32, 48, 6, 2, st, gate="mlp", gate_norm="renorm", bias=True, switchloss=0.01)
    sd = ref.state_dict()
    ours.load_state_dict({k.replace("w_gate.3", "w_gate.2"): v for k, v in sd.items()}, strict=True)
    x = torch.randn(3, 11, 32)
    y_ref, loss_ref = ref(x)
    y, aux = ours(x)
    close(y, y_ref)
    close(aux, loss_ref)


def test_momha_gate_map_reduce():
    from models.momha.momha_attention import SUTGate
    pe = R.sut_moe()
    torch.manual_seed(1)
    ref = pe.MoE(32, 16, 4, 2, bias=False, mlp_router=True)
    ours = SUTGate(32, 16, 4, 2)
    ours.load_state_dict({k.replace("w_gate.3", "w_gate.2"): v for k, v in ref.state_dict().items()}, strict=True)
    x = torch.randn(2, 9, 32)
    yr, _ = ref.map(x)
    y, _, _ = ours.map(x, None, RoutingState())
    close(y, yr)
    h = torch.randn_like(y)
    close(ours.reduce(h), ref.reduce(h))


def test_moa_attention():
    from models.moa.moa_attention import MoAAttention
    M = R.moa()
    torch.manual_seed(2)
    ref = M.MultiheadAttention(32, 3, self_attention=True, num_expert=6, head_dim=8, switchloss=0.01, zloss=0.001)
    ref.eval()
    ours = MoAAttention(32, 8, 6, 3, IDENT, RoutingState(balance="lb"), gate_norm="raw", switchloss=0.01, zloss=0.001)
    sd = {k: v for k, v in ref.state_dict().items()}
    with torch.no_grad():
        nn.init.normal_(sd["q_proj.w_gate"], std=0.5)      # zero-init gate would make the test trivial
    ours.load_state_dict({k: v for k, v in sd.items() if k in dict(ours.named_parameters())}, strict=True)
    x = torch.randn(2, 7, 32)
    xt = x.transpose(0, 1)                                   # fairseq layout T x B x C
    out_ref, _, aux_ref = ref(xt, xt, xt)
    y, aux = ours(x)
    close(y, out_ref.transpose(0, 1))
    close(aux, aux_ref)


@pytest.mark.parametrize("shared,routed", [(0, 4), (2, 2)])
def test_moh_attention(shared, routed):
    from models.moh.moh_attention import MoHAttention
    Ref = R.moh_attention_class()
    torch.manual_seed(3)
    ref = Ref(32, num_heads=6, shared_head=shared, routed_head=routed, head_dim=8)
    ours = MoHAttention(32, 6, 8, IDENT, RoutingState(balance="lb"), shared_head=shared, routed_head=routed, lb_coef=1.0)
    ours.load_state_dict(ref.state_dict(), strict=True)
    x = torch.randn(2, 9, 32)
    Ref.LOAD_BALANCING_LOSSES.clear()
    y_ref = ref(x)
    y, aux = ours(x)
    close(y, y_ref)
    close(aux, Ref.LOAD_BALANCING_LOSSES[-1])


@pytest.mark.parametrize("k", [1, 2])
def test_mom_body(k):
    from models.mom.mom_body import MoMBody
    P = R.pilot_mom()
    cfg = P.MoMConfig(n_embd=32, n_head=2, n_inner=64, num_attn=3, num_ffn=3, depth_per_chunk=2, routing_top_k=k,
                      router_type="naive", router_dim=16, activation_function="gelu", causal=False)
    torch.manual_seed(4)
    ref = P.MoTBlock(cfg, 0)
    ours = MoMBody(32, 2, 16, 64, IDENT, RoutingState(), num_attn=3, num_ffn=3, depth=2, routing_top_k=k,
                   router_type="naive", router_dim=16)
    rs = ref.state_dict()
    m = {}
    for i in range(3):
        m[f"attn_ln.{i}.weight"], m[f"attn_ln.{i}.bias"] = rs[f"mot_attn.attention_layernorm.{i}.weight"], rs[f"mot_attn.attention_layernorm.{i}.bias"]
        m[f"c_attn.{i}.weight"], m[f"c_attn.{i}.bias"] = rs[f"mot_attn.c_attn.{i}.weight"].T, rs[f"mot_attn.c_attn.{i}.bias"]
        m[f"attn_proj.{i}.weight"], m[f"attn_proj.{i}.bias"] = rs[f"mot_attn.c_proj.{i}.weight"].T, rs[f"mot_attn.c_proj.{i}.bias"]
        m[f"ffn_ln.{i}.weight"], m[f"ffn_ln.{i}.bias"] = rs[f"mot_ffn.ffn_layernorm.{i}.weight"], rs[f"mot_ffn.ffn_layernorm.{i}.bias"]
        m[f"c_fc.{i}.weight"], m[f"c_fc.{i}.bias"] = rs[f"mot_ffn.c_fc.{i}.weight"].T, rs[f"mot_ffn.c_fc.{i}.bias"]
        m[f"ffn_proj.{i}.weight"], m[f"ffn_proj.{i}.bias"] = rs[f"mot_ffn.c_proj.{i}.weight"].T, rs[f"mot_ffn.c_proj.{i}.bias"]
    for t in ("attn", "ffn"):
        src = "mot_attn" if t == "attn" else "mot_ffn"
        for j in (0, 2):
            for p in ("weight", "bias"):
                m[f"{t}_router.router.{j}.{p}"] = rs[f"{src}.router.router.{j}.{p}"]
    ours.load_state_dict(m, strict=True)
    x = torch.randn(2, 8, 32)
    y_ref, _, routes = ref(x)
    assert len(set(routes[0][0].flatten().tolist())) > 1        # tokens really take different modules
    y, _ = ours(x, None, None)
    close(y, y_ref, 1e-4)
