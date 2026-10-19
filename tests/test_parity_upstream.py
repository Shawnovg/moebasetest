"""Level-2 parity: our ports vs the pinned upstream code (third_party/, see scripts/fetch_upstream.py).

Weights are copied with ``load_state_dict(strict=True)`` (so parameter names must match exactly),
dropout is off (eval mode), and forward outputs, auxiliary losses, selected experts and parameter
gradients are compared in FP32.
"""
import pytest

torch = pytest.importorskip("torch")

import upstream_loader as UL  # noqa: E402
from moebase.models.sut.attention import MoMHA  # noqa: E402
from moebase.models.sut.halting import ACTWrapper  # noqa: E402
from moebase.models.sut.layers import SUTBlockConfig, SUTDecoderLayer, SUTEncoderLayer  # noqa: E402
from moebase.models.sut.moe import MoE  # noqa: E402

pytestmark = pytest.mark.parity
TOL = dict(rtol=1e-5, atol=1e-6)


@pytest.fixture(scope="module")
def up():
    if not UL.sut_available():
        pytest.skip("third_party/SUT missing: run scripts/fetch_upstream.py")
    try:
        return UL.load_sut()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"could not import upstream SUT with stubs: {e!r}")


def _grads(m):
    return {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}


def _compare_grads(a, b):
    ga, gb = _grads(a), _grads(b)
    assert ga.keys() == gb.keys()
    for n in ga:
        torch.testing.assert_close(ga[n], gb[n], **TOL, msg=n)


def test_moe_ffn_parity(up):
    import torch.nn.functional as F
    kw = dict(cvloss=0.01, switchloss=0.01, zloss=0.001, miloss=0.1, bias=True, activation=F.gelu, acc_aux_loss=True)
    ref = up.MoE(16, 24, 6, 2, **kw).eval()
    ours = MoE(16, 24, 6, 2, **kw).eval()
    ours.load_state_dict(ref.state_dict(), strict=True)
    x = torch.randn(3, 7, 16)
    mask = torch.ones(3, 7); mask[1, 4:] = 0
    y_r, _ = ref(x, skip_mask=mask)
    y_o, _ = ours(x, skip_mask=mask)
    torch.testing.assert_close(y_o, y_r, **TOL)
    assert torch.equal(ref.expert_size, ours.expert_size)
    l_r, l_o = ref.get_aux_loss_and_clear(), ours.get_aux_loss_and_clear()
    torch.testing.assert_close(l_o, l_r, **TOL)
    (y_r.sum() + l_r).backward(); (y_o.sum() + l_o).backward()
    _compare_grads(ours, ref)


@pytest.mark.parametrize("self_attention", [True, False])
def test_attention_parity(up, self_attention):
    kw = dict(num_expert=6, top_k=2, expert_dim=16, head_dim=8, miloss=0.1)
    if self_attention:
        ref = up.MultiheadAttention(32, self_attention=True, **kw).eval()
        ours = MoMHA(32, self_attention=True, **kw).eval()
    else:
        ref = up.MultiheadAttention(32, kdim=32, vdim=32, encoder_decoder_attention=True, **kw).eval()
        ours = MoMHA(32, kdim=32, vdim=32, encoder_decoder_attention=True, **kw).eval()
    with torch.no_grad():
        if self_attention:
            ref.rel_pos_emb.normal_(0, 0.1)
    ours.load_state_dict(ref.state_dict(), strict=True)
    T, S, B = 6, 6 if self_attention else 9, 2
    q = torch.randn(T, B, 32)
    kv = q if self_attention else torch.randn(S, B, 32)
    kpm = torch.zeros(B, S, dtype=torch.bool); kpm[1, -2:] = True
    skip = torch.ones(T, B); skip[-1, 0] = 0
    a_r, _, _ = ref(q, kv, kv, 0, key_padding_mask=kpm, skip_mask=skip)
    a_o, _, _ = ours(q, kv, kv, key_padding_mask=kpm, skip_mask=skip)
    torch.testing.assert_close(a_o, a_r, **TOL)
    a_r.pow(2).sum().backward(); a_o.pow(2).sum().backward()
    _compare_grads(ours, ref)


def _block_cfg():
    return SUTBlockConfig(embed_dim=32, attn_num_expert=6, attn_k=2, attn_expert_dim=16, head_dim=8,
                          ff_num_expert=5, ff_k=2, ff_expert_dim=24, dropout=0.0, miloss=0.1, switchloss=0.01)


def test_encoder_layer_parity(up):
    b = _block_cfg()
    ref = up.EncoderLayer(UL.sut_cfg_namespace(b)).eval()
    ours = SUTEncoderLayer(b).eval()
    with torch.no_grad():
        ref.self_attn.rel_pos_emb.normal_(0, 0.1)
    ours.load_state_dict(ref.state_dict(), strict=True)
    T, B = 7, 3
    x = torch.randn(T, B, 32)
    soft = torch.randn(T, B, 32)
    pad = torch.zeros(B, T, dtype=torch.bool); pad[2, -3:] = True
    halt = ((1 - pad.t().float()) * torch.tensor([1, 1, 0.3, 0.0, 1, 1, 1.0])[:, None]).contiguous()  # upstream requires contiguous
    y_r = ref(x, soft, halt, 0, pad)[0]
    y_o = ours(x, soft, halt, 0, pad)[0]
    torch.testing.assert_close(y_o, y_r, **TOL)
    aux_r = sum(m.get_aux_loss_and_clear() for m in ref.modules() if isinstance(m, up.MoE))
    aux_o = sum(m.get_aux_loss_and_clear() for m in ours.modules() if isinstance(m, MoE))
    torch.testing.assert_close(aux_o, aux_r, **TOL)
    (y_r.sum() + aux_r).backward(); (y_o.sum() + aux_o).backward()
    _compare_grads(ours, ref)


def test_decoder_layer_parity(up):
    b = _block_cfg()
    ref = up.DecoderLayer(UL.sut_cfg_namespace(b)).eval()
    ours = SUTDecoderLayer(b).eval()
    with torch.no_grad():
        ref.self_attn.rel_pos_emb.normal_(0, 0.1)
    ours.load_state_dict(ref.state_dict(), strict=True)
    T, S, B = 4, 6, 2
    x = torch.randn(T, B, 32)
    enc = torch.randn(S, B, 32)
    epad = torch.zeros(B, S, dtype=torch.bool); epad[0, -1] = True
    fut = torch.triu(torch.full((T, T), float("-inf")), 1)
    halt = torch.ones(T, B)
    y_r = ref(x, None, halt, 0, enc, epad, self_attn_mask=fut)[0]
    y_o = ours(x, None, halt, 0, encoder_out=enc, encoder_padding_mask=epad, self_attn_mask=fut)[0]
    torch.testing.assert_close(y_o, y_r, **TOL)


class _Mod(torch.nn.Module):
    embed_dim = 12

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(12, 12)

    def forward(self, x, soft, mask, layer_idx, *args, **kw):
        return (torch.tanh(self.lin(x)) * mask[..., None] + x * (1 - mask[..., None]),)


def test_halting_parity(up):
    mod = _Mod()
    ref = up.ACTWrapper(mod, threshold=0.9).eval()
    ours = ACTWrapper(mod, 12, threshold=0.9).eval()
    with torch.no_grad():
        for p in ref._gate.parameters():
            p.normal_(0, 0.5)
    ours.load_state_dict(ref.state_dict(), strict=True)
    T, B = 5, 3
    pad = torch.ones(T, B); pad[-1, 1] = 0
    h_r = h_o = torch.randn(T, B, 12)
    st_r = st_o = s_r = s_o = None
    for i in range(6):
        st_r, out_r, s_r, l_r = ref(st_r, h_r, s_r, pad, i)
        st_o, out_o, s_o, l_o, _ = ours(st_o, h_o, s_o, pad, i)
        h_r, h_o = out_r[0], out_o[0]
        torch.testing.assert_close(s_o, s_r, **TOL)
        torch.testing.assert_close(torch.as_tensor(l_o), torch.as_tensor(l_r), **TOL)


# ------------------------------------------------------------------------------------- MoM
@pytest.fixture(scope="module")
def upmom():
    if not UL.mom_available():
        pytest.skip("third_party/Mixture-of-Modules missing: run scripts/fetch_upstream.py")
    try:
        return UL.load_mom()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"could not import upstream MoM (needs a compatible `transformers`): {e!r}")


def _mom_cfgs(up, **kw):
    from moebase.models.mom.model import MoMConfig
    base = dict(vocab_size=40, n_positions=32, n_embd=32, n_layer=6, n_head=4, n_inner=64,
                resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, chunks=1, num_bottom_layers=1,
                num_top_layers=0, num_attn=4, num_ffn=4, depth_per_chunk=3, router_type="gru",
                router_dim=16, routing_top_k=2, enable_pause=True)
    base.update(kw)
    ours = MoMConfig(**base)
    ref = up.GPT2Config(**{k: v for k, v in base.items()})
    return ours, ref


@pytest.mark.parametrize("router_type,k", [("gru", 1), ("gru", 2), ("gru_softmax", 2), ("naive", 1)])
def test_mom_block_parity(upmom, router_type, k):
    from moebase.models.mom.model import MoTBlock
    oc, rc = _mom_cfgs(upmom, router_type=router_type, routing_top_k=k)
    ref = upmom.model.MoTBlock(rc, 0).eval()
    ours = MoTBlock(oc, 0).eval()
    ours.load_state_dict(ref.state_dict(), strict=True)
    x = torch.randn(2, 9, 32)
    y_r = ref(x, output_router_logits=True)
    y_o = ours(x, output_router_logits=True)
    torch.testing.assert_close(y_o[0], y_r[0], **TOL)
    for (ra, rf), (oa, of) in zip(y_r[2], y_o[1]):
        torch.testing.assert_close(oa, ra, **TOL)
        torch.testing.assert_close(of, rf, **TOL)
    y_r[0].pow(2).sum().backward(); y_o[0].pow(2).sum().backward()
    _compare_grads(ours, ref)
