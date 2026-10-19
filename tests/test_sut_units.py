"""Behavioural tests of the SUT port (no upstream code needed)."""
import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from moebase.models.sut.attention import MoMHA  # noqa: E402
from moebase.models.sut.halting import ACTWrapper  # noqa: E402
from moebase.models.sut.layers import SUTBlockConfig  # noqa: E402
from moebase.models.sut.model import SUTClassifier, SUTConfig, SUTSeq2Seq, collect_moe_aux_loss  # noqa: E402
from moebase.models.sut.moe import MoE  # noqa: E402

GATE_ONE = 1.0 / (1.0 + 1e-6)  # upstream renormalises top-k gates with +1e-6


def test_single_expert_ffn_equals_dense_mlp():
    moe = MoE(16, 32, 1, 1, bias=True, activation=F.gelu).eval()
    x = torch.randn(3, 5, 16)
    y, _ = moe(x)
    W1, b1 = moe.experts.w[0], moe.experts.b[0]
    W2, b2 = moe.output_experts.w[0], moe.output_experts.b[0]
    ref = (F.gelu(x @ W1 + b1) @ W2 + b2) * GATE_ONE
    torch.testing.assert_close(y, ref, rtol=1e-5, atol=1e-6)


def test_skipped_tokens_are_not_dispatched_and_output_zero():
    moe = MoE(8, 16, 4, 2, bias=True, activation=F.relu).eval()
    x = torch.randn(2, 6, 8)
    mask = torch.ones(2, 6)
    mask[0, 3:] = 0
    y, _ = moe(x, skip_mask=mask)
    assert torch.all(y[0, 3:] == 0)
    assert moe.expert_size.sum().item() == int(mask.sum().item()) * 2  # k=2 slots per active token


def test_topk_gates_sum_to_one_and_are_renormalised():
    moe = MoE(8, 8, 6, 3, activation=F.relu).eval()
    moe(torch.randn(4, 5, 8))
    g = moe.gates  # dense (tokens, E)
    torch.testing.assert_close(g.sum(-1), torch.full((20,), 1 / (1 + 1e-6)), rtol=1e-5, atol=1e-6)
    assert torch.all((g > 0).sum(-1) == 3)


def _mi_for(prob):
    m = MoE(4, 4, prob.size(1), 1, miloss=1.0)
    logits = torch.log(prob)
    return {v: (setattr(m, "mim_variant", v), m.compute_miloss(logits, prob, None))[1].item() for v in ("code", "paper")}


def test_mim_sign_released_vs_paper():
    E, N = 4, 400
    uniform = torch.full((N, E), 1 / E)
    collapsed = torch.full((N, E), 1e-9); collapsed[:, 0] = 1
    collapsed = collapsed / collapsed.sum(-1, keepdim=True)
    idx = torch.arange(N) % E
    balanced = torch.full((N, E), 1e-9); balanced[torch.arange(N), idx] = 1
    balanced = balanced / balanced.sum(-1, keepdim=True)
    u, c, b = _mi_for(uniform), _mi_for(collapsed), _mi_for(balanced)
    # released code: H(E|X) + H(E)  -> collapse is the minimum
    assert u["code"] == pytest.approx(2 * math.log(E), rel=1e-4)
    assert c["code"] < b["code"] < u["code"]
    # paper prose: -MI -> balanced confident routing is the minimum (= -log E)
    assert b["paper"] == pytest.approx(-math.log(E), abs=1e-3)
    assert b["paper"] < u["paper"] and b["paper"] < c["paper"]


def test_momha_single_expert_equals_multihead_attention():
    d, H, Dh = 16, 2, 8
    att = MoMHA(d, self_attention=True, num_expert=1, top_k=1, expert_dim=H * Dh, head_dim=Dh).eval()
    T, B = 5, 3
    x = torch.randn(T, B, d)
    y, _, _ = att(x, x, x)
    Wq, Wo = att.q_proj.experts.w[0], att.q_proj.output_experts.w[0]
    q = (x @ Wq).view(T, B, H, Dh).permute(1, 2, 0, 3) * Dh ** -0.5
    k = att.k_proj(x).view(T, B, H, Dh).permute(1, 2, 0, 3)
    v = att.v_proj(x).view(T, B, H, Dh).permute(1, 2, 0, 3)
    a = torch.softmax(q @ k.transpose(-1, -2), -1) @ v  # rel_pos_emb is zero at init
    ref = (a.permute(2, 0, 1, 3).reshape(T, B, H * Dh) @ Wo) * GATE_ONE
    torch.testing.assert_close(y, ref, rtol=1e-5, atol=1e-6)


def test_relative_position_logits_depend_on_offset_only():
    att = MoMHA(8, self_attention=True, num_expert=1, top_k=1, expert_dim=8, head_dim=4, max_positions=4)
    with torch.no_grad():
        att.rel_pos_emb.normal_()
    q = torch.ones(1, 1, 2, 6, 4)
    lg = att.rel_pos_logits(q, 6)[0, 0, 0]
    assert torch.allclose(lg[0, 1], lg[2, 3]) and torch.allclose(lg[5, 0], lg[4, 0])  # clamp at -3
    assert torch.allclose(lg[0, 5], lg[0, 3])  # clamp at +3


class _Dummy(torch.nn.Module):
    """Deterministic block: new state = tanh(W x) for active tokens, records the masks it received."""

    def __init__(self, d):
        super().__init__()
        self.lin = torch.nn.Linear(d, d)
        self.masks = []

    def forward(self, x, soft, mask, layer_idx, **kw):
        self.masks.append(mask.clone())
        return (torch.tanh(self.lin(x)),)


def _run_act(act, h, pad, steps):
    state, soft, masks = None, None, []
    outs = [h]
    for i in range(steps):
        state, out, soft, loss, pnh = act(state, h, soft, pad, i)
        h = out[0]
        outs.append(h)
    return soft, outs, loss


def test_stick_breaking_mass_and_depth_cap():
    d, T, B, steps = 6, 4, 2, 5
    act = ACTWrapper(_Dummy(d), d, threshold=0.999).eval()
    h0 = torch.randn(T, B, d)
    pad = torch.ones(T, B)
    soft, outs, loss = _run_act(act, h0, pad, steps)
    # zero-initialised gate -> halting prob 1/2 at each gated step: weights 1/2, 1/4, 1/8, 1/16, 1/16(cap)
    w = [0.5, 0.25, 0.125, 0.0625]
    ref = sum(wi * outs[i + 1] for i, wi in enumerate(w)) + 0.0625 * outs[steps]
    torch.testing.assert_close(soft, ref, rtol=1e-5, atol=1e-6)
    # ACT loss: expected depth index sum_j p_j * j + survival * last index
    exp_depth = sum(wi * i for i, wi in enumerate(w)) + 0.0625 * 4
    assert float(loss) == pytest.approx(exp_depth, rel=1e-5)


def test_halted_tokens_get_zero_mask_and_frozen_state():
    d, T, B = 6, 4, 1
    dummy = _Dummy(d)
    act = ACTWrapper(dummy, d, threshold=0.99).eval()
    with torch.no_grad():  # force halting on the first gated step for every token
        act._gate[-1].weight.zero_()
        act._gate[0].weight.zero_(); act._gate[0].bias.fill_(1.0)
        act._gate[-1].weight[1].fill_(10.0)
    pad = torch.ones(T, B); pad[-1] = 0  # last position is padding
    h0 = torch.randn(T, B, d)
    state, soft = None, None
    h = h0
    softs = []
    for i in range(4):
        state, out, soft, loss, pnh = act(state, h, soft, pad, i)
        h = out[0]
        softs.append(soft.clone())
    assert torch.equal(dummy.masks[0], pad)
    for m in dummy.masks[1:]:
        assert torch.all(m == 0)  # halted (and pad) tokens are never dispatched
    torch.testing.assert_close(softs[1], softs[2]); torch.testing.assert_close(softs[2], softs[3])


def _tiny_sut(**kw):
    b = SUTBlockConfig(embed_dim=32, attn_num_expert=4, attn_k=2, attn_expert_dim=16, head_dim=8,
                       ff_num_expert=4, ff_k=2, ff_expert_dim=32, dropout=0.0, miloss=0.1)
    return SUTConfig(vocab_size=30, padding_idx=0, block=b, encoder_layers=4, decoder_layers=3, **kw)


def test_classifier_padding_invariance_and_mim_masking():
    model = SUTClassifier(_tiny_sut(halting=True)).eval()
    x = torch.tensor([[2, 7, 8, 9, 6, 10]])
    xp = torch.tensor([[2, 7, 8, 9, 6, 10, 0, 0, 0]])
    y1, _ = model(x)
    aux1 = collect_moe_aux_loss(model)
    y2, _ = model(xp)
    aux2 = collect_moe_aux_loss(model)
    torch.testing.assert_close(y1, y2, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(torch.as_tensor(aux1), torch.as_tensor(aux2), rtol=1e-4, atol=1e-6)


def test_seq2seq_and_untied_backward_reach_routers():
    for cfg, model_cls, args in [
        (_tiny_sut(halting=True), SUTSeq2Seq, (torch.randint(5, 20, (3, 7)), torch.tensor([[1, 22]] * 3))),
        (_tiny_sut(halting=False, share_layers=False), SUTClassifier, (torch.randint(5, 20, (3, 7)),)),
    ]:
        model = model_cls(cfg).train()
        logits, extra = model(*args)
        loss = logits.float().pow(2).mean() + collect_moe_aux_loss(model) + extra["total_aux_loss"]
        loss.backward()
        grads = [p.grad for n, p in model.named_parameters() if "w_gate" in n]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
