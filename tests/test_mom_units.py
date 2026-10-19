"""Behavioural tests of the MoM port: transfer mapping, hand-specified routes, SKIP, causality."""
import pytest

torch = pytest.importorskip("torch")

from moebase.models.mom.convert import get_layer_map, init_from_vanilla  # noqa: E402
from moebase.models.mom.donor import GPT2LMHead  # noqa: E402
from moebase.models.mom.model import MoMCausalLM, MoMConfig, MoTBlock, gpt2_init_weights  # noqa: E402

TOL = dict(rtol=1e-5, atol=1e-6)


def cfg(**kw):
    base = dict(vocab_size=40, n_positions=32, n_embd=32, n_layer=6, n_head=4, n_inner=64,
                chunks=1, num_bottom_layers=1, num_top_layers=0, num_attn=4, num_ffn=4, depth_per_chunk=4,
                router_type="gru", router_dim=16, routing_top_k=1, enable_pause=True)
    base.update(kw)
    return MoMConfig(**base)


def test_layout_counts():
    c = cfg()
    assert c.donor_layers() == 6 and c.block_applications() == 6
    assert get_layer_map(c) == {"0": "0", "1": "1-0", "2": "1-1", "3": "1-2", "4": "1-3", "5": "2"}
    small = cfg(num_bottom_layers=2, chunks=2, num_top_layers=0)  # paper MoM-small [1-1-4-1-4-1]
    assert small.donor_layers() == 12


def _donor_and_mom(**kw):
    c = cfg(**kw)
    donor = GPT2LMHead(c)
    with torch.no_grad():
        for p in donor.parameters():  # make LayerNorms non-trivial so mapping errors show up
            p.add_(0.05 * torch.randn_like(p))
    mom = MoMCausalLM(c)
    mom, rep = init_from_vanilla(mom, donor.state_dict(), c)
    return c, donor, mom, rep


def test_transfer_copies_donor_tensors_exactly():
    c, donor, mom, rep = _donor_and_mom()
    d, m = donor.state_dict(), dict(mom.named_parameters())
    for new, old in rep["copied"]:
        assert torch.equal(m[new], d[old]), new
    # module j of the chunk is donor layer 1 + j; anchor is donor layer 5
    for j in range(4):
        assert torch.equal(m[f"transformer.h.1.mot_attn.c_attn.{j}.weight"], d[f"transformer.h.{1 + j}.attn.c_attn.weight"])
        assert torch.equal(m[f"transformer.h.1.mot_ffn.c_fc.{j}.bias"], d[f"transformer.h.{1 + j}.mlp.c_fc.bias"])
    assert torch.equal(m["transformer.h.2.mot_ffn.ffn_layernorm.weight"], d["transformer.h.5.ln_2.weight"])
    # faithful upstream behaviour: ln_f and routers are not transferred
    assert "transformer.ln_f.weight" in rep["not_copied"]
    assert all(any(k in n for k in ("router", "ln_f")) for n in rep["not_copied"])
    assert mom.lm_head.weight is mom.transformer.wte.weight


def _force_route(block: MoTBlock, cls: int, score: float = 1.0):
    for r in (block.mot_attn.router, block.mot_ffn.router):
        with torch.no_grad():
            r.router.weight.zero_()
            r.router.bias.fill_(-1.0)
            r.router.bias[cls] = score


def test_hand_route_equals_repeated_donor_layer():
    c, donor, mom, _ = _donor_and_mom()
    mom.eval(); donor.eval()
    block = mom.transformer.h[1]
    x = torch.randn(2, 7, 32)
    for m in range(4):
        _force_route(block, m)
        y, _, routes = block(x)
        ref = x
        for _ in range(c.depth_per_chunk):
            ref = donor.transformer.h[1 + m](ref)  # causal mask inside, no padding
        torch.testing.assert_close(y, ref, **TOL)
        assert all(torch.all(ra == m) and torch.all(rf == m) for ra, rf in routes)


def test_skip_is_identity_and_reentry_possible():
    c, donor, mom, _ = _donor_and_mom()
    block = mom.transformer.h[1].eval()
    x = torch.randn(2, 5, 32)
    _force_route(block, c.num_attn)  # SKIP class
    y, _, _ = block(x)
    torch.testing.assert_close(y, x)


def test_block_is_deterministic_and_router_state_resets_per_call():
    c, donor, mom, _ = _donor_and_mom()
    block = mom.transformer.h[1].eval()
    x = torch.randn(2, 5, 32)
    torch.testing.assert_close(block(x)[0], block(x)[0])


def test_causal_mask_full_model():
    c, donor, mom, _ = _donor_and_mom(routing_top_k=2)
    mom.eval()
    ids = torch.randint(5, 40, (1, 10))
    ids2 = ids.clone(); ids2[0, 6:] = torch.randint(5, 40, (4,))
    l1, l2 = mom(ids)["logits"], mom(ids2)["logits"]
    torch.testing.assert_close(l1[:, :6], l2[:, :6], **TOL)


def test_lm_loss_only_on_target_and_aux_present():
    c, donor, mom, _ = _donor_and_mom()
    mom.train()
    ids = torch.randint(5, 40, (3, 8))
    labels = torch.full_like(ids, -100); labels[:, -1] = ids[:, -1]
    out = mom(ids, attention_mask=torch.ones_like(ids, dtype=torch.float), labels=labels)
    assert out["aux_loss"] is not None and torch.isfinite(out["aux_loss"])
    (out["loss"] + c.router_aux_loss_coef * out["aux_loss"]).backward()
    g = mom.transformer.h[1].mot_attn.router.x2h.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_gpt2_init_scales_only_direct_c_proj():
    c = cfg()
    m = MoMCausalLM(c)
    gpt2_init_weights(m, 0.02, 6)
    anchor = m.transformer.h[0].mot_attn.c_proj.weight.std().item()
    module = m.transformer.h[1].mot_attn.c_proj[0].weight.std().item()
    assert anchor == pytest.approx(0.02 / (12 ** 0.5), rel=0.25)
    assert module == pytest.approx(0.02, rel=0.25)
