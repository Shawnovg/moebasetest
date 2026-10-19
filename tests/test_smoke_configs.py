"""Every config builds, runs forward/backward on a tiny batch with finite gradients, and (slow) can fit
a tiny fixed training set (plan stage 'learning check')."""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("yaml")

from moebase.data import logic  # noqa: E402
from moebase.train.build import build_model, build_optimizer, lr_lambda_factory  # noqa: E402
from moebase.train.config import load_config  # noqa: E402
from moebase.train.data_utils import LogicTensors  # noqa: E402
from moebase.train.loop import forward_loss  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted(p for p in (ROOT / "configs" / "logic").glob("*.yaml") if not p.name.startswith("_"))

EXAMPLES = [
    ("^", "a", "( not a )"), ("<", "a", "( a ( or b ) )"), (">", "( a ( or b ) )", "a"),
    ("=", "( a ( and b ) )", "( b ( and a ) )"), ("|", "( a ( and b ) )", "( not a )"),
    ("v", "a", "( ( not a ) ( or b ) )"), ("#", "a", "b"), ("#", "( c ( or d ) )", "( not e )"),
]


def tiny_examples(n=32):
    out = []
    vars_ = "abcdef"
    for i in range(n):
        lab, p1, p2 = EXAMPLES[i % len(EXAMPLES)]
        # rename variables to vary inputs while keeping the label
        perm = dict(zip(vars_, vars_[i % 6:] + vars_[:i % 6]))
        ren = lambda s: [perm.get(t, t) for t in s.split()]  # rename variable tokens only
        out.append(logic.Example(lab, ren(p1), ren(p2), "tiny", i))
    for e in out:
        assert logic.verify_label(e)[0]
    return out


def small(cfg):
    """Shrink widths so the smoke test is fast; keeps the family's structure."""
    p = cfg["model"]["params"]
    if cfg["model"]["kind"].startswith("sut"):
        p["encoder_layers"] = min(p.get("encoder_layers", 6), 3)
        if "decoder_layers" in p:
            p["decoder_layers"] = 2
    return cfg


@pytest.mark.parametrize("path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_config_forward_backward(path):
    cfg = small(load_config(path))
    vocab = logic.Vocab()
    model, interface, _ = build_model(cfg["model"], vocab)
    data = LogicTensors(tiny_examples(8), interface, vocab)
    batch = next(data.batches(8, shuffle=False))
    model.train()
    loss, logs, probs7, _ = forward_loss(model, cfg["model"]["kind"], batch, cfg.get("loss", {}), vocab, training=True)
    assert torch.isfinite(loss)
    assert probs7.shape == (8, 7) and torch.allclose(probs7.sum(-1), torch.ones(8), atol=1e-5)
    loss.backward()
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), n


@pytest.mark.slow
@pytest.mark.parametrize("name", ["dense6", "sut_common6", "sut_ref_seq2seq", "mom_donor6_causal"])
def test_tiny_set_learning(name):
    cfg = load_config(ROOT / "configs" / "logic" / f"{name}.yaml")
    for blk in ("block",):
        if blk in cfg["model"]["params"]:
            cfg["model"]["params"][blk].update(dropout=0.0, attention_dropout=0.0, activation_dropout=0.0, gating_dropout=0.0)
    vocab = logic.Vocab()
    model, interface, _ = build_model(cfg["model"], vocab)
    data = LogicTensors(tiny_examples(32), interface, vocab)
    o = dict(cfg["optim"]); o.update(lr=1e-3, warmup_steps=20, warmup_ratio=0.0, schedule="constant", warmup_init_lr=1e-7)
    if o["type"] == "adam_fairseq":
        o["schedule"] = "constant"
    opt = build_optimizer(model, o)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(o, 300))
    lcfg = dict(cfg.get("loss", {})); lcfg["label_smoothing"] = 0.0
    model.train()
    batch = next(data.batches(32, shuffle=False))
    for _ in range(300):
        loss, _, _, _ = forward_loss(model, cfg["model"]["kind"], batch, lcfg, vocab, training=True)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()
    _, _, probs7, _ = forward_loss(model, cfg["model"]["kind"], batch, lcfg, vocab, training=False)
    from moebase.models.sut.model import clear_moe_statistics
    clear_moe_statistics(model)
    acc = (probs7.argmax(-1) == batch["y"]).float().mean().item()
    assert acc >= 0.9, f"{name} only reached {acc:.2f} on 32 memorisable examples"
