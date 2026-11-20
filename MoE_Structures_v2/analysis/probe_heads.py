"""Probe 2, per-head readout (fact): where is each fact readable, and does the router deliver it?

For every trained model in runs_probe/ and every layer, at the answer positions, compute EVERY head's output
(the head's 32-number attention result before the output projection), even for heads the router switched off.
Then train small linear classifiers (the model stays frozen) to predict the answer value (128 classes) from:

* one head alone           -> "can this head on its own decode the fact?"   (per-head accuracy, specialisation)
* all heads together       -> "is the fact readable anywhere in this layer?"           (STORAGE)
* the actual layer output  -> sum of gate x head contribution, as the model uses it    (DELIVERED)

STORAGE - DELIVERED is the access gap: information the heads hold but the router does not pass on.
For router-free controls all heads are always on, so the gap measures only what the output projection loses.
Router coverage: per answer, how many of the heads that decode it alone were actually selected.

A linear probe shows information is READABLE, not that the model uses it; pair with probe_forced_access.py.
Output: reports/probe/heads.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backbone.data import Dataset  # noqa: E402
from backbone.metrics import IGN  # noqa: E402
from backbone.paths import DATA_ROOT  # noqa: E402
from backbone.layers import masked_softmax_attention  # noqa: E402
from analysis.probe_forced_access import RUNS, OUT, load  # noqa: E402

N_TRAIN_SEQ, N_TEST_SEQ = 20000, 4000


def heads_of(attn, x, pad):
    """x: layer input after ln1 (B,T,d). Returns per-head outputs (B,T,H,hd), per-head output maps (H,hd,d),
    gates (B,T,H) (1 for router-free), and the delivered layer output (B,T,d)."""
    B, T, _ = x.shape
    name = type(attn).__name__
    if name in ("DenseAttention", "GroupedAttention"):
        H, hd = attn.n_heads, attn.head_dim
        nkv = getattr(attn, "n_kv", H)
        q = attn.q_proj(x).view(B, T, H, hd).transpose(1, 2)
        k = attn.k_proj(x).view(B, T, nkv, hd).transpose(1, 2)
        v = attn.v_proj(x).view(B, T, nkv, hd).transpose(1, 2)
        if nkv != H:
            k = k.repeat_interleave(H // nkv, 1); v = v.repeat_interleave(H // nkv, 1)
        q, k = attn.rope(q), attn.rope(k)
        out = masked_softmax_attention(q, k, v, pad, hd ** -0.5).transpose(1, 2) * attn.out_scale
        Wo = attn.o_proj.weight.T.reshape(H, hd, -1)
        gates = x.new_ones(B, T, H)
    elif name == "MoAAttention":
        g = attn.q_proj
        H, hd = g.num_experts, attn.head_dim
        q = torch.einsum("btd,hde->bhte", x, g.experts.w) * attn.scaling
        k, v = attn.k_proj(x), attn.v_proj(x)
        q, k = attn.rope(q), attn.rope(k)
        w = torch.einsum("bhie,bje->bhij", q, k).masked_fill(pad[:, None, None, :], float("-inf"))
        out = torch.einsum("bhij,bje->bihe", torch.softmax(w, -1), v)
        Wo = g.output_experts.w
        logits = g.w_gate_mlp(x) if g.router == "mlp" else x @ g.w_gate
        probs = torch.softmax(logits, -1)
        tv, ti = probs.topk(g.k, -1)
        if g.gate_norm != "raw":
            tv = tv / tv.sum(-1, keepdim=True)
        gates = torch.zeros_like(probs).scatter(-1, ti, tv)
    elif name == "MoHAttention":
        H, hd = attn.num_heads, attn.head_dim
        qkv = attn.qkv(x).reshape(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = attn.rope(q), attn.rope(k)
        out = masked_softmax_attention(q, k, v, pad, attn.scale).transpose(1, 2)
        Wo = attn.proj.weight.T.reshape(H, hd, -1)
        p = torch.softmax(attn.wg(x), -1)
        n_sel = attn.routed_head // attn.head_group
        tv, ti = p.topk(n_sel, -1)
        if n_sel > 1:
            tv = tv / tv.sum(-1, keepdim=True)
        if attn.gate_scale == "k":
            tv = tv * n_sel
        gates = torch.zeros_like(p).scatter(-1, ti, tv).repeat_interleave(attn.head_group, -1)
    else:
        raise ValueError(name)
    delivered = torch.einsum("bth,bthe,hed->btd", gates, out, Wo)
    return out, gates, delivered


def features(model, ds, n_seq, ent0, rel0, sym0, bs=250):
    """Per layer: per-head outputs at answer slots (N,H,hd), gates (N,H), delivered attention output (N,d) and the
    residual stream after the whole block (N,d). Targets (N,4): answer value, first name, last name, relation."""
    caps, post = {}, {}
    hooks = [blk.ln1.register_forward_hook(lambda m, i, o, li=li: caps.__setitem__(li, o))
             for li, blk in enumerate(model.blocks)]
    hooks += [blk.register_forward_hook(lambda m, i, o, li=li: post.__setitem__(li, o[0]))
              for li, blk in enumerate(model.blocks)]
    per = {li: {"out": [], "gates": [], "deliv": [], "resid": []} for li in range(len(model.blocks))}
    ys = []
    n = 0
    with torch.no_grad():
        for tok, tgt, pad in ds.eval_batches(bs, None):
            if n >= n_seq:
                break
            m = tgt != IGN
            model(tok, pad, select=m)
            for li, blk in enumerate(model.blocks):
                out, gates, deliv = heads_of(blk.attn, caps[li], pad)
                per[li]["out"].append(out[m].half()); per[li]["gates"].append(gates[m].half())
                per[li]["deliv"].append(deliv[m].half()); per[li]["resid"].append(post[li][m].half())
            bi, ti = m.nonzero(as_tuple=True)
            ys.append(torch.stack([tgt[bi, ti] - sym0, tok[bi, ti - 3] - ent0, tok[bi, ti - 2] - ent0,
                                   tok[bi, ti - 1] - rel0], 1))
            n += tok.shape[0]
    for h in hooks:
        h.remove()
    return {li: {k: torch.cat(v).float() for k, v in d.items()} for li, d in per.items()}, torch.cat(ys)


def fit_linear(Xtr, ytr, Xte, yte, classes, epochs=12, bs=2048, lr=3e-3):
    """Multinomial logistic regression (Adam, light weight decay); returns test accuracy and per-item correctness."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-5
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    lin = torch.nn.Linear(Xtr.shape[1], classes)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=1e-5)
    g = torch.Generator().manual_seed(0)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), bs):
            j = perm[i:i + bs]
            loss = F.cross_entropy(lin(Xtr[j]), ytr[j])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        ok = lin(Xte).argmax(-1) == yte
    return float(ok.float().mean()), ok


TARGETS = {"answer": 128, "first": 90, "last": 90, "relation": 4}


def probe_run(run):
    """Attention's job in fact is to GATHER the three query words into the answer slot; the lookup itself is
    done afterwards (FFN). So heads are probed for the query words; the answer is probed on the residual stream."""
    cfg, model, vocab = load(run)
    ids = dict(ent0=vocab.index("e0"), rel0=vocab.index("r0"), sym0=vocab.index("n0"))
    tr, ytr = features(model, Dataset("fact", "train", DATA_ROOT), N_TRAIN_SEQ, **ids)
    te, yte = features(model, Dataset("fact", "test", DATA_ROOT), N_TEST_SEQ, **ids)
    for j, c in enumerate(TARGETS.values()):
        assert ytr[:, j].min() >= 0 and ytr[:, j].max() < c, (j, ytr[:, j].min(), ytr[:, j].max())
    res = {"structure": cfg["structure"], "variant": cfg["variant"], "seed": cfg["train"]["seed"], "layers": {}}
    with torch.enable_grad():
        for li in tr:
            H = tr[li]["out"].shape[1]
            sel = te[li]["gates"] > 0
            L = {"answer_after_block": fit_linear(tr[li]["resid"], ytr[:, 0], te[li]["resid"], yte[:, 0], 128)[0],
                 "answer_from_attention": fit_linear(tr[li]["deliv"], ytr[:, 0], te[li]["deliv"], yte[:, 0], 128)[0],
                 "selected_heads_mean": float(sel.sum(1).float().mean()), "usage": sel.float().mean(0).tolist()}
            for j, (t, c) in list(enumerate(TARGETS.items()))[1:]:
                L[f"{t}_storage"] = fit_linear(tr[li]["out"].flatten(1), ytr[:, j], te[li]["out"].flatten(1), yte[:, j], c)[0]
                L[f"{t}_delivered"] = fit_linear(tr[li]["deliv"], ytr[:, j], te[li]["deliv"], yte[:, j], c)[0]
                accs, oks = [], []
                for h in range(H):
                    a, ok = fit_linear(tr[li]["out"][:, h], ytr[:, j], te[li]["out"][:, h], yte[:, j], c, epochs=6)
                    accs.append(a); oks.append(ok)
                oks = torch.stack(oks, 1)
                n_dec = oks.sum(1).float()
                L[f"{t}_per_head"] = accs
                L[f"{t}_heads_decoding"] = float(n_dec.mean())
                L[f"{t}_coverage"] = float((oks & sel).any(1).float()[n_dec > 0].mean()) if (n_dec > 0).any() else None
            res["layers"][li] = L
            print(run, li, {k: round(v, 3) for k, v in L.items() if isinstance(v, float)}, flush=True)
    return res


def main():
    torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    shard = sys.argv[1] if len(sys.argv) > 1 else "0/1"
    i, n = map(int, shard.split("/"))
    path = OUT / f"heads2_{i}of{n}.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    runs = sorted(d.name for d in RUNS.iterdir())[i::n]
    for r in runs:
        if r in done:
            continue
        done[r] = probe_run(r)
        path.write_text(json.dumps(done, indent=1))


if __name__ == "__main__":
    main()
