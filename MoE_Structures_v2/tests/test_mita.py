"""MiTA: the vectorised module equals a per-query loop written directly from the reference equations.

The loop follows QishuaiWen/MiTA ``MiTA_Attention.mita`` + ``MixedAttention`` literally, per batch
element / head / query: landmarks by window means, each landmark's top-k keys, agent values by
softmax(router K^T * scale) V, expert = argmax_j router_j . q, then ONE softmax over
[q . (router * scale) * scale ; q . k_sel * scale] applied to [agent_value ; v_sel].
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backbone.layers import Rotary  # noqa: E402
from backbone.routing import RoutingState  # noqa: E402
from models.mita.mita_attention import MiTAAttention  # noqa: E402


def naive(mod, x, key_pad):
    B, N, C = x.shape
    H, d, m, kk, s = mod.num_heads, mod.head_dim, mod.m, mod.kv_topk, mod.router_topk
    sc = d ** -0.5
    q, k, v = mod.qkv(x).reshape(B, N, 3, H, d).permute(2, 0, 3, 1, 4).unbind(0)
    q, k = mod.rope(q), mod.rope(k)
    out = torch.zeros(B, H, N, d, dtype=torch.float64)
    for b in range(B):
        L = int((~key_pad[b]).sum())
        for h in range(H):
            Q, K, Vv = q[b, h, :L].double(), k[b, h, :L].double(), v[b, h, :L].double()
            R = []
            for i in range(m):
                st, en = (i * L) // m, -((-(i + 1) * L) // m)
                R.append(Q[st:en].mean(0))
            R = torch.stack(R)                                          # m x d
            topk = (R @ K.T).topk(min(kk, L), -1).indices               # m x kk
            agent = torch.softmax(R @ K.T * sc, -1) @ Vv                # m x d
            for n in range(L):
                ex = (R @ Q[n]).topk(s).indices
                idx = torch.cat([topk[e] for e in ex])
                logits = [(Q[n] @ (R * sc).T) * sc] if mod.shared else []
                vals = [agent] if mod.shared else []
                logits.append(Q[n] @ K[idx].T * sc)
                vals.append(Vv[idx])
                p = torch.softmax(torch.cat(logits), 0)
                out[b, h, n] = p @ torch.cat(vals)
    o = out.transpose(1, 2).reshape(B, N, C).float()
    y = mod.proj(o)
    return y, ~key_pad


def check(**kw):
    torch.manual_seed(0)
    B, N, C, H = 3, 23, 32, 2
    mod = MiTAAttention(C, H, rope=Rotary(C // H), state=RoutingState(), **kw)
    x = torch.randn(B, N, C)
    key_pad = torch.zeros(B, N, dtype=torch.bool)
    key_pad[1, 17:] = True
    key_pad[2, 11:] = True
    y, _ = mod(x, key_pad)
    y_ref, valid = naive(mod, x, key_pad)
    err = (y - y_ref)[valid].abs().max().item()
    assert err < 1e-4, (kw, err)
    return err


def test_default():
    check(num_landmarks=4, kv_topk=5)


def test_s2():
    check(num_landmarks=4, kv_topk=5, router_topk=2)


def test_noshared():
    check(num_landmarks=3, kv_topk=4, shared=False)


if __name__ == "__main__":
    for f in (test_default, test_s2, test_noshared):
        f()
        print(f.__name__, "ok")
