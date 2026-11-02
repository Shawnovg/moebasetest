"""Evaluation: answer accuracy (overall, per task tag, whole-sequence exact match) and routing usage.

Routing usage is read from each routed module's ``route_record`` (set when ``state.record`` is True):
an integer tensor of unit ids with -1 at padding. For every module we count, per task tag, how many
(token, pick) pairs went to each unit. Summaries per module:

* ``load``      fraction of picks per unit (sums to 1)
* ``entropy``   normalised entropy of ``load`` (1 = perfectly even, 0 = one unit takes everything)
* ``max_load``  share of the busiest unit;  ``dead`` units with < 1 % of the even share
MoM's record is split into one group per (step, attention / FFN); MiTA's into one per head.
"""
from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn.functional as F

IGN = -100


def route_groups(module):
    rec = module.route_record
    if rec is None:
        return {}
    if rec.dim() == 5:                                       # MoM: B,T,steps,2,k
        return {f"step{s}.{'attn' if t == 0 else 'ffn'}": rec[:, :, s, t]
                for s in range(rec.shape[2]) for t in range(2)}
    return {"": rec}                                         # B,T,k (MiTA: B,T,heads -> pooled over heads)


def usage_summary(counts: torch.Tensor):
    tot = counts.sum()
    load = (counts / tot.clamp_min(1)).tolist()
    p = counts.float() / tot.clamp_min(1)
    ent = float(-(p[p > 0] * p[p > 0].log()).sum() / math.log(len(p))) if len(p) > 1 else 1.0
    return {"load": [round(x, 5) for x in load], "entropy": round(ent, 5), "max_load": round(max(load), 5),
            "dead": int((p < 0.01 / len(p)).sum())}


@torch.no_grad()
def evaluate(model, dataset, batch_size=256, max_batches=None, routing=True, task_ids=None):
    """Returns dict: loss, acc, seq_acc, per_task {tag: acc}, routing {module.group: summary + per_task}."""
    model.eval()
    st = model.state
    st.record = routing
    random_was = st.random_active
    st.random_active = False
    loss_sum, n_ans, n_ok, n_seq, n_seq_ok = 0.0, 0, 0, 0, 0
    task_ok, task_n = defaultdict(int), defaultdict(int)
    counts = {}
    for tok, tgt, pad in dataset.eval_batches(batch_size, max_batches):
        m = tgt != IGN
        logits, _ = model(tok, pad, select=m)
        loss_sum += F.cross_entropy(logits, tgt[m], reduction="sum").item()
        ok = torch.zeros_like(m)
        ok[m] = logits.argmax(-1) == tgt[m]
        n_ans += int(m.sum()); n_ok += int(ok.sum())
        seq_ok = (ok.sum(1) == m.sum(1))
        n_seq += tok.shape[0]; n_seq_ok += int(seq_ok.sum())
        tags = tok[:, 0]
        for t in tags.unique().tolist():
            sel = tags == t
            task_ok[t] += int(ok[sel].sum()); task_n[t] += int(m[sel].sum())
        if routing:
            for name, mod in model.routed_modules():
                for g, rec in route_groups(mod).items():
                    key = f"{name}.{g}" if g else name
                    U = mod.num_units
                    if key not in counts:
                        counts[key] = defaultdict(lambda U=U: torch.zeros(U, dtype=torch.long))
                    for t in tags.unique().tolist():
                        r = rec[tags == t].reshape(-1)
                        r = r[r >= 0]
                        counts[key][t] += torch.bincount(r, minlength=U)[:U]
    st.record = False
    st.random_active = random_was
    names = task_ids or {}
    out = {"loss": loss_sum / max(n_ans, 1), "acc": n_ok / max(n_ans, 1), "seq_acc": n_seq_ok / max(n_seq, 1),
           "per_task": {names.get(t, str(t)): task_ok[t] / max(task_n[t], 1) for t in task_ok}}
    if routing:
        r = {}
        for key, per_t in counts.items():
            total = sum(per_t.values())
            s = usage_summary(total)
            s["per_task"] = {names.get(t, str(t)): c.tolist() for t, c in per_t.items()}
            r[key] = s
        out["routing"] = r
    model.train()
    return out
