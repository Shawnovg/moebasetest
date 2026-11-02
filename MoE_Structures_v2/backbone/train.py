"""One training run: config in, ``runs/<run_name>/`` out.

    train_run(cfg) -> result dict

Steps:
1. seed everything (``train.seed``), load the dataset (DATA_ROOT/<dataset>), build ``EncoderModel``;
2. AdamW (weight decay on matrices only), linear LR warm-up then cosine decay to 0, grad-clip;
3. each step: loss = cross-entropy on answer slots + aux (the structure's balancing loss, non-zero
   only when routing.balance == 'lb'); in mode 'warmup' the first ``warmup_frac`` of the steps use
   random routing (``state.random_active``);
4. every ``eval_every`` steps: validation accuracy (history.jsonl);
5. at the end: validation and test accuracy, per-task accuracy and routing usage on the test set,
   parameter counts, wall time -> result.json; final weights -> model.pt; config -> config.json.
A run whose result.json exists is skipped (resumable sweeps).
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import dumps, run_name
from .data import Dataset, load_vocab
from .metrics import IGN, evaluate
from .model import EncoderModel, count_parameters
from .paths import DATA_ROOT, RUNS_ROOT


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def lr_at(step, cfg):
    t = cfg["train"]
    if step < t["warmup_steps"]:
        return t["lr"] * (step + 1) / t["warmup_steps"]
    prog = (step - t["warmup_steps"]) / max(1, t["steps"] - t["warmup_steps"])
    return t["lr"] * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


def build_optimizer(model, cfg):
    t = cfg["train"]
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and not n.startswith("embed")]
    no_decay = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and not n.startswith("embed"))]
    return torch.optim.AdamW([{"params": decay, "weight_decay": t["weight_decay"]},
                              {"params": no_decay, "weight_decay": 0.0}], lr=t["lr"], betas=tuple(t["betas"]))


def train_run(cfg, runs_root: Path = RUNS_ROOT, data_root: Path = DATA_ROOT, verbose=True, save_model=True):
    name = run_name(cfg)
    out = Path(runs_root) / name
    if (out / "result.json").exists():
        return json.loads((out / "result.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    t = cfg["train"]
    torch.set_num_threads(t.get("threads", 1))
    set_seed(t["seed"])
    vocab = load_vocab(data_root)
    cfg["vocab_size"] = len(vocab)
    task_ids = {i: tok[3:-1] for i, tok in enumerate(vocab) if tok.startswith("<t:")}
    train = Dataset(cfg["dataset"], "train", data_root)
    valid = Dataset(cfg["dataset"], "valid", data_root)
    test = Dataset(cfg["dataset"], "test", data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EncoderModel(cfg, len(vocab)).to(device)
    model.state.generator = torch.Generator(device=device).manual_seed(t["seed"] + 7)
    opt = build_optimizer(model, cfg)
    (out / "config.json").write_text(dumps(cfg))

    balance = cfg["routing"]["balance"]
    warm_steps = int(cfg["routing"].get("warmup_frac", 0.1) * t["steps"]) if balance == "warmup" else 0
    batches = train.train_batches(t["batch_size"], seed=t["seed"] + 1)
    hist = open(out / "history.jsonl", "w")
    t0 = time.time()
    run_loss = run_aux = 0.0
    for step in range(t["steps"]):
        model.state.random_active = step < warm_steps
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        tok, tgt, pad = (x.to(device) for x in next(batches))
        m = tgt != IGN
        logits, aux = model(tok, pad, select=m)
        loss = F.cross_entropy(logits, tgt[m])
        opt.zero_grad(set_to_none=True)
        (loss + aux).backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        opt.step()
        run_loss += loss.item(); run_aux += float(aux)
        if (step + 1) % t["eval_every"] == 0 or step + 1 == t["steps"]:
            ev = evaluate(model, valid, max_batches=t.get("eval_batches"), routing=False, task_ids=task_ids)
            n = t["eval_every"] if (step + 1) % t["eval_every"] == 0 else (step + 1) % t["eval_every"]
            rec = {"step": step + 1, "train_loss": run_loss / n, "train_aux": run_aux / n, "grad_norm": float(gn),
                   "valid_loss": ev["loss"], "valid_acc": ev["acc"], "valid_per_task": ev["per_task"],
                   "time": time.time() - t0}
            hist.write(json.dumps(rec) + "\n"); hist.flush()
            if verbose:
                print(f"[{name}] step {step + 1} loss {rec['train_loss']:.3f} aux {rec['train_aux']:.4f} "
                      f"valid {ev['acc']:.3f} ({rec['time']:.0f}s)", flush=True)
            run_loss = run_aux = 0.0
    hist.close()
    train_time = time.time() - t0
    ev_valid = evaluate(model, valid, routing=False, task_ids=task_ids)
    ev_test = evaluate(model, test, routing=True, task_ids=task_ids)
    result = {"run": name, "dataset": cfg["dataset"], "structure": cfg["structure"], "variant": cfg["variant"],
              "balance": balance, "seed": t["seed"], "steps": t["steps"],
              "params": count_parameters(model), "params_total": count_parameters(model, False),
              "valid_acc": ev_valid["acc"], "valid_loss": ev_valid["loss"],
              "test_acc": ev_test["acc"], "test_loss": ev_test["loss"], "test_seq_acc": ev_test["seq_acc"],
              "test_per_task": ev_test["per_task"], "routing": ev_test.get("routing", {}),
              "train_time_s": train_time, "device": str(device)}
    if save_model:
        torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
                   out / "model.pt")
    (out / "result.json").write_text(json.dumps(result, indent=1))
    return result
