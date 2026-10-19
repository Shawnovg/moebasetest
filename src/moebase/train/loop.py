"""Config-driven training/evaluation loop with run records.

Losses follow each source:
* SUT (``aux_reduction: fairseq``): fairseq divides the *summed* loss, including the MoE auxiliary
  terms returned by ``get_aux_loss_and_clear`` and the ACT term, by ``sample_size`` (= number of
  non-pad target tokens) before the update. The effective MIM weight is therefore miloss/sample_size.
  ``aux_reduction: mean`` (mean task loss + raw aux) is offered as a labelled alternative.
* MoM: ``loss + router_aux_loss_coef * aux_loss`` (upstream train.py Trainer.compute_loss).

Checkpoint selection: lowest validation Brier score (project rule). Auxiliary losses never enter the
selection score. Test partitions are evaluated only for the selected checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import platform
import random
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .. import paths
from ..data.logic import Vocab
from ..models.mom.convert import init_from_vanilla
from ..models.sut.model import clear_moe_statistics, collect_moe_aux_loss, label_smoothed_nll_loss, moe_modules
from .build import build_model, build_optimizer, lr_lambda_factory
from .data_utils import load_split
from .metrics import summarize


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((paths.PROJECT_ROOT / "src").rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def to_device(batch: Dict[str, torch.Tensor], device):
    return {k: v.to(device) for k, v in batch.items()}


def _depth_per_example(logs, extra, tokens, vocab):
    d = extra.get("expected_depth_tokens")
    if d is None:
        return
    m = tokens.ne(vocab.pad).float()  # B x T
    logs["depth_ex"] = (d.t() * m).sum(1) / m.sum(1)


# ------------------------------------------------------------------ forward + loss per interface
def forward_loss(model, kind: str, batch, loss_cfg: dict, vocab: Vocab, training: bool):
    """Returns (loss for backward, logs dict, probs7 (B x 7), invalid (B) or None)."""
    eps = loss_cfg.get("label_smoothing", 0.0)
    logs = {}
    invalid = None
    if kind == "sut_seq2seq":
        logits, extra = model(batch["src"], batch["prev"])
        lprobs = F.log_softmax(logits.float(), dim=-1)
        loss_sum, nll_sum = label_smoothed_nll_loss(lprobs.view(-1, lprobs.size(-1)), batch["tgt"].view(-1), eps,
                                                    ignore_index=vocab.pad)
        sample_size = batch["tgt"].ne(vocab.pad).sum()
        aux = extra["total_aux_loss"] + (collect_moe_aux_loss(model) if training else 0.0)
        if loss_cfg.get("aux_reduction", "fairseq") == "fairseq":
            loss = (loss_sum + aux) / sample_size
        else:
            loss = loss_sum / sample_size + aux
        first = logits[:, 0, :]
        probs7 = F.softmax(first[:, vocab.label_ids].float(), dim=-1)
        invalid = ~torch.isin(first.argmax(-1), torch.tensor(vocab.label_ids, device=first.device))
        logs.update(nll=(nll_sum / sample_size).item(), aux=float(aux) if not torch.is_tensor(aux) else aux.item(),
                    enc_halt=float(extra["encoder_expected_halt"]), dec_halt=float(extra["decoder_expected_halt"]))
        _depth_per_example(logs, extra, batch["src"], vocab)
    elif kind == "sut_classifier":
        logits, extra = model(batch["input"])
        loss_sum = F.cross_entropy(logits.float(), batch["y"], reduction="sum", label_smoothing=eps)
        n = batch["y"].numel()
        aux = extra["total_aux_loss"] + (collect_moe_aux_loss(model) if training else 0.0)
        loss = (loss_sum + aux) / n if loss_cfg.get("aux_reduction", "fairseq") == "fairseq" else loss_sum / n + aux
        probs7 = F.softmax(logits.float(), dim=-1)
        logs.update(ce=(loss_sum / n).item(), aux=float(aux) if not torch.is_tensor(aux) else aux.item(),
                    enc_halt=float(extra["encoder_expected_halt"]))
        _depth_per_example(logs, extra, batch["input"], vocab)
    elif kind in ("mom_causal", "gpt2_causal"):
        out = model(batch["input"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        loss = out["loss"]
        if out["aux_loss"] is not None:
            loss = loss + model.config.router_aux_loss_coef * out["aux_loss"]
            logs["aux"] = out["aux_loss"].item()
        logs["ce"] = out["loss"].item()
        idx = torch.arange(batch["input"].size(0), device=batch["input"].device)
        pred_logits = out["logits"][idx, batch["pos"] - 1]  # logits at t predict token t+1
        probs7 = F.softmax(pred_logits[:, vocab.label_ids].float(), dim=-1)
        invalid = ~torch.isin(pred_logits.argmax(-1), torch.tensor(vocab.label_ids, device=pred_logits.device))
        if "routes" in out:
            logs["routes"] = out["routes"]
    elif kind in ("mom_classifier", "gpt2_classifier"):
        out = model(batch["input"], attention_mask=batch["attention_mask"])
        loss = F.cross_entropy(out["logits"].float(), batch["y"], label_smoothing=eps)
        logs["ce"] = loss.item()
        if out["aux_loss"] is not None:
            loss = loss + model.config.router_aux_loss_coef * out["aux_loss"]
            logs["aux"] = out["aux_loss"].item()
        probs7 = F.softmax(out["logits"].float(), dim=-1)
        if "routes" in out:
            logs["routes"] = out["routes"]
    else:
        raise ValueError(kind)
    return loss, logs, probs7, invalid


# ------------------------------------------------------------------ routing diagnostics
class RouteStats:
    def __init__(self):
        self.moe_counts: Dict[str, torch.Tensor] = {}
        self.mom_counts: Dict[str, torch.Tensor] = {}
        self.active_tokens = []

    def update_sut(self, model):
        for name, m in moe_modules(model).items():
            if m.usage_counts is not None:
                self.moe_counts[name] = self.moe_counts.get(name, 0) + m.usage_counts.cpu()
            m.reset_usage()

    def update_mom(self, routes, mask: Optional[torch.Tensor]):
        valid = None if mask is None else mask.reshape(-1).bool().cpu()
        for step, (ra, rf) in enumerate(routes):
            for tag, r in (("attn", ra), ("ffn", rf)):
                if r is None:
                    continue
                r = r.detach().cpu()
                if valid is not None:
                    r = r[valid]
                c = torch.bincount(r.reshape(-1), minlength=r.max().item() + 1).float()
                key = f"step{step}.{tag}"
                prev = self.mom_counts.get(key)
                if prev is not None and prev.numel() < c.numel():
                    prev = F.pad(prev, (0, c.numel() - prev.numel()))
                if prev is not None and c.numel() < prev.numel():
                    c = F.pad(c, (0, prev.numel() - c.numel()))
                self.mom_counts[key] = c if prev is None else prev + c

    def summary(self):
        def frac(d):
            return {k: (v / v.sum().clamp_min(1)).tolist() for k, v in d.items()}
        return {"sut_expert_usage": frac(self.moe_counts), "mom_module_usage": frac(self.mom_counts)}


@torch.no_grad()
def evaluate(model, kind, data, loss_cfg, vocab, device, batch_size=512, route_stats: Optional[RouteStats] = None):
    model.eval()
    clear_moe_statistics(model)
    for m in moe_modules(model).values():
        m.reset_usage()
    all_p, all_y, all_len, all_inv, all_dup, all_depth = [], [], [], [], [], []
    for batch in data.batches(batch_size, shuffle=False):
        batch = to_device(batch, device)
        _, logs, probs7, invalid = forward_loss(model, kind, batch, loss_cfg, vocab, training=False)
        if route_stats is not None:
            if kind.startswith("sut"):
                route_stats.update_sut(model)
            elif "routes" in logs:
                route_stats.update_mom(logs["routes"], batch.get("attention_mask"))
        clear_moe_statistics(model)  # eval batches must not accumulate aux statistics
        all_p.append(probs7.cpu()); all_y.append(batch["y"].cpu()); all_len.append(batch["len_bin"].cpu()); all_dup.append(batch["dup"].cpu())
        if invalid is not None:
            all_inv.append(invalid.cpu())
        if "depth_ex" in logs:
            all_depth.append(logs["depth_ex"].cpu())
    clear_moe_statistics(model)
    model.train()
    return summarize(torch.cat(all_p), torch.cat(all_y), torch.cat(all_len), torch.cat(all_inv) if all_inv else None,
                     torch.cat(all_dup), torch.cat(all_depth) if all_depth else None)


# ------------------------------------------------------------------ main entry
def run(cfg: dict, out_dir: Optional[Path] = None, device: Optional[str] = None, data_dir: Optional[Path] = None,
        evaluate_tests: bool = True) -> dict:
    tcfg, ocfg, lcfg, dcfg = cfg["train"], cfg["optim"], cfg.get("loss", {}), cfg["data"]
    seed = tcfg.get("seed", 0)
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(data_dir or paths.DATA_DIR / paths.DATA_VERSION)
    vocab = Vocab()
    model, interface, mcfg = build_model(cfg["model"], vocab)
    kind = cfg["model"]["kind"]

    init_report = None
    if cfg.get("init_from"):
        ckpt = torch.load(cfg["init_from"]["checkpoint"], map_location="cpu")
        sd = ckpt["model"]
        if cfg["init_from"].get("convert") == "mom":
            model, init_report = init_from_vanilla(model, sd, mcfg)
            init_report = {"copied": len(init_report["copied"]), "not_copied": init_report["not_copied"],
                           "donor_unused": init_report["donor_unused"]}
        else:
            model.load_state_dict(sd)
            init_report = {"loaded": "full state dict"}
    model.to(device)

    regime = dcfg.get("regime", "length")
    train = load_split(data_dir, regime, "train", interface, vocab, dcfg.get("train_limit"), seed)
    valid = load_split(data_dir, regime, "valid", interface, vocab, dcfg.get("valid_limit"), seed)

    max_steps = tcfg["max_steps"]
    opt = build_optimizer(model, ocfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(ocfg, max_steps))
    clip = ocfg.get("clip_norm", 0.0)

    run_name = f"{cfg['name']}_s{seed}_{config_hash(cfg)}"
    out_dir = Path(out_dir or paths.RUNS_DIR / run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "name": cfg["name"], "config": cfg, "config_hash": config_hash(cfg), "code_hash": code_hash(),
        "upstream": paths.UPSTREAM, "data_dir": str(data_dir), "seed": seed, "device": device,
        "torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform(),
        "n_params": sum(p.numel() for p in model.parameters()), "init_report": init_report,
        "metric_definitions": {"brier": "mean over examples and 7 classes of (p - onehot)^2",
                               "selection": "min validation brier"},
    }
    (out_dir / "record.json").write_text(json.dumps(record, indent=1, default=str))
    log_f = open(out_dir / "log.jsonl", "a", encoding="utf-8")

    rng = random.Random(seed)
    step, best, best_step, bad_evals = 0, float("inf"), -1, 0
    eval_every = tcfg.get("eval_every", 500)
    patience = tcfg.get("patience")
    t0 = time.time()
    model.train()
    while step < max_steps:
        for batch in train.batches(tcfg["batch_size"], shuffle=True, rng=rng):
            batch = to_device(batch, device)
            loss, logs, _, _ = forward_loss(model, kind, batch, lcfg, vocab, training=True)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {logs}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip and clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            sched.step()
            step += 1
            if step % tcfg.get("log_every", 50) == 0:
                logs.pop("routes", None)
                logs.pop("depth_ex", None)
                log_f.write(json.dumps({"step": step, "loss": loss.item(), "lr": sched.get_last_lr()[0],
                                        "time": time.time() - t0, **logs}) + "\n")
                log_f.flush()
            if step % eval_every == 0 or step == max_steps:
                vm = evaluate(model, kind, valid, lcfg, vocab, device, tcfg.get("eval_batch_size", 512))
                log_f.write(json.dumps({"step": step, "valid": {k: v for k, v in vm.items() if k != "confusion"}}) + "\n")
                log_f.flush()
                if vm["brier"] < best:
                    best, best_step, bad_evals = vm["brier"], step, 0
                    torch.save({"model": model.state_dict(), "step": step, "valid": vm}, out_dir / "best.pt")
                else:
                    bad_evals += 1
                if patience and bad_evals >= patience:
                    max_steps = step
            if step >= max_steps:
                break
    log_f.close()

    result = {"best_step": best_step, "best_valid_brier": best, "train_time_s": time.time() - t0}
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    rs = RouteStats()
    result["valid"] = evaluate(model, kind, valid, lcfg, vocab, device, route_stats=rs)
    result["routing"] = rs.summary()
    if evaluate_tests:
        result["test"] = {}
        for part in cfg.get("eval_parts", _default_parts(regime)):
            data = load_split(data_dir, regime, part, interface, vocab)
            result["test"][part] = evaluate(model, kind, data, lcfg, vocab, device)
    (out_dir / "result.json").write_text(json.dumps(result, indent=1))
    return result


def _default_parts(regime: str):
    if regime == "length":
        return ["test_iid", "test_len"]
    return ["test_iid", "test_comp_seenlen", "test_comp_newlen"]
