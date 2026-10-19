"""Tensors for the three task interfaces (classifier / seq2seq / causal)."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import torch

from ..data import logic


class LogicTensors:
    def __init__(self, examples: List[logic.Example], interface: str, vocab: logic.Vocab):
        self.examples = examples
        self.interface = interface
        self.vocab = vocab
        self.items = [self._encode(e) for e in examples]

    def _encode(self, e):
        v = self.vocab
        if self.interface == "classifier":
            ids, y = logic.encode_classifier(e, v)
            return {"input": ids, "y": y, "len_bin": e.length_bin, "dup": e.swap_dup}
        if self.interface == "seq2seq":
            src, prev, tgt, y = logic.encode_seq2seq(e, v)
            return {"src": src, "prev": prev, "tgt": tgt, "y": y, "len_bin": e.length_bin, "dup": e.swap_dup}
        if self.interface == "causal":
            ids, pos, y = logic.encode_causal(e, v)
            return {"input": ids, "pos": pos, "y": y, "len_bin": e.length_bin, "dup": e.swap_dup}
        raise ValueError(self.interface)

    def __len__(self):
        return len(self.items)

    def _len(self, i):
        it = self.items[i]
        return len(it["src"]) if "src" in it else len(it["input"])

    def batches(self, batch_size: int, shuffle: bool, rng: Optional[random.Random] = None,
                bucket: bool = True, bucket_batches: int = 50):
        """Length-bucketed batches (as fairseq does): shuffle, sort within windows of `bucket_batches`
        batches by length, then shuffle the batch order. Evaluation (shuffle=False) sorts globally by length;
        metrics are order-free because predictions and labels are kept aligned."""
        rng = rng or random
        order = list(range(len(self.items)))
        if shuffle:
            rng.shuffle(order)
            if bucket:
                w = batch_size * bucket_batches
                order = [j for k in range(0, len(order), w) for j in sorted(order[k:k + w], key=self._len)]
        elif bucket:
            order.sort(key=self._len)
        chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        if shuffle:
            rng.shuffle(chunks)
        for c in chunks:
            yield self.collate([self.items[j] for j in c])

    def collate(self, items: List[dict]) -> Dict[str, torch.Tensor]:
        pad = self.vocab.pad
        out = {"y": torch.tensor([it["y"] for it in items]), "len_bin": torch.tensor([it["len_bin"] for it in items]),
               "dup": torch.tensor([bool(it["dup"]) for it in items])}

        def pad_to(seqs, value=pad):
            m = max(len(s) for s in seqs)
            return torch.tensor([s + [value] * (m - len(s)) for s in seqs])

        if self.interface == "classifier":
            out["input"] = pad_to([it["input"] for it in items])
            out["attention_mask"] = out["input"].ne(pad).float()
        elif self.interface == "seq2seq":
            out["src"] = pad_to([it["src"] for it in items])
            out["prev"] = pad_to([it["prev"] for it in items])
            out["tgt"] = pad_to([it["tgt"] for it in items])
        else:
            ids = pad_to([it["input"] for it in items])
            out["input"] = ids
            out["attention_mask"] = ids.ne(pad).float()
            out["pos"] = torch.tensor([it["pos"] for it in items])
            labels = torch.full_like(ids, -100)
            labels[torch.arange(len(items)), out["pos"]] = ids[torch.arange(len(items)), out["pos"]]
            out["labels"] = labels  # loss only on the relation token (task adaptation, see audit)
        return out


def load_split(data_dir: Path, regime: str, part: str, interface: str, vocab: logic.Vocab,
               limit: Optional[int] = None, seed: int = 0) -> LogicTensors:
    exs = logic.read_jsonl(Path(data_dir) / regime / f"{part}.jsonl")
    if limit is not None and limit < len(exs):
        rng = random.Random(seed)
        exs = rng.sample(exs, limit)
    return LogicTensors(exs, interface, vocab)
