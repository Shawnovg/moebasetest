"""Load a dataset written by data_gen/build_all.py and serve padded batches.

``Dataset(name, split)`` keeps the whole split in memory (int64 tensors). ``train_batches`` yields
random batches (with replacement, seeded) forever; ``eval_batches`` walks a split in order.
Every batch is trimmed to its longest sequence. ``pad_mask`` is True at padding.

``DynamicsDataset(name, split)`` serves the switching-dynamics task (data_gen/dynamics.py): a batch is
(inputs B x 64 x 4, targets B x 64 x 4, regimes B x 64), states normalised with the training mean/std
from manifest.json; target t is state t+1, regimes[:, t] the hidden regime of that transition.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .paths import DATA_ROOT

PAD = 0


def load_vocab(root: Path = DATA_ROOT):
    return json.loads((Path(root) / "vocab.json").read_text())


class Dataset:
    def __init__(self, name: str, split: str, root: Path = DATA_ROOT):
        z = np.load(Path(root) / name / f"{split}.npz")
        self.tokens = torch.from_numpy(z["tokens"].astype(np.int64))
        self.targets = torch.from_numpy(z["targets"].astype(np.int64))
        self.lengths = torch.from_numpy(z["lengths"].astype(np.int64))
        self.name, self.split = name, split

    def __len__(self):
        return self.tokens.shape[0]

    def batch(self, idx):
        T = int(self.lengths[idx].max())
        tok, tgt = self.tokens[idx, :T], self.targets[idx, :T]
        return tok, tgt, tok == PAD

    def train_batches(self, batch_size: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        while True:
            yield self.batch(torch.randint(0, len(self), (batch_size,), generator=g))

    def eval_batches(self, batch_size: int, max_batches: int | None = None):
        n = len(self) if max_batches is None else min(len(self), batch_size * max_batches)
        for i in range(0, n, batch_size):
            yield self.batch(torch.arange(i, min(i + batch_size, n)))


class DynamicsDataset:
    def __init__(self, name: str, split: str, root: Path = DATA_ROOT):
        man = json.loads((Path(root) / name / "manifest.json").read_text())
        z = np.load(Path(root) / name / f"{split}.npz")
        self.mean = torch.tensor(man["mean"], dtype=torch.float32)
        self.std = torch.tensor(man["std"], dtype=torch.float32)
        s = (torch.from_numpy(z["states"]) - self.mean) / self.std
        self.inputs, self.targets = s[:, :-1].contiguous(), s[:, 1:].contiguous()
        self.raw = torch.from_numpy(z["states"])
        self.regimes = torch.from_numpy(z["regimes"].astype(np.int64))
        self.manifest, self.name, self.split = man, name, split

    def __len__(self):
        return self.inputs.shape[0]

    def batch(self, idx):
        return self.inputs[idx], self.targets[idx], self.regimes[idx]

    def train_batches(self, batch_size: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        while True:
            yield self.batch(torch.randint(0, len(self), (batch_size,), generator=g))

    def eval_batches(self, batch_size: int, max_batches: int | None = None):
        n = len(self) if max_batches is None else min(len(self), batch_size * max_batches)
        for i in range(0, n, batch_size):
            yield self.batch(torch.arange(i, min(i + batch_size, n)))
