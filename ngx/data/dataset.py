"""Datasets over a collected run.

Two consumers, two shapes:

* :class:`FrameDataset` -- single frames, for training the tokenizer.
* :class:`TokenSequenceDataset` -- windows of pre-tokenized frames plus the
  actions between them, for training the dynamics model.

The dynamics model reads tokens written once by ``ngx.data.tokenize`` rather
than re-encoding pixels every epoch. The tokenizer is frozen by then, and
encoding on the fly is a large slice of each step otherwise.

Train/val is split by *episode*, not by frame. Windows overlap heavily, so a
frame-level split would put near-duplicates of a validation window into train.

Big arrays are opened lazily. DataLoader workers pickle the dataset object, and
a live ``np.memmap`` attribute would be serialised in full.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

#: every VAL_EVERY-th episode is held out
VAL_EVERY = 20


def load_meta(root: str) -> dict:
    with open(os.path.join(root, "meta.json")) as f:
        return json.load(f)


def _episode_mask(episodes: np.ndarray, split: str) -> np.ndarray:
    is_val = (episodes % VAL_EVERY) == 0
    if split == "train":
        return ~is_val
    if split == "val":
        return is_val
    if split == "all":
        return np.ones_like(is_val)
    raise ValueError(f"unknown split {split!r}")


class _LazyArrays:
    """Mixin that memmaps ``<root>/<name>.npy`` on first touch, per process."""

    root: str

    def _arr(self, name: str) -> np.ndarray:
        cache = self.__dict__.setdefault("_mm", {})
        if name not in cache:
            cache[name] = np.load(os.path.join(self.root, f"{name}.npy"), mmap_mode="r")
        return cache[name]

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state.pop("_mm", None)  # reopened in the worker
        return state


class FrameDataset(_LazyArrays, Dataset):
    """Single frames as float tensors in [-1, 1], shape ``(3, S, S)``."""

    def __init__(self, root: str, split: str = "train") -> None:
        self.root = root
        self.meta = load_meta(root)
        episodes = np.load(os.path.join(root, "episodes.npy"))
        self.index = np.nonzero(_episode_mask(episodes, split))[0]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> torch.Tensor:
        x = np.asarray(self._arr("frames")[self.index[i]], dtype=np.float32)
        x = x / 127.5 - 1.0
        return torch.from_numpy(x).permute(2, 0, 1).contiguous()


class TokenSequenceDataset(_LazyArrays, Dataset):
    """Windows of ``context + 1`` tokenized frames.

    Returns ``(tokens, actions)`` where ``tokens`` is ``(T, L)`` int64 and
    ``actions`` is ``(T,)`` int64, with ``actions[t]`` the action applied at
    frame ``t``. The model predicts ``tokens[t + 1]`` from everything up to and
    including ``actions[t]``.
    """

    def __init__(self, root: str, context: int = 16, split: str = "train",
                 preload: bool = False) -> None:
        self.root = root
        self.meta = load_meta(root)
        self.context = context
        self.window = context + 1
        if preload:
            # Load tokens and actions fully into RAM (~260 MB for 2M frames).
            # On a GPU box the per-item memmap read is pure overhead, and a
            # preloaded dataset needs no DataLoader workers at all.
            self.__dict__["_mm"] = {
                "tokens": np.load(os.path.join(root, "tokens.npy")),
                "actions": np.load(os.path.join(root, "actions.npy")),
            }

        episodes = np.load(os.path.join(root, "episodes.npy"))
        n = len(episodes)
        if n < self.window:
            raise ValueError(f"need at least {self.window} frames, have {n}")
        # A window is valid when its first and last frame share an episode.
        # Episode ids are non-decreasing within a collection worker and jump
        # between workers, so equal endpoints imply the whole window is inside
        # one episode.
        starts = np.nonzero(
            episodes[: n - self.window + 1] == episodes[self.window - 1 :]
        )[0]
        self.index = starts[_episode_mask(episodes[starts], split)]
        if len(self.index) == 0:
            raise ValueError(f"no {split} windows of length {self.window} in {root}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        s = int(self.index[i])
        e = s + self.window
        tok = torch.from_numpy(np.asarray(self._arr("tokens")[s:e], dtype=np.int64))
        act = torch.from_numpy(np.asarray(self._arr("actions")[s:e], dtype=np.int64))
        return tok, act
