"""The retrieval-accuracy numbers in DRIFT.md are computed, and computed right.

Frames and poses are synthetic and the tokenizer is replaced by a lookup, so
every expected outcome can be read off the setup: one revisit the key gets
right, one it gets wrong because a different place looks more similar, and one
where the only frames of that place are still inside the exclusion window.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ngx.eval import baselines  # noqa: E402
from ngx.eval.drift import retrieval_accuracy  # noqa: E402

A, B, C = (0.0, 0.0, 0.0), (1000.0, 0.0, 0.0), (0.0, 1000.0, 0.0)


def _world():
    #          frames      code  place
    layout = [(range(0, 10), 0, A),   # the place revisited later
              (range(10, 13), 5, B),  # a different place that looks like frame 38
              (range(13, 28), 1, B),
              (range(28, 35), 3, C),  # a place seen only recently
              (range(35, 38), 0, A),
              (range(38, 40), 4, A)]
    codes = np.zeros(40, dtype=np.int64)
    poses = np.zeros((40, 3))
    for frames, code, place in layout:
        for t in frames:
            codes[t] = code
            poses[t] = place
    frames = np.zeros((40, 64, 64, 3), dtype=np.uint8)
    frames[:, 0, 0, 0] = np.arange(40)  # lets the fake encoder recover the index

    embed = torch.zeros(6, 6)
    embed[0, 0] = 1.0
    embed[1, 2] = 1.0
    embed[3, 3] = 1.0
    embed[4, :2] = torch.tensor([0.8, 0.6])  # frame 38 looks exactly like the decoys...
    embed[5, :2] = torch.tensor([0.8, 0.6])  # ...and only 0.8 like its real place
    vq = SimpleNamespace(quantizer=SimpleNamespace(embed=embed))
    tokens = torch.as_tensor(codes)[:, None].repeat(1, 64)
    return vq, frames, poses, tokens


def test_retrieval_accuracy_counts_hits_misses_and_empty_memory(monkeypatch):
    vq, frames, poses, tokens = _world()
    monkeypatch.setattr(baselines, "encode",
                        lambda _vq, fr, _dev: tokens[torch.as_tensor(fr[:, 0, 0, 0].astype(np.int64))])
    pairs = [(5, 37),   # A again: same codes as frames 0-9, a clean hit
             (29, 32),  # C: frames 28-31 are among the 5 most recent writes, so nothing to find
             (6, 38)]   # A, but the decoys at B look more similar: a miss at top-1 and top-2
    r = retrieval_accuracy(vq, frames, poses, pairs, torch.device("cpu"),
                           write_every=1, exclude_recent=5, pos_tol=40.0, ang_tol=20.0)
    assert r["pairs"] == 2
    assert r["no_candidate"] == 1
    assert r["top1"] == 0.5
    assert r["top2"] == 0.5
