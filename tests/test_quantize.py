"""int8 weight accounting: the benchmark table's weights column depends on it."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ngx.infer.quantize import quantize_int8, weight_bytes  # noqa: E402


def test_int8_weights_are_counted_at_one_byte_each():
    model = nn.Sequential(nn.Embedding(10, 32), nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
    fp32 = weight_bytes(model)
    assert fp32 == 4 * sum(p.numel() for p in model.parameters())

    q = quantize_int8(model)
    linear_w = 32 * 64 + 64 * 32           # int8, one byte each
    linear_b = 4 * (64 + 32)               # biases stay float32
    embedding = 4 * 10 * 32                # embeddings are not quantised
    assert weight_bytes(q) == linear_w + linear_b + embedding
    assert weight_bytes(q) < fp32 / 2
