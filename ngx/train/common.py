"""Shared training plumbing."""

from __future__ import annotations

import contextlib
import math
import os
import time

import cv2
import numpy as np
import torch


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def human(n: int) -> str:
    for unit, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= scale:
            return f"{n / scale:.1f}{unit}"
    return str(n)


def cosine_warmup(step: int, total: int, warmup: int, base_lr: float, min_lr: float = 0.0) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * min(t, 1.0)))


def amp_context(device: torch.device, enabled: bool):
    """bf16 autocast where it is available, otherwise a no-op.

    bf16 rather than fp16 because it needs no loss scaler, and the dynamics
    model's logits over a 512-entry codebook sit comfortably in its range.
    """
    if not enabled:
        return contextlib.nullcontext(), None
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.autocast("cuda", dtype=torch.bfloat16), None
        return torch.autocast("cuda", dtype=torch.float16), torch.amp.GradScaler("cuda")
    return torch.autocast("cpu", dtype=torch.bfloat16), None


def infinite(loader):
    while True:
        yield from loader


def save_ckpt(path: str, model: torch.nn.Module, cfg: dict, **extra) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg, **extra}, path)


def load_ckpt(path: str, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """``(B, 3, H, W)`` in [-1, 1] -> ``(B, H, W, 3)`` uint8 RGB."""
    x = ((x.detach().float().clamp(-1, 1) + 1) * 127.5).round().byte()
    return x.permute(0, 2, 3, 1).cpu().numpy()


def save_grid(path: str, rows: list[torch.Tensor]) -> None:
    """Stack rows of images into one PNG; each row is ``(B, 3, H, W)``."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    strips = [np.concatenate(list(to_uint8(r)), axis=1) for r in rows]
    grid = np.concatenate(strips, axis=0)
    cv2.imwrite(path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


class Timer:
    def __init__(self) -> None:
        self.t0 = time.time()

    def rate(self, n: int) -> float:
        return n / max(time.time() - self.t0, 1e-9)

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0
