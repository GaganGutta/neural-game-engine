"""Tiny YAML config loader with dotted CLI overrides.

``--set dynamics.d_model=512 tokenizer.steps=1000`` beats maintaining a
parallel argparse surface for every knob.
"""

from __future__ import annotations

import os

import torch
import yaml


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override {ov!r} is not key=value")
        key, raw = ov.split("=", 1)
        *parts, last = key.split(".")
        node = cfg
        for p in parts:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(raw)
    return cfg


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_dir(cfg: dict, stage: str) -> str:
    d = os.path.join(cfg.get("run_root", "runs"), cfg.get("name", "default"), stage)
    os.makedirs(d, exist_ok=True)
    return d


def find_ckpt(cfg: dict, stage: str, filename: str) -> str:
    """Prefer a locally trained checkpoint, fall back to the one in the repo.

    Training writes to ``runs/<name>/<stage>/``. A fresh clone has no ``runs/``,
    so ``checkpoints/<name>/`` carries the weights that make ``python play.py``
    work without training anything first.

    Every consumer of a checkpoint goes through this. Three of them used to
    build the ``runs/`` path by hand, which meant a misplaced run directory
    turned into a FileNotFoundError in the middle of a long job rather than a
    graceful fall back to the shipped weights.
    """
    explicit = cfg.get(f"{stage}_ckpt")
    if explicit:
        if os.path.exists(explicit):
            return explicit
        raise FileNotFoundError(f"{stage}_ckpt is set to {explicit}, which does not exist")
    local = os.path.join(run_dir(cfg, stage), filename)
    if os.path.exists(local):
        return local
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shipped = os.path.join(root, "checkpoints", cfg.get("name", "default"), filename)
    if os.path.exists(shipped):
        return shipped
    raise FileNotFoundError(
        f"no checkpoint at {local} or {shipped}. Train it first -- see the README."
    )
