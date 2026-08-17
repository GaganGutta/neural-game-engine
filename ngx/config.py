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
