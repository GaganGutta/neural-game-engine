"""Encode a collected run to tokens once, so training never re-encodes pixels.

    python -m ngx.data.tokenize --config configs/small.yaml

Writes ``<data.root>/tokens.npy`` of shape ``(N, 64)`` uint16. The tokenizer is
frozen from here on, so this is a one-shot cost rather than part of every epoch.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from numpy.lib.format import open_memmap

from ..config import load_config, pick_device, run_dir
from ..models.vqvae import VQVAE
from ..train.common import Timer, load_ckpt


@torch.no_grad()
def tokenize(root: str, ckpt_path: str, device: torch.device, batch: int = 256) -> str:
    ck = load_ckpt(ckpt_path, map_location="cpu")
    tc = ck["cfg"]["tokenizer"]
    model = VQVAE(
        ch=tc["ch"], embed_dim=tc["embed_dim"], num_codes=tc["num_codes"],
        n_res=tc.get("n_res", 2),
    )
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    frames = np.load(os.path.join(root, "frames.npy"), mmap_mode="r")
    n = len(frames)
    if tc["num_codes"] > np.iinfo(np.uint16).max:
        raise ValueError("codebook too large for uint16 token storage")

    out_path = os.path.join(root, "tokens.npy")
    tokens = open_memmap(out_path, mode="w+", dtype="uint16", shape=(n, model.tokens_per_frame))

    timer = Timer()
    for i in range(0, n, batch):
        chunk = np.asarray(frames[i : i + batch], dtype=np.float32) / 127.5 - 1.0
        x = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(device)
        tokens[i : i + batch] = model.encode_indices(x).cpu().numpy().astype(np.uint16)
        if i % (batch * 40) == 0:
            done = i + len(chunk)
            print(f"  {done:,}/{n:,}  {timer.rate(done):,.0f} frames/s", flush=True)
    tokens.flush()

    used = len(np.unique(np.asarray(tokens[:: max(n // 50_000, 1)])))
    print(
        f"wrote {out_path}  ({n:,} x {model.tokens_per_frame} tokens, "
        f"{used}/{tc['num_codes']} codes seen, {timer.elapsed:.0f}s)"
    )
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--device", default="auto")
    p.add_argument("--ckpt", default=None, help="defaults to the run's tokenizer/vqvae.pt")
    p.add_argument("--batch", type=int, default=256)
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    ckpt = a.ckpt or os.path.join(run_dir(cfg, "tokenizer"), "vqvae.pt")
    tokenize(cfg["data"]["root"], ckpt, pick_device(a.device), a.batch)


if __name__ == "__main__":
    main()
