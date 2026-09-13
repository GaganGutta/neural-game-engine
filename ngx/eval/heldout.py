"""Held-out loss on a fixed validation set, independent of batch size and context.

    python -m ngx.eval.heldout --config configs/ladder/8m.yaml \\
        --set name=ladder-8m-t4-s0 --out runs/ladder-8m-t4-s0/heldout.json

The trainer's in-loop validation reads the first 20 batches of the unshuffled
validation split. How many windows that covers depends on the batch size
(10,240 at batch 512, 1,280 at batch 64), so rungs with different batch sizes
were scored on different, contiguous slices of a handful of episodes. That is
fine for choosing the best step inside one run and wrong for comparing rungs.

Here every checkpoint is scored on the same target frames: ``--targets``
frames spread evenly over the whole validation split, each with at least
``MAX_CONTEXT`` frames of history inside its episode, so a 6-, 12- or 24-frame
model predicts exactly the same frames, each from its own full context. Masks
come from seeded CPU generators with a fixed batch size, so the numbers do not
depend on the device or on how the run was trained.

``loss``
    the training objective itself (cosine masking over every target frame of
    the window). This is the held-out loss the pre-registration asks for, and
    the number the 8M learning-rate probe is decided on.
``last_loss``
    the same masked objective on the final frame only, which has the full
    context behind it. Its masks are shared across context lengths.
``cold_loss`` / ``cold_acc``
    the final frame hidden completely, which is what the first MaskGIT pass
    sees at play time. No randomness.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from ..config import find_ckpt, load_config, pick_device
from ..data.dataset import TokenSequenceDataset, load_meta
from ..models.dynamics import DynamicsTransformer
from ..train.common import load_ckpt

MAX_CONTEXT = 24
BATCH = 64
SEED = 0
TOKENS_PER_FRAME = 64


def target_frames(root: str, n: int, max_context: int = MAX_CONTEXT) -> np.ndarray:
    """Absolute indices of ``n`` validation frames, each with ``max_context``
    frames of in-episode history, spread evenly over the split."""
    ref = TokenSequenceDataset(root, max_context, "val")
    picks = np.unique(np.linspace(0, len(ref) - 1, n).round().astype(np.int64))
    return ref.index[picks].astype(np.int64) + max_context


def _cosine_mask(gen: torch.Generator, B: int, S: int, L: int) -> torch.Tensor:
    """Same schedule as ``DynamicsTransformer.sample_mask``, from a CPU generator."""
    u = torch.rand(B, S, generator=gen)
    n_mask = (torch.cos(u * math.pi / 2) * L).ceil().clamp(1, L).long()
    rank = torch.rand(B, S, L, generator=gen).argsort(-1).argsort(-1)
    return rank < n_mask.unsqueeze(-1)


def load_dynamics(path: str, root: str, device: torch.device):
    ck = load_ckpt(path, map_location="cpu")
    cfg, d = ck["cfg"], ck["cfg"]["dynamics"]
    tok = load_ckpt(find_ckpt(cfg, "tokenizer", "vqvae.pt"), map_location="cpu")["cfg"]["tokenizer"]
    model = DynamicsTransformer(
        num_codes=tok["num_codes"],
        num_actions=load_meta(root)["num_actions"],
        tokens_per_frame=TOKENS_PER_FRAME,
        context=d["context"],
        d_model=d["d_model"], n_layers=d["n_layers"], n_heads=d["n_heads"],
        dropout=0.0,
        pos_encoding=d.get("pos_encoding", "rope"),
    )
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


@torch.no_grad()
def evaluate(model: DynamicsTransformer, root: str, targets: np.ndarray, device: torch.device,
             batch: int = BATCH, seed: int = SEED) -> dict:
    C, L = model.context, model.L
    tokens = np.load(os.path.join(root, "tokens.npy"), mmap_mode="r")
    actions = np.load(os.path.join(root, "actions.npy"), mmap_mode="r")
    g_ctx = torch.Generator().manual_seed(seed)       # frames before the last: shape depends on C
    g_last = torch.Generator().manual_seed(seed + 1)  # the last frame: identical for every C
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda"
                else contextlib.nullcontext())

    s = dict(loss=0.0, n=0, last=0.0, n_last=0, cold=0.0, hit=0, n_cold=0)
    for i in range(0, len(targets), batch):
        t = targets[i:i + batch]
        idx = (t[:, None] - C) + np.arange(C + 1)[None]
        tok = torch.from_numpy(np.asarray(tokens[idx], dtype=np.int64)).to(device)
        act = torch.from_numpy(np.asarray(actions[idx], dtype=np.int64)).to(device)
        B = tok.shape[0]
        tgt = tok[:, 1:]

        mask = torch.cat([_cosine_mask(g_ctx, B, C - 1, L), _cosine_mask(g_last, B, 1, L)], dim=1).to(device)
        with autocast:
            logits = model(tok, act, mask)
        logits = logits.float()
        s["loss"] += F.cross_entropy(logits[mask], tgt[mask], reduction="sum").item()
        s["n"] += int(mask.sum())
        lm = mask[:, -1]
        s["last"] += F.cross_entropy(logits[:, -1][lm], tgt[:, -1][lm], reduction="sum").item()
        s["n_last"] += int(lm.sum())

        cold = torch.zeros_like(mask)
        cold[:, -1] = True
        with autocast:
            cl = model(tok, act, cold)[:, -1]
        cl = cl.float()
        s["cold"] += F.cross_entropy(cl.reshape(-1, cl.shape[-1]), tgt[:, -1].reshape(-1), reduction="sum").item()
        s["hit"] += int((cl.argmax(-1) == tgt[:, -1]).sum())
        s["n_cold"] += tgt[:, -1].numel()

    return {
        "loss": s["loss"] / max(s["n"], 1),
        "last_loss": s["last"] / max(s["n_last"], 1),
        "cold_loss": s["cold"] / max(s["n_cold"], 1),
        "cold_acc": s["hit"] / max(s["n_cold"], 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--ckpt", default=None, help="default: runs/<name>/dynamics/final.pt")
    p.add_argument("--targets", type=int, default=4096)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    root = cfg["data"]["root"]
    path = a.ckpt or os.path.join(cfg.get("run_root", "runs"), cfg["name"], "dynamics", "final.pt")
    device = pick_device(a.device)
    model, ck = load_dynamics(path, root, device)
    targets = target_frames(root, a.targets)
    r = evaluate(model, root, targets, device)
    r.update(
        name=cfg["name"], ckpt=path, context=int(model.context),
        params=int(sum(q.numel() for q in model.parameters())),
        tokens_seen=ck.get("tokens_seen"), epochs=ck.get("epochs"), step=ck.get("step"),
        n_targets=int(len(targets)), max_context=MAX_CONTEXT, batch=BATCH, seed=SEED,
    )
    print(json.dumps(r, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(r, f, indent=1)


if __name__ == "__main__":
    main()
