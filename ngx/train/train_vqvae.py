"""Stage 2: train the frame tokenizer.

    python -m ngx.train.train_vqvae --config configs/small.yaml

Writes ``runs/<name>/tokenizer/{vqvae.pt, recon_*.png}``. The recon PNGs are the
sanity check the plan calls for: top row is ground truth, bottom row is the
frame after a round trip through 64 discrete tokens.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import load_config, pick_device, run_dir
from ..data import FrameDataset
from ..models.vqvae import VQVAE, psnr
from .common import (
    Timer,
    amp_context,
    cosine_warmup,
    count_params,
    human,
    infinite,
    save_ckpt,
    save_grid,
)


@torch.no_grad()
def evaluate(model: VQVAE, loader: DataLoader, device: torch.device, max_batches: int = 20):
    model.eval()
    tot_psnr, tot_perp, n = 0.0, 0.0, 0
    for i, x in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        x_hat, _, stats = model(x)
        tot_psnr += psnr(x_hat, x)
        tot_perp += stats["perplexity"]
        n += 1
    model.train()
    return tot_psnr / max(n, 1), tot_perp / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--set", nargs="*", default=[], help="dotted overrides, k=v")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    tc, dc = cfg["tokenizer"], cfg["data"]
    device = pick_device(args.device)
    out = run_dir(cfg, "tokenizer")
    torch.manual_seed(cfg.get("seed", 0))

    train_ds = FrameDataset(dc["root"], "train")
    val_ds = FrameDataset(dc["root"], "val")
    nw = cfg.get("num_workers", 4)
    train_dl = DataLoader(
        train_ds, batch_size=tc["batch_size"], shuffle=True, num_workers=nw,
        drop_last=True, persistent_workers=nw > 0, pin_memory=device.type == "cuda",
    )
    val_dl = DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=False, num_workers=0)

    model = VQVAE(
        ch=tc["ch"], embed_dim=tc["embed_dim"], num_codes=tc["num_codes"],
        n_res=tc.get("n_res", 2), commit_beta=tc.get("commit_beta", 0.25),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], betas=(0.9, 0.95), weight_decay=1e-4)
    autocast, scaler = amp_context(device, cfg.get("amp", False))

    steps = tc["steps"]
    warmup = tc.get("warmup", max(steps // 50, 100))
    print(
        f"tokenizer: {human(count_params(model))} params | {len(train_ds):,} train frames "
        f"| {len(val_ds):,} val | {steps:,} steps | device={device}"
    )

    # Fixed val batch so the recon PNGs are comparable across checkpoints.
    # Spread across the split rather than taking the first eight, which would be
    # eight consecutive frames of the same room.
    picks = np.linspace(0, len(val_ds) - 1, 8).astype(int)
    fixed = torch.stack([val_ds[int(i)] for i in picks]).to(device)

    timer = Timer()
    it = infinite(train_dl)
    for step in range(steps):
        lr = cosine_warmup(step, steps, warmup, tc["lr"], tc["lr"] * 0.05)
        for g in opt.param_groups:
            g["lr"] = lr

        x = next(it).to(device, non_blocking=True)
        with autocast:
            _, loss, stats = model(x)
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if step % cfg.get("log_every", 100) == 0:
            print(
                f"step {step:6d}/{steps} loss {loss.item():.4f} recon {stats['recon']:.4f} "
                f"commit {stats['commit']:.4f} perp {stats['perplexity']:6.1f} "
                f"codes {stats['codes_used']:4d} lr {lr:.2e} "
                f"| {timer.rate(step + 1):.1f} it/s",
                flush=True,
            )
        if step and step % cfg.get("eval_every", 2000) == 0:
            pv, perp = evaluate(model, val_dl, device)
            with torch.no_grad():
                rec, _, _ = model(fixed)
            save_grid(os.path.join(out, f"recon_{step:06d}.png"), [fixed, rec])
            print(f"  [val] step {step} psnr {pv:.2f} dB  perplexity {perp:.1f}", flush=True)

    pv, perp = evaluate(model, val_dl, device)
    with torch.no_grad():
        rec, _, _ = model(fixed)
    save_grid(os.path.join(out, "recon_final.png"), [fixed, rec])
    save_ckpt(
        os.path.join(out, "vqvae.pt"), model, cfg,
        val_psnr=pv, perplexity=perp, steps=steps,
    )
    print(
        f"done in {timer.elapsed / 60:.1f} min | val PSNR {pv:.2f} dB | perplexity {perp:.1f} "
        f"| saved {os.path.join(out, 'vqvae.pt')}"
    )


if __name__ == "__main__":
    main()
