"""Stage 3: train the action-conditioned dynamics model.

    python -m ngx.data.tokenize        --config configs/small.yaml
    python -m ngx.train.train_dynamics --config configs/small.yaml

Writes ``runs/<name>/dynamics/{dynamics.pt, pred_*.png}``.

Two validation numbers are tracked because they measure different things:

``masked acc``
    token accuracy under the same cosine masking used in training. Optimistic:
    most of the frame is usually visible.
``cold acc``
    token accuracy with the *entire* next frame masked, which is exactly the
    first iteration of MaskGIT decoding at play time. This is the number that
    predicts whether rollouts hold together.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import find_ckpt, load_config, pick_device, run_dir
from ..data import TokenSequenceDataset
from ..models.dynamics import DynamicsTransformer
from ..models.vqvae import VQVAE
from .common import (
    Timer,
    amp_context,
    cosine_warmup,
    count_params,
    human,
    infinite,
    load_ckpt,
    save_ckpt,
    save_grid,
)


def build_tokenizer(cfg: dict, device: torch.device) -> tuple[VQVAE, dict]:
    ck = load_ckpt(find_ckpt(cfg, "tokenizer", "vqvae.pt"), map_location="cpu")
    tc = ck["cfg"]["tokenizer"]
    vq = VQVAE(ch=tc["ch"], embed_dim=tc["embed_dim"], num_codes=tc["num_codes"],
               n_res=tc.get("n_res", 2))
    vq.load_state_dict(ck["model"])
    return vq.to(device).eval(), tc


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 20):
    model.eval()
    tot_loss = tot_masked = tot_cold = n = 0.0
    for i, (tok, act) in enumerate(loader):
        if i >= max_batches:
            break
        tok, act = tok.to(device), act.to(device)
        loss, stats = model.loss(tok, act)
        tot_loss += stats["loss"]
        tot_masked += stats["token_acc"]

        # Cold start: hide the final frame completely.
        mask = torch.zeros(tok.shape[0], tok.shape[1] - 1, model.L, dtype=torch.bool, device=device)
        mask[:, -1] = True
        logits = model(tok, act, mask)[:, -1]
        tot_cold += (logits.argmax(-1) == tok[:, -1]).float().mean().item()
        n += 1
    model.train()
    k = max(n, 1)
    return tot_loss / k, tot_masked / k, tot_cold / k


@torch.no_grad()
def sample_grid(model, vq, ds, device, path: str, n: int = 8) -> None:
    """Ground truth on top, cold-start single-pass prediction below.

    Windows are spread across the split; consecutive ones would all show the
    same room.
    """
    model.eval()
    picks = np.linspace(0, len(ds) - 1, n).astype(int)
    batch = [ds[int(i)] for i in picks]
    tok = torch.stack([b[0] for b in batch]).to(device)
    act = torch.stack([b[1] for b in batch]).to(device)
    mask = torch.zeros(tok.shape[0], tok.shape[1] - 1, model.L, dtype=torch.bool, device=device)
    mask[:, -1] = True
    pred = model(tok, act, mask)[:, -1].argmax(-1)
    save_grid(path, [vq.decode_indices(tok[:, -1]), vq.decode_indices(pred)])
    model.train()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    dyn, data = cfg["dynamics"], cfg["data"]
    device = pick_device(args.device)
    out = run_dir(cfg, "dynamics")
    torch.manual_seed(cfg.get("seed", 0))

    vq, tc = build_tokenizer(cfg, device)
    ctx = dyn["context"]
    train_ds = TokenSequenceDataset(data["root"], ctx, "train")
    val_ds = TokenSequenceDataset(data["root"], ctx, "val")
    nw = min(cfg.get("num_workers", 4), 2)  # token windows are cheap to fetch
    train_dl = DataLoader(train_ds, batch_size=dyn["batch_size"], shuffle=True,
                          num_workers=nw, drop_last=True, persistent_workers=nw > 0)
    val_dl = DataLoader(val_ds, batch_size=dyn["batch_size"], shuffle=False, num_workers=0)

    model = DynamicsTransformer(
        num_codes=tc["num_codes"],
        num_actions=train_ds.meta["num_actions"],
        tokens_per_frame=vq.tokens_per_frame,
        context=ctx,
        d_model=dyn["d_model"], n_layers=dyn["n_layers"], n_heads=dyn["n_heads"],
        dropout=dyn.get("dropout", 0.0),
        pos_encoding=dyn.get("pos_encoding", "rope"),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=dyn["lr"], betas=(0.9, 0.95), weight_decay=0.01)
    autocast, scaler = amp_context(device, cfg.get("amp", False))
    steps = dyn["steps"]
    warmup = dyn.get("warmup", max(steps // 50, 100))
    seq_len = ctx * (2 * vq.tokens_per_frame + 1)
    print(
        f"dynamics: {human(count_params(model))} params | seq {seq_len} tokens | "
        f"context {ctx} frames | {len(train_ds):,} train windows | {len(val_ds):,} val | "
        f"{steps:,} steps | device={device}",
        flush=True,
    )

    timer = Timer()
    it = infinite(train_dl)
    best = float("inf")
    for step in range(steps):
        lr = cosine_warmup(step, steps, warmup, dyn["lr"], dyn["lr"] * 0.05)
        for g in opt.param_groups:
            g["lr"] = lr

        tok, act = next(it)
        tok, act = tok.to(device, non_blocking=True), act.to(device, non_blocking=True)
        with autocast:
            loss, stats = model.loss(tok, act)
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
                f"step {step:6d}/{steps} loss {stats['loss']:.4f} "
                f"masked_acc {stats['token_acc']:.3f} lr {lr:.2e} "
                f"| {timer.rate(step + 1):.2f} it/s",
                flush=True,
            )
        if step and step % cfg.get("eval_every", 2000) == 0:
            vl, ma, ca = evaluate(model, val_dl, device)
            sample_grid(model, vq, val_ds, device, os.path.join(out, f"pred_{step:06d}.png"))
            print(f"  [val] step {step} loss {vl:.4f} masked acc {ma:.3f} cold acc {ca:.3f}",
                  flush=True)
            if vl < best:
                best = vl
                save_ckpt(os.path.join(out, "dynamics.pt"), model, cfg,
                          val_loss=vl, cold_acc=ca, step=step,
                          action_names=train_ds.meta["action_names"])

    vl, ma, ca = evaluate(model, val_dl, device)
    sample_grid(model, vq, val_ds, device, os.path.join(out, "pred_final.png"))
    save_ckpt(os.path.join(out, "dynamics.pt"), model, cfg,
              val_loss=vl, cold_acc=ca, step=steps,
              action_names=train_ds.meta["action_names"])
    print(
        f"done in {timer.elapsed / 60:.1f} min | val loss {vl:.4f} | "
        f"masked acc {ma:.3f} | cold acc {ca:.3f} | saved {os.path.join(out, 'dynamics.pt')}"
    )


if __name__ == "__main__":
    main()
