"""Stage 3: train the action-conditioned dynamics model.

    python -m ngx.data.tokenize        --config configs/small.yaml
    python -m ngx.train.train_dynamics --config configs/small.yaml

Writes ``runs/<name>/dynamics/{dynamics.pt, ckpt_*.pt, pred_*.png}``.

Two validation numbers are tracked because they measure different things:

``masked acc``
    token accuracy under the same cosine masking used in training. Optimistic:
    most of the frame is usually visible.
``cold acc``
    token accuracy with the *entire* next frame masked, which is exactly the
    first iteration of MaskGIT decoding at play time. This is the number that
    predicts whether rollouts hold together.

Stopping, for the scaling ladder
--------------------------------
Three ways a run ends, all optional and composable, first one hit wins:

``dynamics.steps``
    a hard cap, and the horizon the cosine schedule is planned over.
``dynamics.token_budget``
    stop once this many tokens have been seen. "Tokens seen" is defined as
    ``windows x context x 65`` -- the stream-A tokens (64 frame tokens plus one
    action slot per context frame). This is the number matched across ladder
    rungs; the stream-B supervision copies are not data and are not counted.
``dynamics.plateau_patience``
    stop after this many consecutive evals without the held-out loss improving
    by at least ``dynamics.min_delta``. This is what "trained to convergence"
    means for the 2M rung: a plateau, not a step count.

Every checkpoint carries ``tokens_seen``, ``windows_seen``, ``epochs`` and
``seed``, so a ladder table can be built from checkpoints alone. Besides the
best-so-far ``dynamics.pt``, a step-numbered ``ckpt_<step>.pt`` is written
every ``checkpoint_every`` steps so a sync loop pulling the run directory
loses minutes, not the run, on a preemption.
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
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)

    vq, tc = build_tokenizer(cfg, device)
    ctx = dyn["context"]
    preload = bool(data.get("preload", False))
    train_ds = TokenSequenceDataset(data["root"], ctx, "train", preload=preload)
    val_ds = TokenSequenceDataset(data["root"], ctx, "val", preload=preload)
    # Preloaded data needs no workers; memmapped data gets a couple.
    nw = 0 if preload else min(cfg.get("num_workers", 4), 2)
    train_dl = DataLoader(train_ds, batch_size=dyn["batch_size"], shuffle=True,
                          num_workers=nw, drop_last=True, persistent_workers=nw > 0,
                          pin_memory=device.type == "cuda")
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

    # -- stopping plan ------------------------------------------------------
    # Tokens-seen accounting: stream-A tokens per window (matched across rungs).
    window_tokens = ctx * (vq.tokens_per_frame + 1)
    batch_tokens = dyn["batch_size"] * window_tokens
    token_budget = dyn.get("token_budget")
    steps = dyn.get("steps")
    if steps is None:
        if token_budget is None:
            raise ValueError("dynamics needs `steps`, `token_budget`, or both")
        steps = int(np.ceil(token_budget / batch_tokens))
    elif token_budget is not None:
        # The budget can only shorten the planned horizon, never extend it.
        steps = min(steps, int(np.ceil(token_budget / batch_tokens)))
    patience = dyn.get("plateau_patience")          # evals, not steps
    min_delta = float(dyn.get("min_delta", 0.005))
    eval_every = cfg.get("eval_every", 2000)
    ckpt_every = int(dyn.get("checkpoint_every", eval_every))
    warmup = dyn.get("warmup", max(steps // 50, 100))

    seq_len = ctx * (2 * vq.tokens_per_frame + 1)
    print(
        f"dynamics: {human(count_params(model))} params | seq {seq_len} tokens | "
        f"context {ctx} frames | pos {model.pos_encoding} | seed {seed} | "
        f"{len(train_ds):,} train windows | {len(val_ds):,} val | "
        f"plan {steps:,} steps"
        + (f" | budget {token_budget:,} tokens" if token_budget else "")
        + (f" | plateau patience {patience}" if patience else "")
        + f" | device={device}",
        flush=True,
    )

    def extras(step):
        windows = step * dyn["batch_size"]
        return dict(
            seed=seed, step=step, windows_seen=windows,
            tokens_seen=windows * window_tokens,
            epochs=windows / max(len(train_ds), 1),
            action_names=train_ds.meta["action_names"],
        )

    timer = Timer()
    it = infinite(train_dl)
    best = float("inf")
    evals_since_best = 0
    stop_reason = "step cap"
    step = 0
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
        if step and step % ckpt_every == 0:
            save_ckpt(os.path.join(out, f"ckpt_{step:07d}.pt"), model, cfg, **extras(step))
        if step and step % eval_every == 0:
            vl, ma, ca = evaluate(model, val_dl, device)
            sample_grid(model, vq, val_ds, device, os.path.join(out, f"pred_{step:06d}.png"))
            print(f"  [val] step {step} loss {vl:.4f} masked acc {ma:.3f} cold acc {ca:.3f} "
                  f"tokens {extras(step)['tokens_seen']:,} "
                  f"epochs {extras(step)['epochs']:.2f}",
                  flush=True)
            if vl < best - min_delta:
                best = vl
                evals_since_best = 0
                save_ckpt(os.path.join(out, "dynamics.pt"), model, cfg,
                          val_loss=vl, cold_acc=ca, **extras(step))
            else:
                evals_since_best += 1
                if patience and evals_since_best >= patience:
                    stop_reason = f"plateau ({patience} evals without {min_delta} improvement)"
                    break
    else:
        stop_reason = ("token budget" if token_budget
                       and (steps * batch_tokens) >= token_budget else "step cap")

    vl, ma, ca = evaluate(model, val_dl, device)
    sample_grid(model, vq, val_ds, device, os.path.join(out, "pred_final.png"))
    # Keep dynamics.pt as the best-by-val-loss model; the final state is only
    # promoted when it is the best we have seen.
    if vl < best:
        save_ckpt(os.path.join(out, "dynamics.pt"), model, cfg,
                  val_loss=vl, cold_acc=ca, **extras(step + 1))
    save_ckpt(os.path.join(out, "final.pt"), model, cfg,
              val_loss=vl, cold_acc=ca, **extras(step + 1))
    e = extras(step + 1)
    print(
        f"stopped: {stop_reason} | {timer.elapsed / 60:.1f} min | "
        f"val loss {vl:.4f} (best {min(best, vl):.4f}) | masked acc {ma:.3f} | "
        f"cold acc {ca:.3f} | tokens {e['tokens_seen']:,} | epochs {e['epochs']:.2f} | "
        f"saved {os.path.join(out, 'dynamics.pt')}"
    )


if __name__ == "__main__":
    main()
