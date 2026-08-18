"""B0.5a: can the architecture memorise a single batch?

    python -m ngx.train.overfit --config configs/small.yaml

The question this answers is "broken or starved", and it is the only cheap way
to answer it. A model that cannot drive training loss to ~0 and token accuracy
to ~1.0 on four fixed windows, given unlimited steps on those four windows, has
a defect in the model, the loss, or the data pipeline. Scaling a defective model
on rented hardware just buys a more expensive wrong answer.

Nothing is made artificially easy. The batch is real held-out-format data from
the tokenized dataset, the masking is the same random cosine schedule used in
real training, and the model is built from the same config. The only change is
that the batch never varies.

Two accuracies are tracked because they answer different questions:

``masked``
    accuracy on the randomly-masked positions, i.e. the training objective.
``cold``
    accuracy with the *entire* target frame masked, which is what the first
    MaskGIT pass faces at play time. Memorising under partial masking while
    failing cold would point at the model leaning on visible neighbours inside
    the target frame rather than on the context and action.
"""

from __future__ import annotations

import argparse
import os

import torch

from ..config import find_ckpt, load_config, pick_device
from ..data import TokenSequenceDataset
from ..models.dynamics import DynamicsTransformer
from ..models.vqvae import VQVAE
from .common import Timer, count_params, human, load_ckpt


@torch.no_grad()
def cold_accuracy(model, tokens, actions) -> float:
    """Token accuracy with the whole final frame hidden."""
    mask = torch.zeros(tokens.shape[0], tokens.shape[1] - 1, model.L,
                       dtype=torch.bool, device=tokens.device)
    mask[:, -1] = True
    logits = model(tokens, actions, mask)[:, -1]
    return float((logits.argmax(-1) == tokens[:, -1]).float().mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--device", default="auto")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--target-acc", type=float, default=0.99,
                   help="masked accuracy counted as memorised; recorded, not stopped on")
    p.add_argument("--out", default="docs/OVERFIT.md")
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    device = pick_device(a.device)
    torch.manual_seed(0)

    tc = load_ckpt(find_ckpt(cfg, "tokenizer", "vqvae.pt"), map_location="cpu")["cfg"]["tokenizer"]

    dyn = cfg["dynamics"]
    ds = TokenSequenceDataset(cfg["data"]["root"], dyn["context"], "train")
    picks = [ds[i] for i in range(a.batch)]
    tokens = torch.stack([q[0] for q in picks]).to(device)
    actions = torch.stack([q[1] for q in picks]).to(device)

    model = DynamicsTransformer(
        num_codes=tc["num_codes"], num_actions=ds.meta["num_actions"],
        tokens_per_frame=VQVAE(ch=tc["ch"], embed_dim=tc["embed_dim"],
                               num_codes=tc["num_codes"]).tokens_per_frame,
        context=dyn["context"], d_model=dyn["d_model"], n_layers=dyn["n_layers"],
        n_heads=dyn["n_heads"], dropout=0.0,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)

    print(f"overfitting {a.batch} windows | {human(count_params(model))} params | "
          f"context {dyn['context']} | lr {a.lr} | device {device}")
    curve, timer = [], Timer()
    hit = None
    for step in range(a.steps + 1):
        loss, stats = model.loss(tokens, actions)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if hit is None and stats["token_acc"] >= a.target_acc:
            hit = step
        # Always record the last step, so the verdict is never computed from a
        # stale earlier sample.
        if step % a.log_every == 0 or step == a.steps:
            model.eval()
            cold = cold_accuracy(model, tokens, actions)
            model.train()
            curve.append((step, stats["loss"], stats["token_acc"], cold))
            print(f"  step {step:5d}  loss {stats['loss']:7.4f}  "
                  f"masked acc {stats['token_acc']:.3f}  cold acc {cold:.3f}")

    step, final_loss, final_acc, final_cold = curve[-1]
    ok = final_acc >= 0.95 and final_loss < 0.5
    reached = f"Masked accuracy first cleared {a.target_acc} at step {hit}. " if hit is not None else ""
    verdict = reached + (
        "**PASS.** The architecture can memorise a batch, so the loss, the masking, "
        "the attention mask and the data pipeline are wired correctly. A weak model on "
        "the full dataset is an undertrained or undersized model, not a broken one."
        if ok else
        "**FAIL.** The architecture cannot memorise four windows given unlimited steps on "
        "them. Something is wrong in the model, the loss, or the data pipeline, and no "
        "amount of scaling fixes that. Do not rent a GPU until this is found."
    )
    print(f"\n{'PASS' if ok else 'FAIL'}: loss {final_loss:.4f}, masked acc {final_acc:.3f}, "
          f"cold acc {final_cold:.3f} after {step} steps ({timer.elapsed / 60:.1f} min)")

    lines = [
        "# B0.5a: overfit one batch",
        "",
        f"{a.batch} fixed windows from the tokenized dataset, "
        f"{human(count_params(model))}-parameter model built from `{a.config}`, "
        f"lr {a.lr}, same random cosine masking as real training. The batch never changes.",
        "",
        "| step | train loss | masked acc | cold acc |",
        "|---|---|---|---|",
    ]
    for s, l, acc, cold in curve:
        lines.append(f"| {s} | {l:.4f} | {acc:.3f} | {cold:.3f} |")
    lines += [
        "",
        verdict,
        "",
        "`masked acc` is accuracy on the randomly masked positions, i.e. the training "
        "objective. `cold acc` hides the entire target frame, which is what the first "
        "MaskGIT pass faces at play time; it is the harder number and it is reported so "
        "that memorising via visible neighbours inside the target frame cannot pass as "
        "learning the dynamics.",
        "",
        "Regenerate with `python -m ngx.train.overfit --config configs/small.yaml`.",
        "",
    ]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
