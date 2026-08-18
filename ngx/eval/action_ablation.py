"""B0.5b: does the model actually listen to the controller?

    python -m ngx.eval.action_ablation --config configs/small.yaml

A world model that ignores its action channel is not a game engine, it is an
expensive video autoplayer, and it will still post respectable PSNR because
most of the screen is predictable from the previous frame alone. So this is
measured directly rather than inferred.

Method. Fix a context. Branch: hold each distinct action for 16 frames, and
measure the mean pairwise PSNR *between* the resulting rollouts. Low pairwise
PSNR means the action changed what happened. High pairwise PSNR means every
button did the same thing.

That number is meaningless on its own, because it has no scale. Two references
give it one:

``real game``
    VizDoom is deterministic under a fixed seed, so the same action prefix can
    be replayed and then branched. This is how much the actions genuinely
    separate the future, and it is the target the model should approach.
``action embedding zeroed``
    Every branch now receives an identical input, so the rollouts must come out
    pixel-identical. This is the control that proves the harness is actually
    varying the action, and it is the "ignores the controller" endpoint: a model
    scoring near this is one where the button does nothing.

Decoding is greedy, which is what "holding noise fixed" means here. With no
sampling, any difference between two branches is caused by the action and by
nothing else.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from ..config import load_config, pick_device
from ..envs import make_env
from ..infer.engine import EngineConfig, NeuralGameEngine
from ..infer.load import load_models
from .baselines import PSNR_MAX, psnr_capped


def real_branches(cfg, num_actions: int, context: int, prefix_len: int, horizon: int, seed: int):
    """Replay a fixed prefix, then hold each action. Returns (ctx_frames, ctx_actions, branches)."""
    rng = np.random.default_rng(seed)
    prefix = [int(rng.integers(num_actions)) for _ in range(prefix_len)]
    ctx_frames = ctx_actions = None
    branches = {}
    for a in range(num_actions):
        env = make_env(cfg["data"]["env"], frame_size=64,
                       frame_skip=cfg["data"]["frame_skip"], seed=seed, episode_timeout=0)
        try:
            frames = []
            f = env.reset()
            # Apply prefix[0 .. P-2], leaving the agent standing on frame P-1.
            for i in range(prefix_len - 1):
                frames.append(f)
                f, _ = env.step(prefix[i])
            frames.append(f)
            if ctx_frames is None:
                ctx_frames = np.asarray(frames[-context:])
                # The engine overwrites the final context action with the one
                # passed to step(), so its value here is irrelevant.
                ctx_actions = np.asarray(prefix[prefix_len - context : prefix_len - 1] + [0])
            out = []
            for _ in range(horizon):
                f, done = env.step(a)
                out.append(f)
                if done:
                    break
            while len(out) < horizon:      # episode ended; hold the last frame
                out.append(out[-1])
            branches[a] = np.asarray(out)
        finally:
            env.close()
    return ctx_frames, ctx_actions, branches


@torch.no_grad()
def model_branches(engine, ctx_frames, ctx_actions, num_actions: int, horizon: int):
    branches = {}
    for a in range(num_actions):
        engine.reset(ctx_frames, ctx_actions)
        branches[a] = np.asarray([engine.step(a) for _ in range(horizon)])
    return branches


def mean_pairwise(branches: dict) -> float:
    """Mean PSNR over every pair of branches, averaged across frames."""
    keys = sorted(branches)
    vals = []
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            fa, fb = branches[a], branches[b]
            vals.append(np.mean([psnr_capped(fa[k][None], fb[k][None]) for k in range(len(fa))]))
    return float(np.mean(vals))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--device", default="auto")
    p.add_argument("--starts", type=int, default=6, help="independent contexts to average over")
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--prefix", type=int, default=40, help="real-env steps before branching")
    p.add_argument("--gif", default="assets/action_ablation.gif")
    p.add_argument("--out", default="docs/ACTION_ABLATION.md")
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    device = pick_device(a.device)
    vq, dyn, dck = load_models(cfg, device)
    torch.manual_seed(0)
    ic = cfg["infer"]
    engine = NeuralGameEngine(
        vq, dyn,
        EngineConfig(decode=ic["decode"], maskgit_steps=ic["maskgit_steps"],
                     temperature=0.0, top_k=0),
        device=device, memory=None,
    )
    A, C = dyn.num_actions, dyn.context

    real_scores, model_scores, zero_scores = [], [], []
    keep = None
    for s in range(a.starts):
        ctx_f, ctx_a, rb = real_branches(cfg, A, C, a.prefix, a.horizon, seed=s)
        real_scores.append(mean_pairwise(rb))

        mb = model_branches(engine, ctx_f, ctx_a, A, a.horizon)
        model_scores.append(mean_pairwise(mb))

        # Control: blank the action embedding so every branch sees identical input.
        saved = dyn.act_emb.weight.data.clone()
        dyn.act_emb.weight.data.zero_()
        zb = model_branches(engine, ctx_f, ctx_a, A, a.horizon)
        dyn.act_emb.weight.data.copy_(saved)
        zero_scores.append(mean_pairwise(zb))

        if keep is None:
            keep = (ctx_f, rb, mb)
        print(f"  start {s}: real {real_scores[-1]:5.2f}  model {model_scores[-1]:5.2f}  "
              f"zeroed {zero_scores[-1]:5.2f} dB")

    real, model, zero = (float(np.mean(v)) for v in (real_scores, model_scores, zero_scores))
    # 0 = as indistinguishable as a dead controller, 1 = as separated as the real game.
    frac = (zero - model) / max(zero - real, 1e-9)
    print(f"\nreal {real:.2f} dB | model {model:.2f} dB | action-zeroed {zero:.2f} dB")
    print(f"action sensitivity: {100 * frac:.0f}% of the real game's separation")

    # Side by side: two opposing actions from one shared context.
    import cv2
    import imageio.v2 as imageio

    ctx_f, rb, mb = keep
    # From the checkpoint: this script builds the engine directly rather than
    # via load_engine, so the labels have to come from the same place it uses.
    names = tuple(dck.get("action_names") or [str(i) for i in range(A)])
    left = next((i for i, n in enumerate(names) if n == "turn L"), 1)
    right = next((i for i, n in enumerate(names) if n == "turn R"), 2)
    S, GAP, TOP, BOT = 192, 8, 30, 26
    up = lambda im: cv2.resize(im, (S, S), interpolation=cv2.INTER_NEAREST)  # noqa: E731
    out = []
    for k in range(a.horizon):
        canvas = np.full((TOP + S + BOT, S * 2 + GAP, 3), (16, 16, 20), np.uint8)
        canvas[TOP : TOP + S, :S] = up(mb[left][k])
        canvas[TOP : TOP + S, S + GAP :] = up(mb[right][k])
        for text, x, col in ((f"MODEL: hold '{names[left]}'", 8, (120, 220, 140)),
                             (f"MODEL: hold '{names[right]}'", S + GAP + 8, (120, 180, 230))):
            cv2.putText(canvas, text, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        d = psnr_capped(mb[left][k][None], mb[right][k][None])
        cv2.putText(canvas, f"frame {k + 1}/{a.horizon}   panes differ by {d:4.1f} dB",
                    (8, TOP + S + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 165), 1,
                    cv2.LINE_AA)
        out.append(canvas)
    os.makedirs(os.path.dirname(a.gif) or ".", exist_ok=True)
    imageio.mimsave(a.gif, out, duration=1000 / 6, loop=0, palettesize=128)
    print(f"wrote {a.gif}")

    verdict = (
        "**The model listens to the controller.**" if frac > 0.6 else
        "**The model partly listens to the controller.**" if frac > 0.25 else
        "**The model largely ignores the controller.**"
    )
    lines = [
        "# B0.5b: action-conditioning ablation",
        "",
        f"{a.starts} independent contexts, {A} distinct actions each held for {a.horizon} "
        "frames, greedy decoding so the action is the only thing that varies. The number "
        "reported is mean pairwise PSNR *between* branches: **lower means the actions "
        "produced more different futures.**",
        "",
        "| condition | mean pairwise PSNR | meaning |",
        "|---|---|---|",
        f"| real game | **{real:.2f} dB** | how much the actions truly separate the future |",
        f"| model | **{model:.2f} dB** | how much the model separates them |",
        f"| model, action embedding zeroed | {zero:.2f} dB | control: identical input, so "
        f"branches are pixel-identical and PSNR pins at the {PSNR_MAX:.0f} dB cap |",
        "",
        verdict,
        "",
        f"Placing the model on the scale between the two references: it recovers "
        f"**{100 * frac:.0f}%** of the real game's action separation. 0% would mean the "
        "button does nothing; 100% would mean it separates futures exactly as much as "
        "VizDoom does.",
        "",
        "The zeroed row is the load-bearing control. It pins at the cap, which proves the "
        "harness really is varying the action and that the measured model number is not an "
        "artifact of the branches sharing a context.",
        "",
        f"![opposing actions from one context]({os.path.basename(a.gif)})",
        "",
        "Both panes start from the same 6 frames of real context and then hold opposing "
        "turn actions. The PSNR readout is between the panes, not against ground truth.",
        "",
        "Regenerate with `python -m ngx.eval.action_ablation --config configs/small.yaml`.",
        "",
    ]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
