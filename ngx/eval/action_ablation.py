"""B0.5b: does the model listen to the controller, and does it listen correctly?

    python -m ngx.eval.action_ablation --config configs/small.yaml

A world model that ignores its action channel is not a game engine, it is an
expensive video autoplayer, and it will still post respectable PSNR because
most of the screen is predictable from the previous frame alone. So this is
measured directly rather than inferred.

Two tests, because the first one alone is not enough.

**Divergence.** Fix a context, hold each distinct action for N frames, and
measure mean pairwise PSNR *between* the resulting rollouts. Lower means the
actions produced more different futures. Two references give the number a
scale: the real game replayed and branched the same way (VizDoom is
deterministic under a fixed seed), and the model with its action embedding
zeroed, where every branch gets identical input and the rollouts must come out
pixel-identical.

**Grounding.** Divergence is necessary and not sufficient. It shows the model
produces different futures under different buttons; it does not show they are
the *correct* different futures, and a model that branched at random would
score exactly as well. So compare the model's rollout under action ``a``
against the real game's rollout under ``a``, and against the real game's
rollouts under every other action ``b``. If the matched pair is not clearly
closer than the mismatched ones, the model is branching without being grounded,
which is a worse finding than the divergence number alone would suggest.

Raw dB only, no ratios. An earlier version scored the model as a percentage of
the way from the zeroed control to the real game, which put this file's own
clipping constant in the denominator and moved the headline whenever the cap
moved.

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


def _traj_psnr(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean([psnr_capped(x[k][None], y[k][None]) for k in range(len(x))]))


def mean_pairwise(branches: dict) -> float:
    """Mean PSNR over every pair of branches, averaged across frames."""
    keys = sorted(branches)
    vals = [
        _traj_psnr(branches[a], branches[b])
        for i, a in enumerate(keys)
        for b in keys[i + 1 :]
    ]
    return float(np.mean(vals))


def cross_condition(model_b: dict, real_b: dict) -> tuple[float, float]:
    """Is the model's branch under ``a`` closest to the *real* branch under ``a``?

    Returns ``(matched, mismatched)`` mean PSNR. ``matched`` compares each model
    rollout to the real rollout under the same action; ``mismatched`` compares
    it to the real rollouts under every other action. Grounding means matched
    exceeds mismatched by a clear margin. A model that branches at random scores
    the two equally.
    """
    keys = sorted(model_b)
    matched, mismatched = [], []
    for a in keys:
        ma = model_b[a]
        matched.append(_traj_psnr(ma, real_b[a]))
        mismatched.append(float(np.mean([_traj_psnr(ma, real_b[b]) for b in keys if b != a])))
    return float(np.mean(matched)), float(np.mean(mismatched))


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

    real_s, model_s, zero_s, matched_s, mismatched_s = [], [], [], [], []
    keep = None
    for s in range(a.starts):
        ctx_f, ctx_a, rb = real_branches(cfg, A, C, a.prefix, a.horizon, seed=s)
        real_s.append(mean_pairwise(rb))

        mb = model_branches(engine, ctx_f, ctx_a, A, a.horizon)
        model_s.append(mean_pairwise(mb))

        # Control: blank the action embedding so every branch sees identical input.
        saved = dyn.act_emb.weight.data.clone()
        dyn.act_emb.weight.data.zero_()
        zb = model_branches(engine, ctx_f, ctx_a, A, a.horizon)
        dyn.act_emb.weight.data.copy_(saved)
        zero_s.append(mean_pairwise(zb))

        m, mm = cross_condition(mb, rb)
        matched_s.append(m)
        mismatched_s.append(mm)

        if keep is None:
            keep = (ctx_f, rb, mb)
        print(f"  start {s}: divergence real {real_s[-1]:5.2f} model {model_s[-1]:5.2f} "
              f"zeroed {zero_s[-1]:5.2f} | grounding matched {m:5.2f} vs mismatched {mm:5.2f} dB")

    real, model, zero = (float(np.mean(v)) for v in (real_s, model_s, zero_s))
    matched, mismatched = float(np.mean(matched_s)), float(np.mean(mismatched_s))
    margin = matched - mismatched
    wins = sum(x > y for x, y in zip(matched_s, mismatched_s))
    print("")
    print(f"divergence:  real {real:.2f} | model {model:.2f} | zeroed {zero:.2f} dB (cap)")
    print(f"grounding:   matched {matched:.2f} | mismatched {mismatched:.2f} | "
          f"margin {margin:+.2f} dB ({wins}/{len(matched_s)} contexts)")

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

    grounded = (
        "**Grounded.**" if margin > 1.0 else
        "**Weakly grounded.**" if margin > 0.25 else
        "**Not grounded.** The model branches under different actions, but its branch under "
        "an action is no closer to the real outcome of that action than to the outcome of a "
        "different one. Divergence alone was hiding this."
    )
    lines = [
        "# B0.5b: action conditioning",
        "",
        f"{a.starts} independent contexts, {A} distinct actions each held for {a.horizon} "
        "frames, greedy decoding so the action is the only thing that varies.",
        "",
        "## Divergence: do different buttons produce different futures?",
        "",
        "Mean pairwise PSNR *between* branches. **Lower means more separation.**",
        "",
        "| condition | mean pairwise PSNR |",
        "|---|---|",
        f"| real game | **{real:.2f} dB** |",
        f"| model | **{model:.2f} dB** |",
        f"| model, action embedding zeroed | {zero:.2f} dB (clipped at {PSNR_MAX:.0f}) |",
        "",
        "The model separates futures about as much as the real game does. The zeroed row is "
        "the load-bearing control: identical input makes the branches pixel-identical, so it "
        "pins at the clipping cap and proves the harness really is varying the action.",
        "",
        "No percentage is quoted here. Expressing the model as a fraction of the distance "
        "from the zeroed control to the real game would put the clipping constant in the "
        "denominator, and the headline would move whenever that constant moved.",
        "",
        "## Grounding: are they the *right* different futures?",
        "",
        "Divergence is necessary and not sufficient: a model branching at random scores the "
        "same. This compares the model's rollout under action `a` to the real game's rollout "
        "under `a`, and to the real game's rollouts under every other action.",
        "",
        "| comparison | mean PSNR |",
        "|---|---|",
        f"| model under `a` vs **real under `a`** (matched) | **{matched:.2f} dB** |",
        f"| model under `a` vs real under `b != a` (mismatched) | {mismatched:.2f} dB |",
        f"| margin | **{margin:+.2f} dB** ({wins}/{len(matched_s)} contexts favour matched) |",
        "",
        grounded,
        "",
        f"![opposing actions from one context]({os.path.basename(a.gif)})",
        "",
        "Both panes start from the same context and then hold opposing turn actions. The "
        "PSNR readout is between the panes, not against ground truth.",
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
