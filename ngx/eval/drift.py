"""Stage 5: quantify drift, and whether retrieval memory reduces it.

    python -m ngx.eval.drift --config configs/small.yaml

Two numbers, measuring two different failure modes.

**Drift over N frames.** Run the model and the real game from the same seed
frames under an identical action sequence, and track PSNR between them as the
rollout gets longer. This decays no matter what -- the model is sampling, not
simulating -- so the interesting part is the shape of the curve, not the floor.

**Return-to-place consistency.** Find two moments in the real trajectory where
the player stood in the same spot facing the same way, with a long gap between
them. Ask whether the model drew the same room both times. This is the metric
retrieval memory is built to move, and it is reported against a *control*: the
real game's own PSNR between those two moments, which is below infinity because
"same pose" is a tolerance, not an identity. The control is the ceiling; the
model's score only means something relative to it.

Writes docs/DRIFT.md comparing memory on against memory off.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from ..config import load_config, pick_device
from ..envs import make_env
from ..infer.load import load_engine
from .bench import psnr_u8

CHECKPOINTS = (1, 10, 25, 50, 100, 250, 500, 1000)


def _rollout(cfg: dict, steps: int, seed: int):
    """One episode of the real game under the explorer policy."""
    from ..data.policies import Explorer

    env = make_env(
        cfg["data"]["env"], frame_size=64, frame_skip=cfg["data"]["frame_skip"],
        seed=seed, episode_timeout=0,  # 0 disables VizDoom's episode cap
    )
    rng = np.random.default_rng(seed)
    pol = Explorer(env.action_names, rng=rng)
    try:
        frame = env.reset()
        frames, actions, poses = [], [], []
        for _ in range(steps):
            pose = env.pose()
            a = pol.act(pose)
            frames.append(frame)
            actions.append(a)
            poses.append(pose if pose is not None else (np.nan,) * 3)
            frame, done = env.step(a)
            if done:  # goal reached; stop rather than splice two episodes
                break
    finally:
        env.close()
    return np.asarray(frames), np.asarray(actions), np.asarray(poses, dtype=np.float64)


def reference_trajectory(cfg: dict, steps: int, seed: int = 0, max_tries: int = 16):
    """The longest single episode we can find, up to ``steps`` frames.

    ``my_way_home`` ends the moment the player stumbles onto the goal, which
    for an explorer policy can happen after 200 steps or not for 1500. Drift
    has to be measured inside one continuous episode -- splicing two together
    would put a teleport in the middle and call it drift -- so search seeds
    until one runs long enough, and report which one was used.
    """
    best = None
    for s in range(seed, seed + max_tries):
        frames, actions, poses = _rollout(cfg, steps, s)
        if best is None or len(frames) > len(best[0]):
            best = (frames, actions, poses, s)
        if len(frames) >= steps:
            break
    return best


def find_revisits(poses, start: int, min_gap: int = 60, pos_tol: float = 40.0,
                  ang_tol: float = 20.0, limit: int = 40):
    """Index pairs ``(i, j)`` where the player returned to a pose, ``j - i >= min_gap``."""
    out, used = [], set()
    n = len(poses)
    for i in range(start, n):
        if not np.isfinite(poses[i]).all() or i in used:
            continue
        for j in range(i + min_gap, n):
            if j in used or not np.isfinite(poses[j]).all():
                continue
            if np.linalg.norm(poses[j, :2] - poses[i, :2]) > pos_tol:
                continue
            d = abs(poses[j, 2] - poses[i, 2]) % 360.0
            if min(d, 360.0 - d) > ang_tol:
                continue
            out.append((i, j))
            used.update((i, j))
            break
        if len(out) >= limit:
            break
    return out


def model_rollout(engine, frames, actions, n: int):
    """Predict ``n`` frames; ``preds[k]`` lines up with ``frames[C + k]``."""
    C = engine.C
    engine.reset(frames[:C], actions[:C])
    preds, retr = [], 0
    for k in range(n):
        preds.append(engine.step(int(actions[C - 1 + k])))
        retr += int(engine.last_retrieved > 0)
    return np.asarray(preds), retr


def evaluate(engine, frames, actions, poses, steps: int, pairs=None):
    C = engine.C
    n = min(steps, len(frames) - C)
    preds, retr = model_rollout(engine, frames, actions, n)
    real = frames[C : C + n]

    curve = {k: psnr_u8(preds[k - 1 : k], real[k - 1 : k]) for k in CHECKPOINTS if k <= n}

    if pairs is None:
        pairs = find_revisits(poses, start=C)
    model_scores, env_scores = [], []
    for i, j in pairs:
        mi, mj = i - C, j - C
        if mi < 0 or mj >= n:
            continue
        model_scores.append(psnr_u8(preds[mi : mi + 1], preds[mj : mj + 1]))
        env_scores.append(psnr_u8(frames[i : i + 1], frames[j : j + 1]))
    return {
        "frames": n,
        "curve": curve,
        "revisits": len(model_scores),
        "model_revisit_psnr": float(np.mean(model_scores)) if model_scores else float("nan"),
        "env_revisit_psnr": float(np.mean(env_scores)) if env_scores else float("nan"),
        "retrieval_hits": retr,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--device", default="auto")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pos-tol", type=float, default=40.0, help="revisit radius, map units")
    p.add_argument("--ang-tol", type=float, default=20.0, help="revisit heading tolerance, deg")
    p.add_argument("--min-gap", type=int, default=60, help="min frames between the two visits")
    p.add_argument("--out", default="docs/DRIFT.md")
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    device = pick_device(a.device)
    torch.manual_seed(a.seed)

    print(f"searching for a real episode of at least {a.steps} steps...")
    frames, actions, poses, used_seed = reference_trajectory(cfg, a.steps + 64, a.seed)
    print(f"  got {len(frames)} frames from seed {used_seed}")

    # Find the revisit pairs once, so both configurations are scored on exactly
    # the same moments.
    ctx = load_config(a.config, a.set)["dynamics"]["context"]
    pairs = find_revisits(poses, start=ctx, min_gap=a.min_gap,
                          pos_tol=a.pos_tol, ang_tol=a.ang_tol)
    gaps = [j - i for i, j in pairs]
    print(
        f"  {len(pairs)} revisit pairs (median gap {int(np.median(gaps)) if gaps else 0} frames)"
    )

    results = {}
    for use_mem in (False, True):
        torch.manual_seed(a.seed)
        engine = load_engine(cfg, device=device, memory=use_mem)
        label = "memory" if use_mem else "sliding context only"
        print(f"evaluating: {label}")
        results[label] = evaluate(engine, frames, actions, poses, a.steps, pairs)
        r = results[label]
        print(
            f"  revisit PSNR {r['model_revisit_psnr']:.2f} dB over {r['revisits']} pairs "
            f"(game itself: {r['env_revisit_psnr']:.2f} dB), "
            f"retrieval fired on {r['retrieval_hits']}/{r['frames']} frames"
        )

    ks = [k for k in CHECKPOINTS if k <= min(r["frames"] for r in results.values())]
    lines = [
        "# Drift",
        "",
        f"Reference trajectory: {len(frames)} real frames from one unbroken episode, "
        f"explorer policy, env seed {used_seed}.",
        "",
        "## Divergence from the real game",
        "",
        "PSNR between the model's frame and the game's frame at step *k*, both driven "
        "by the same actions from the same starting frames.",
        "",
        "| config | " + " | ".join(f"k={k}" for k in ks) + " |",
        "|---" * (len(ks) + 1) + "|",
    ]
    for label, r in results.items():
        cells = " | ".join(f"{r['curve'].get(k, float('nan')):.1f}" for k in ks)
        lines.append(f"| {label} | {cells} |")

    lines += [
        "",
        "## Return-to-place consistency",
        "",
        f"Pairs of steps where the real player stood within {a.pos_tol:g} map units and "
        f"{a.ang_tol:g} degrees of a pose from at least {a.min_gap} steps earlier "
        f"(median actual gap: {int(np.median(gaps)) if gaps else 0} frames -- far outside "
        f"the model's {ctx}-frame context, so nothing but memory can carry the room "
        f"across). `game` is the same measurement on the real frames: the ceiling, since "
        f"matching poses are close but never identical.",
        "",
        "| config | pairs | model | game (ceiling) | gap | retrieval fired |",
        "|---|---|---|---|---|---|",
    ]
    for label, r in results.items():
        gap = r["env_revisit_psnr"] - r["model_revisit_psnr"]
        lines.append(
            f"| {label} | {r['revisits']} | **{r['model_revisit_psnr']:.2f} dB** | "
            f"{r['env_revisit_psnr']:.2f} dB | {gap:.2f} dB | "
            f"{r['retrieval_hits']}/{r['frames']} frames |"
        )
    lines += ["", "Regenerate with `python -m ngx.eval.drift --config configs/small.yaml`.", ""]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
