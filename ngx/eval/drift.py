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

Writes docs/DRIFT.md comparing memory on against memory off, plus how often the
memory key retrieves a frame of the same place at a genuine revisit. Every
sentence in the report is computed from this run; none is carried over from
an earlier checkpoint.
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


def retrieval_accuracy(vq, frames, poses, pairs, device, write_every: int, exclude_recent: int,
                       pos_tol: float, ang_tol: float, batch: int = 256) -> dict:
    """How often the memory key finds the room at a genuine revisit.

    Tokenizer-only: keys are built from the real frames exactly as
    :class:`RetrievalMemory` builds them, so the answer is the same for every
    dynamics checkpoint that shares this tokenizer. For each revisit pair
    ``(i, j)`` the candidates are the real frames memory would hold at step
    ``j``: every ``write_every``-th frame before ``j``, minus the most recent
    ``exclude_recent`` writes. A retrieval is correct when the stored frame was
    taken within the revisit tolerance of frame ``j``'s pose. Pairs with no
    correct candidate in memory are left out and counted separately.
    """
    from .baselines import encode

    E = vq.quantizer.embed.detach().float().to(device)
    toks = torch.cat([encode(vq, frames[s:s + batch], device) for s in range(0, len(frames), batch)])
    keys = E[toks.long()].mean(1)
    keys = keys / keys.norm(dim=1, keepdim=True).clamp_min(1e-8)
    stored = np.arange(write_every - 1, len(frames), write_every)
    finite = np.isfinite(poses).all(1)

    n = top1 = top2 = no_candidate = 0
    for _i, j in pairs:
        cand = stored[stored < j]
        cand = cand[: max(len(cand) - exclude_recent, 0)]
        if len(cand) == 0 or not finite[j]:
            no_candidate += 1
            continue
        dpos = np.linalg.norm(poses[cand, :2] - poses[j, :2], axis=1)
        dang = np.abs(poses[cand, 2] - poses[j, 2]) % 360.0
        dang = np.minimum(dang, 360.0 - dang)
        ok = finite[cand] & (dpos <= pos_tol) & (dang <= ang_tol)
        if not ok.any():
            no_candidate += 1
            continue
        sims = (keys[torch.as_tensor(cand, device=device)] @ keys[j]).cpu().numpy()
        order = np.argsort(-sims)
        n += 1
        top1 += int(ok[order[0]])
        top2 += int(ok[order[:2]].any())
    return {"pairs": n, "no_candidate": no_candidate,
            "top1": top1 / max(n, 1), "top2": top2 / max(n, 1)}


def main() -> None:
    from ..config import find_ckpt
    from ..train.common import load_ckpt

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--device", default="auto")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=4,
                   help="rollouts per configuration; decoding samples, so a single "
                        "rollout cannot distinguish an effect from noise")
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

    probe = load_engine(cfg, device=device, memory=False)
    ctx = probe.C
    ckpt_path = find_ckpt(cfg, "dynamics", "dynamics.pt")
    ck = load_ckpt(ckpt_path, map_location="cpu")
    params = sum(q.numel() for q in probe.model.parameters())

    # Find the revisit pairs once, so both configurations are scored on exactly
    # the same moments.
    pairs = find_revisits(poses, start=ctx, min_gap=a.min_gap,
                          pos_tol=a.pos_tol, ang_tol=a.ang_tol)
    gaps = [j - i for i, j in pairs]
    print(f"  {len(pairs)} revisit pairs (median gap {int(np.median(gaps)) if gaps else 0} frames)")

    mc = cfg.get("memory", {})
    write_every = int(mc.get("write_every", 4))
    exclude_recent = int(mc.get("exclude_recent", 64))
    ret = retrieval_accuracy(probe.vq, frames, poses, pairs, device, write_every, exclude_recent,
                             a.pos_tol, a.ang_tol)
    print(f"  memory key finds the place: top-1 {100 * ret['top1']:.0f}%, top-2 "
          f"{100 * ret['top2']:.0f}% over {ret['pairs']} revisits "
          f"({ret['no_candidate']} with no matching frame in memory)")
    del probe

    results = {}
    for use_mem in (False, True):
        label = "memory" if use_mem else "sliding context only"
        print(f"evaluating: {label}  ({a.seeds} rollouts)")
        runs = []
        for sd in range(a.seeds):
            torch.manual_seed(a.seed + sd)
            engine = load_engine(cfg, device=device, memory=use_mem)
            runs.append(evaluate(engine, frames, actions, poses, a.steps, pairs))
        agg = {
            "frames": runs[0]["frames"],
            "revisits": runs[0]["revisits"],
            "env_revisit_psnr": runs[0]["env_revisit_psnr"],
            "revisit_mean": float(np.mean([r["model_revisit_psnr"] for r in runs])),
            "revisit_sd": float(np.std([r["model_revisit_psnr"] for r in runs])),
            "hits": float(np.mean([r["retrieval_hits"] for r in runs])),
            "curve": {
                k: (
                    float(np.mean([r["curve"][k] for r in runs])),
                    float(np.std([r["curve"][k] for r in runs])),
                )
                for k in runs[0]["curve"]
            },
        }
        results[label] = agg
        print(
            f"  revisit PSNR {agg['revisit_mean']:.2f} +/- {agg['revisit_sd']:.2f} dB "
            f"over {agg['revisits']} pairs (game itself: {agg['env_revisit_psnr']:.2f} dB), "
            f"retrieval fired on {agg['hits']:.0f}/{agg['frames']} frames"
        )

    trained = ""
    if ck.get("tokens_seen"):
        trained = f", trained on {ck['tokens_seen'] / 1e9:.2f}B tokens ({ck.get('epochs', 0):.2f} epochs)"
    regen = " ".join(["python -m ngx.eval.drift --config", a.config]
                     + (["--set", *a.set] if a.set else [])
                     + ([f"--out {a.out}"] if a.out != "docs/DRIFT.md" else []))

    ks = [k for k in CHECKPOINTS if k <= min(r["frames"] for r in results.values())]
    lines = [
        "# Drift",
        "",
        f"Checkpoint: `{ckpt_path.replace(os.sep, '/')}`, {params / 1e6:.1f}M parameters, "
        f"{ctx}-frame context{trained}.",
        "",
        f"Reference trajectory: {len(frames)} real frames from one unbroken episode, "
        f"explorer policy, env seed {used_seed}. Every number below is the mean over "
        f"{a.seeds} rollouts with different sampling seeds, +/- one standard deviation. "
        "Decoding samples, so a single rollout cannot tell an effect from noise.",
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
        cells = " | ".join(f"{r['curve'][k][0]:.1f}" for k in ks)
        lines.append(f"| {label} | {cells} |")

    lines += [
        "",
        "## Return-to-place consistency",
        "",
        f"Pairs of steps where the real player stood within {a.pos_tol:g} map units and "
        f"{a.ang_tol:g} degrees of a pose from at least {a.min_gap} steps earlier "
        f"(median actual gap: {int(np.median(gaps)) if gaps else 0} frames, far outside "
        f"the model's {ctx}-frame context, so nothing but memory can carry the room "
        f"across). `game` is the same measurement on the real frames: the ceiling, since "
        f"matching poses are close but never identical.",
        "",
        "| config | pairs | model | game (ceiling) | gap | retrieval fired |",
        "|---|---|---|---|---|---|",
    ]
    for label, r in results.items():
        gap = r["env_revisit_psnr"] - r["revisit_mean"]
        lines.append(
            f"| {label} | {r['revisits']} | **{r['revisit_mean']:.2f} +/- "
            f"{r['revisit_sd']:.2f} dB** | {r['env_revisit_psnr']:.2f} dB | {gap:.2f} dB | "
            f"{r['hits']:.0f}/{r['frames']} frames |"
        )

    base, mem = results["sliding context only"], results["memory"]
    delta = mem["revisit_mean"] - base["revisit_mean"]
    pooled = (base["revisit_sd"] ** 2 + mem["revisit_sd"] ** 2) ** 0.5
    if abs(delta) <= pooled:
        verdict = (f"**Retrieval memory does not measurably change return-to-place on this "
                   f"checkpoint.** It moves the score by {delta:+.2f} dB against a run-to-run "
                   f"spread of +/-{pooled:.2f} dB.")
    elif delta > 0:
        verdict = (f"**Retrieval memory improves return-to-place by {delta:.2f} dB**, outside "
                   f"the run-to-run spread of +/-{pooled:.2f} dB.")
    else:
        verdict = (f"**Retrieval memory makes return-to-place worse by {-delta:.2f} dB**, "
                   f"outside the run-to-run spread of +/-{pooled:.2f} dB.")
    lines += [
        "",
        "## What this says",
        "",
        verdict,
        "",
        f"*Does the key find the place?* For {ret['pairs']} of the revisits, memory holds at "
        f"least one frame taken within the revisit tolerance of the current pose. The most "
        f"similar stored frame is one of them {100 * ret['top1']:.0f}% of the time, and one of "
        f"the top two is {100 * ret['top2']:.0f}% of the time. For the other "
        f"{ret['no_candidate']} revisits nothing from that place is in memory yet. Keys come "
        f"from the tokenizer alone, so these rates are the same for every dynamics checkpoint "
        f"that shares it.",
        "",
        f"`memory.enabled` is `{str(bool(mc.get('enabled', False))).lower()}` in this config; "
        "the `M` key toggles it in `play.py`.",
        "",
        f"One structural note on the curve: `exclude_recent` blocks retrieval until "
        f"{exclude_recent} writes have accumulated (one every {write_every} frames), so the two "
        "configurations are identical by construction for the first few hundred frames.",
        "",
        f"Regenerate with `{regen}`.",
        "",
    ]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
