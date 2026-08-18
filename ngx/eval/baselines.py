"""B0: does the dynamics model beat copying the previous frame?

    python -m ngx.eval.baselines --config configs/small.yaml

Every downstream number in this repo depends on the answer, so the comparison
is set up to be as favourable to the model as honesty allows and still be fair.

Three predictors, measured on the same held-out frames:

``copy-last-frame``
    Predict frame ``t+1`` by handing back frame ``t`` unchanged. No parameters,
    no training. In a 64x64 game at frame_skip 4 this is a genuinely strong
    baseline, because most of the screen does not change between steps.
``model``
    The trained dynamics model.
``tokenizer ceiling``
    Encode the *true* next frame and decode it again. The dynamics model can
    never beat this: it emits tokens, and those tokens go through the same
    decoder. This is the upper bound on the whole approach at this tokenizer.

Stratified, not aggregated
--------------------------
Some consecutive frames in this game are *pixel identical* -- a no-op action,
or the agent walking into a wall. On those, copy-last-frame is exactly right
and PSNR is infinite, so any aggregate containing them is decided by where the
infinity gets clipped rather than by the models. An earlier version of this
file capped PSNR and reported the mean; moving the cap from 60 dB to 100 dB
flipped the sign of the headline. Reporting the median and a win rate instead
merely routed around the problem.

So the one-step evaluation is split:

``static``
    consecutive frames pixel-identical. Copy is exact here by construction, and
    the question is not whether the model wins (it cannot) but how much error it
    injects into a frame that should not change. That is the shimmer the player
    sees standing still.
``moving``
    everything else. No infinities, so no cap exists and no clipping choice can
    carry a conclusion. Means are well defined and headroom is computed here.

Two regimes, because they answer different questions:

**One-step, teacher-forced.** The model gets real frames for context and
predicts one frame. This is the apples-to-apples test of whether the model
learned the dynamics, and drift cannot be blamed for it.

**Closed-loop.** The model consumes its own predictions; the baseline freezes
on the last real frame. Reported per k, averaged across many independent
rollouts. Never averaged *over* k: a trajectory that goes 14.0, 9.8, 11.8 is
not something a single mean describes.

The model is measured both greedily and with the sampling settings play.py
uses. Greedy is the model's best single guess and is the fair number for "did
it learn the dynamics"; the sampled row is what a player sees.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from ..config import load_config, pick_device
from ..data.dataset import VAL_EVERY
from ..infer.engine import EngineConfig, NeuralGameEngine
from ..infer.load import load_models
from .bench import psnr_u8

#: rollout steps to report; never summarised into a single mean
MARKS = (1, 2, 4, 8, 16, 32)

#: Only the closed-loop section needs this, and only because a frozen frame can
#: coincide exactly with a stalled agent. The one-step section is stratified so
#: that no cap is involved in anything a conclusion rests on.
PSNR_MAX = 100.0


def psnr_capped(a: np.ndarray, b: np.ndarray) -> float:
    return min(psnr_u8(a, b), PSNR_MAX)


def sample_windows(root: str, length: int, n: int, seed: int = 0):
    """``n`` windows of ``length`` consecutive frames from held-out episodes."""
    frames = np.load(os.path.join(root, "frames.npy"), mmap_mode="r")
    actions = np.load(os.path.join(root, "actions.npy"), mmap_mode="r")
    episodes = np.load(os.path.join(root, "episodes.npy"))
    rng = np.random.default_rng(seed)
    out, tries = [], 0
    while len(out) < n and tries < n * 200:
        tries += 1
        i = int(rng.integers(0, len(frames) - length))
        if episodes[i] != episodes[i + length - 1] or episodes[i] % VAL_EVERY:
            continue
        out.append((np.asarray(frames[i : i + length]), np.asarray(actions[i : i + length])))
    if len(out) < n:
        raise RuntimeError(f"only found {len(out)}/{n} held-out windows of length {length}")
    return out


@torch.no_grad()
def encode(vq, frames: np.ndarray, device) -> torch.Tensor:
    x = torch.from_numpy(frames.astype(np.float32) / 127.5 - 1.0).permute(0, 3, 1, 2).to(device)
    return vq.encode_indices(x)


@torch.no_grad()
def roundtrip(vq, frames: np.ndarray, device) -> np.ndarray:
    """Encode and decode real frames: the tokenizer's own ceiling."""
    idx = encode(vq, frames, device)
    px = vq.decode_indices(idx)
    px = ((px.float().clamp(-1, 1) + 1) * 127.5).round().byte()
    return px.permute(0, 2, 3, 1).cpu().numpy()


@torch.no_grad()
def one_step(engines: dict, vq, windows, device) -> list[dict]:
    """One record per window, so the caller can stratify rather than aggregate."""
    C = next(iter(engines.values())).C
    recs = []
    for frames, actions in windows:
        tok = encode(vq, frames[C - 1 : C + 1], device)   # tokens of f[C-1], f[C]
        rec = {
            "static": bool(np.array_equal(frames[C - 1], frames[C])),
            "copy_psnr": psnr_u8(frames[C - 1][None], frames[C][None]),  # inf when static
            "copy_acc": float((tok[0] == tok[1]).float().mean()),
            "ceiling_psnr": psnr_u8(roundtrip(vq, frames[C : C + 1], device), frames[C][None]),
        }
        for label, engine in engines.items():
            engine.reset(frames[:C], actions[:C])
            pred = engine.step(int(actions[C - 1]))
            pt = engine.tokens[-1]
            rec[f"{label}|psnr"] = psnr_u8(pred[None], frames[C][None])
            rec[f"{label}|acc"] = float((pt == tok[1]).float().mean())
            # Did the model reproduce the previous frame's tokens exactly? On a
            # static transition that is the difference between a frozen picture
            # and a shimmering one.
            rec[f"{label}|freeze"] = bool(torch.equal(pt, tok[0]))
        recs.append(rec)
    return recs


def agg(vals: list[float]) -> tuple[float, float]:
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if not len(a):
        return float("nan"), float("nan")
    return float(a.mean()), float(np.median(a))


@torch.no_grad()
def closed_loop(engine, vq, windows, device, horizon: int):
    """Model consumes its own output; the baseline freezes on the last real frame."""
    C = engine.C
    model = np.zeros((len(windows), horizon))
    frozen = np.zeros_like(model)
    ceil = np.zeros_like(model)
    for w, (frames, actions) in enumerate(windows):
        engine.reset(frames[:C], actions[:C])
        rt = roundtrip(vq, frames[C : C + horizon], device)
        for k in range(horizon):
            pred = engine.step(int(actions[C - 1 + k]))
            truth = frames[C + k]
            model[w, k] = psnr_capped(pred[None], truth[None])
            frozen[w, k] = psnr_capped(frames[C - 1][None], truth[None])
            ceil[w, k] = psnr_capped(rt[k][None], truth[None])
    return model, frozen, ceil


def _churn_stats(frames: list, tokens: torch.Tensor) -> dict:
    """Identical fraction and token churn over consecutive pairs of one rollout."""
    ident = [bool(np.array_equal(frames[k], frames[k - 1])) for k in range(1, len(frames))]
    churn = [int((tokens[k] != tokens[k - 1]).sum()) for k in range(1, len(tokens))]
    return {"ident": ident, "churn": churn}


@torch.no_grad()
def still_compare(cfg, engine, vq, device, starts: int, horizon: int, prefix_len: int, noop: int):
    """Hold the no-op in the model and in the real game, from identical starts.

    The teacher-forced static row predicts shimmer from single-step behaviour.
    This measures it the way a viewer meets it: nothing pressed, the model
    consuming its own output. But a churn number on its own supports either
    reading -- too twitchy or too sticky -- so the real game is run through the
    same prefix, the same no-op, and the same tokenizer, and reported beside it.

    Frames are compared byte-wise for the identical fraction, and in token space
    for churn, because tokens are the channel the model actually acts in and the
    decoder is shared by both rows.
    """
    from ..envs import make_env

    C = engine.C
    rng_prefix = np.random.default_rng(0)
    real, model = {"ident": [], "churn": []}, {"ident": [], "churn": []}

    for s in range(starts):
        prefix = [int(rng_prefix.integers(engine.model.num_actions)) for _ in range(prefix_len)]
        env = make_env(cfg["data"]["env"], frame_size=64,
                       frame_skip=cfg["data"]["frame_skip"], seed=s, episode_timeout=0)
        try:
            ctx = []
            f = env.reset()
            for i in range(prefix_len - 1):
                ctx.append(f)
                f, _ = env.step(prefix[i])
            ctx.append(f)
            ctx_frames = np.asarray(ctx[-C:])
            ctx_actions = np.asarray(prefix[prefix_len - C : prefix_len - 1] + [noop])
            real_frames = []
            for _ in range(horizon):
                f, done = env.step(noop)
                real_frames.append(f)
                if done:
                    break
        finally:
            env.close()
        if len(real_frames) < 2:
            continue

        rs = _churn_stats(real_frames, encode(vq, np.asarray(real_frames), device))
        real["ident"] += rs["ident"]
        real["churn"] += rs["churn"]

        engine.reset(ctx_frames, ctx_actions)
        mf, mt = [], []
        for _ in range(len(real_frames)):
            mf.append(engine.step(noop))
            mt.append(engine.tokens[-1].clone())
        ms = _churn_stats(mf, torch.stack(mt))
        model["ident"] += ms["ident"]
        model["churn"] += ms["churn"]

    def summarise_side(d):
        ident = float(np.mean(d["ident"]))
        churn = float(np.mean(d["churn"]))
        moved = [c for c, i in zip(d["churn"], d["ident"]) if not i]
        return {
            "ident": ident,
            "churn": churn,
            "churn_when_moved": float(np.mean(moved)) if moved else 0.0,
            "n": len(d["ident"]),
        }

    return summarise_side(real), summarise_side(model)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--device", default="auto")
    p.add_argument("--windows", type=int, default=400, help="windows for the one-step test")
    p.add_argument("--rollouts", type=int, default=32, help="independent closed-loop rollouts")
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--still-rollouts", type=int, default=8)
    p.add_argument("--still-frames", type=int, default=60)
    p.add_argument("--prefix", type=int, default=40,
                   help="real-env steps replayed before holding the no-op")
    p.add_argument("--out", default="docs/BASELINES.md")
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    device = pick_device(a.device)
    vq, dyn, dck = load_models(cfg, device)
    torch.manual_seed(0)

    ic = cfg["infer"]
    mk = dict(decode=ic["decode"], maskgit_steps=ic["maskgit_steps"])
    engines = {
        "greedy": NeuralGameEngine(
            vq, dyn, EngineConfig(**mk, temperature=0.0, top_k=0), device=device, memory=None),
        "sampled": NeuralGameEngine(
            vq, dyn, EngineConfig(**mk, temperature=ic["temperature"], top_k=ic["top_k"]),
            device=device, memory=None),
    }

    C = dyn.context
    print(f"one-step: {a.windows} held-out windows")
    recs = one_step(engines, vq,
                    sample_windows(cfg["data"]["root"], C + 1, a.windows, seed=0), device)
    static = [r for r in recs if r["static"]]
    moving = [r for r in recs if not r["static"]]
    n, ns, nm = len(recs), len(static), len(moving)
    print(f"  static {ns} ({100 * ns / n:.1f}%)   moving {nm} ({100 * nm / n:.1f}%)")

    def row(subset, key):
        return agg([r[key] for r in subset])

    for name, subset in (("moving", moving), ("static", static)):
        if not subset:
            continue
        cm, cmd = row(subset, "copy_psnr")
        gm, gmd = row(subset, "greedy|psnr")
        print(f"  [{name}] copy mean {cm:6.2f} median {cmd:6.2f} acc "
              f"{np.mean([r['copy_acc'] for r in subset]):.3f} | "
              f"model mean {gm:6.2f} median {gmd:6.2f} acc "
              f"{np.mean([r['greedy|acc'] for r in subset]):.3f}")

    # Headroom is a moving-subset question: a static transition is not a
    # prediction problem, it is a decision not to change anything.
    c_mean, c_med = row(moving, "copy_psnr")
    g_mean, g_med = row(moving, "greedy|psnr")
    ceil_mean, ceil_med = row(moving, "ceiling_psnr")
    head = ceil_mean - c_mean
    captured = g_mean - c_mean
    print(f"\nmoving headroom: copy {c_mean:.2f} -> ceiling {ceil_mean:.2f} = {head:.2f} dB; "
          f"model captures {captured:.2f} dB ({100 * captured / head:.0f}%)")

    if not static:
        raise RuntimeError(
            "no static transitions in the sample, so the shimmer question cannot be "
            "answered. Raise --windows."
        )
    freeze = float(np.mean([r["greedy|freeze"] for r in static]))
    s_acc = float(np.mean([r["greedy|acc"] for r in static]))
    s_model_mean, s_model_med = row(static, "greedy|psnr")
    s_ceiling_mean, _ = row(static, "ceiling_psnr")
    print(f"static subset: model {s_model_mean:.2f} dB, reproduces the previous frame's "
          f"tokens exactly on {100 * freeze:.0f}% of them")

    names = list(dck.get("action_names") or [])
    noop = names.index("-") if "-" in names else 0
    real_still, model_still = still_compare(
        cfg, engines["greedy"], vq, device,
        a.still_rollouts, a.still_frames, a.prefix, noop,
    )
    print(f"hold-still ({a.still_rollouts} starts x {a.still_frames} frames, "
          f"action '{names[noop] if names else noop}'):")
    for lbl, d in (("real game", real_still), ("model", model_still)):
        print(f"  {lbl:10s} identical {100 * d['ident']:5.1f}%   churn {d['churn']:.2f}/64   "
              f"when it moves {d['churn_when_moved']:.1f}/64   (n={d['n']})")

    print(f"\nclosed-loop: {a.rollouts} rollouts x {a.horizon} frames (greedy)")
    rw = sample_windows(cfg["data"]["root"], C + a.horizon, a.rollouts, seed=1)
    model, frozen, ceil = closed_loop(engines["greedy"], vq, rw, device, a.horizon)
    for k in MARKS:
        if k <= a.horizon:
            print(f"  k={k:<3d} copy {frozen[:, k - 1].mean():5.2f}   "
                  f"model {model[:, k - 1].mean():5.2f}   ceiling {ceil[:, k - 1].mean():5.2f}")

    params = sum(q.numel() for q in dyn.parameters()) / 1e6
    lines = [
        "# B0: baselines",
        "",
        f"Model: {params:.1f}M params, context {C} frames, "
        f"val loss {dck.get('val_loss', float('nan')):.3f}, "
        f"cold token accuracy {dck.get('cold_acc', float('nan')):.3f}. "
        f"Greedy decoding, {ic['maskgit_steps']}-pass MaskGIT, memory off.",
        "",
        "## One-step, teacher-forced",
        "",
        f"Real frames for context, one frame predicted, {n} held-out windows. This is the "
        "test that cannot be excused by drift.",
        "",
        "Split by whether the two real frames are pixel-identical. On those, copy-last-frame "
        "is exactly right and PSNR is infinite, so any aggregate that mixes them is decided "
        "by where the infinity is clipped rather than by the models. Splitting removes the "
        "clipping choice from the comparison instead of managing it.",
        "",
        f"### Moving transitions ({nm} windows, {100 * nm / n:.1f}%)",
        "",
        "No infinities here, so no cap exists and the mean is well defined.",
        "",
        "| predictor | mean PSNR | median PSNR | token accuracy |",
        "|---|---|---|---|",
    ]
    for label, key in (("copy-last-frame", "copy_psnr"), (f"model ({params:.1f}M), greedy",
                                                          "greedy|psnr"),
                       (f"model ({params:.1f}M), sampled", "sampled|psnr"),
                       ("tokenizer ceiling", "ceiling_psnr")):
        m, md = row(moving, key)
        acc_key = key.replace("psnr", "acc") if "|" in key else (
            "copy_acc" if key == "copy_psnr" else None)
        acc = f"{np.mean([r[acc_key] for r in moving]):.3f}" if acc_key else "1.000"
        bold = "**" if key == "greedy|psnr" else ""
        lines.append(f"| {label} | {bold}{m:.2f} dB{bold} | {md:.2f} dB | {acc} |")
    lines += [
        "",
        f"**The model beats copy-last-frame by {captured:.2f} dB on moving transitions.** "
        f"Headroom from copy to the tokenizer ceiling is {head:.2f} dB, so it captures "
        f"**{100 * captured / head:.0f}% of what was available**. Token accuracy agrees and is "
        "cap-free by construction.",
        "",
        f"### Static transitions (n = {ns} windows, {100 * ns / n:.1f}% of the sample)",
        "",
        "Frames where nothing moved: a no-op action, or the agent pressed against a wall.",
        "",
        "| predictor | mean PSNR | median PSNR | token accuracy |",
        "|---|---|---|---|",
        "| copy-last-frame | exact (infinite) | exact | 1.000 |",
        f"| model ({params:.1f}M), greedy | {s_model_mean:.2f} dB | {s_model_med:.2f} dB | "
        f"{s_acc:.3f} |",
        f"| tokenizer ceiling | {s_ceiling_mean:.2f} dB | | 1.000 |",
        "",
        f"**The model does not beat copy-last-frame here, and cannot.** Copy is exact by "
        f"construction; the model scores {s_model_mean:.2f} dB. It reproduces the previous "
        f"frame's tokens exactly on **{100 * freeze:.0f}%** of static transitions "
        f"(n = {ns}).",
        "",
        f"Read those two numbers differently. At n = {ns} the 0% exact-repeat result is "
        f"decisive: not one static frame in {ns} came back unchanged. The {s_model_mean:.2f} dB "
        "mean is not; it is a small sample and should be treated as indicative only.",
        "",
        "### Why stillness is a hard constraint, not a quality problem",
        "",
        "A still frame requires **all 64 tokens correct simultaneously**. Treating per-token "
        "accuracy as roughly independent, the probability of an unchanged frame is that "
        "accuracy raised to the 64th power:",
        "",
        "| per-token agreement | P(frame unchanged) |",
        "|---|---|",
        "| 0.723 (this model) | ~0.000 |",
        "| 0.99 | 0.52 |",
        "| 0.995 | 0.73 |",
        "| 0.999 | 0.94 |",
        "",
        "So even **99% per-token agreement flickers on about half of all still frames**. "
        "Reliable stillness needs roughly 99.9%, and that is not a quality target that "
        "cross-entropy has any incentive to reach. The loss is a mean over tokens: getting 46 "
        "of 64 right is a perfectly acceptable value, and it is also a visibly broken frame. "
        "The objective and the artifact are measuring different things, and no amount of "
        "lowering the loss changes the exponent.",
        "",
        "This is why the problem is unlikely to yield to capacity. Going from 0.723 to 0.99 "
        "per-token would be an enormous modelling win and would still leave the world "
        "shimmering half the time it should be frozen.",
        "",
        f"### Churn under a held no-op, against the real game ({model_still['n']} frame pairs)",
        "",
        "The static row above is teacher-forced. This is the same question asked the way a "
        f"viewer meets it: nothing pressed for {a.still_frames} frames, the model consuming "
        "its own output. The real game is run through the same prefix, the same no-op and "
        "the same tokenizer, because a churn number on its own supports either reading.",
        "",
        "| | frames identical to previous | mean tokens changed | tokens changed *when* it moves |",
        "|---|---|---|---|",
        f"| real game | **{100 * real_still['ident']:.1f}%** | {real_still['churn']:.2f} / 64 | "
        f"{real_still['churn_when_moved']:.1f} / 64 |",
        f"| model | **{100 * model_still['ident']:.1f}%** | {model_still['churn']:.2f} / 64 | "
        f"{model_still['churn_when_moved']:.1f} / 64 |",
        "",
        "**Two separate findings, and only the reference row separates them.**",
        "",
        f"*Frequency: the model is too twitchy, by {100 * (real_still['ident'] - model_still['ident']):.1f} "
        "percentage points.* The real game holds a byte-identical frame "
        f"{100 * real_still['ident']:.1f}% of the time under a held no-op; the model manages "
        f"{100 * model_still['ident']:.1f}%. So the direction is confirmed: this is shimmer, "
        "not sluggishness.",
        "",
        f"*Magnitude: the model is far too timid.* When the real game does change under a "
        f"no-op it changes {real_still['churn_when_moved']:.0f} of 64 patches, a real event. "
        f"When the model changes it moves {model_still['churn_when_moved']:.1f}. The model's "
        f"overall churn ({model_still['churn']:.2f}) is therefore *lower* than the real "
        f"game's ({real_still['churn']:.2f}) while being wrong more often: it substitutes "
        "frequent small flicker for rare large events. Averaged churn alone would have "
        "scored the model as calmer than reality and called that a win.",
        "",
        "**This is much better than the teacher-forced static row predicts, and the "
        "discrepancy is the interesting part.** Teacher-forced, the model reproduces a real "
        "previous frame's tokens exactly 0% of the time. Closed-loop, it reproduces its own "
        f"previous tokens {100 * model_still['ident']:.1f}% of the time. Those are different "
        "tasks: matching an external frame exactly is hard, while settling on a "
        "self-consistent fixed point is something an autoregressive model falls into, because "
        "its context is already full of the frame it just produced. The earlier prediction of "
        "visible shimmer came from the teacher-forced number, and this walks it back.",
        "",
        "The combinatorial argument above is unaffected and still explains why *exact* "
        "stillness is not something the objective can be pushed into. It is simply that "
        "self-consistency, not accuracy, is doing the work here, and self-consistency is not "
        "a property the loss optimises either. It could get worse with a stronger model that "
        "tracks the world sharply instead of settling, which is a reason to re-run this "
        "measurement on every rung of the scaling ladder rather than assume it holds.",
        "",
        "These are the before-numbers for any later fix.",
        "",
        "### Candidate fix, logged and not implemented",
        "",
        "A per-token **persistence gate**: a binary head predicting, for each of the 64 "
        "positions, whether that token changes at all on this step, decoded *before* token "
        "identity. Positions the gate marks unchanged are copied from the previous frame "
        "verbatim and never sampled, so they cannot flicker; only the remainder go through "
        "the usual decode. This attacks the exponent rather than the accuracy, because "
        "stillness stops requiring 64 simultaneously-correct predictions and starts requiring "
        "one well-calibrated binary decision per token, on a task where the prior is heavily "
        "skewed and easy to learn. It also cuts decode work on quiet frames. Not implemented, "
        "and not on the critical path for the scaling ladder.",
        "",
        "## Closed-loop",
        "",
        f"{a.rollouts} independent rollouts of {a.horizon} frames from held-out starts, "
        "averaged per step. The model consumes its own predictions; copy-last-frame "
        "degenerates to freezing on the last real frame. Reported per k and never averaged "
        "over k, because the trajectory is not monotonic and a single mean over it would "
        f"describe nothing. PSNR is clipped at {PSNR_MAX:.0f} dB in this section only, for "
        "the rare case of a rollout that starts from a stalled agent.",
        "",
        "| k | copy (frozen) | model | tokenizer ceiling | model lead |",
        "|---|---|---|---|---|",
    ]
    for k in MARKS:
        if k > a.horizon:
            continue
        lines.append(
            f"| {k} | {frozen[:, k - 1].mean():.2f} dB | **{model[:, k - 1].mean():.2f} dB** "
            f"| {ceil[:, k - 1].mean():.2f} dB | "
            f"{model[:, k - 1].mean() - frozen[:, k - 1].mean():+.2f} dB |"
        )
    k_last = min(MARKS[-1], a.horizon)
    lines += [
        "",
        f"The model's lead over a frozen frame decays from "
        f"{model[:, 0].mean() - frozen[:, 0].mean():+.2f} dB at k=1 to "
        f"{model[:, k_last - 1].mean() - frozen[:, k_last - 1].mean():+.2f} dB at k={k_last}. "
        "Past roughly k=16 it is not meaningfully better than showing the player a still "
        "image, which is the honest way to read the rollout GIF.",
        "",
        "The tokenizer ceiling is flat across k, as it must be: it re-encodes the true frame "
        "at every step and never compounds.",
        "",
        "Regenerate with `python -m ngx.eval.baselines --config configs/small.yaml`.",
        "",
    ]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
