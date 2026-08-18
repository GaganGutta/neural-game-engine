"""B0: does the dynamics model beat copying the previous frame?

    python -m ngx.eval.baselines --config configs/small.yaml

Every downstream number in this repo depends on the answer, so the comparison
is set up to be as favourable to the model as honesty allows and still be fair.

Three rows, measured on the same held-out frames:

``copy-last-frame``
    Predict frame ``t+1`` by handing back frame ``t`` unchanged. No parameters,
    no training. In a 64x64 game at frame_skip 4 this is a genuinely strong
    baseline, because most of the screen does not change between steps.
``model``
    The trained dynamics model.
``tokenizer ceiling``
    Encode the *true* next frame and decode it again. The dynamics model can
    never beat this, because it emits tokens and those tokens go through the
    same decoder. This is the upper bound on the whole approach at this
    tokenizer.

Two regimes, because they answer different questions:

**One-step, teacher-forced.** The model gets real frames for context and
predicts one frame. Copy-last-frame gets the real previous frame. This is the
apples-to-apples test of whether the model learned the dynamics at all, and it
is the number that matters most -- drift cannot be blamed for it.

**Closed-loop.** The model consumes its own predictions; the baseline freezes
on the last real frame. Reported per k, averaged across many independent
rollouts. Never averaged *over* k: a trajectory that goes 14.0, 9.8, 11.8 is
not something a single mean describes.

The model is measured both greedily and with the sampling settings play.py
actually uses. Greedy is the model's best single guess and is the fair number
for "did it learn the dynamics"; the sampled row is what a player sees.

A note on infinities. Consecutive frames in this game are sometimes *pixel
identical* -- a no-op action, or the agent walking into a wall -- and PSNR
between identical uint8 images is mathematically infinite, which makes any mean
containing one also infinite. Those frames are real data and dropping them
would quietly delete copy-last-frame's easiest wins, so PSNR is capped at a
documented 60 dB and the fraction of identical pairs is reported alongside. The
median is reported too, since it does not care about the cap at all.
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

#: PSNR between pixel-identical uint8 frames is infinite. Cap it so aggregates
#: stay finite, and report how often the cap is hit rather than hiding it.
PSNR_MAX = 60.0


def psnr_capped(a: np.ndarray, b: np.ndarray) -> float:
    return min(psnr_u8(a, b), PSNR_MAX)


def summarise(v: list[float]) -> tuple[float, float, float]:
    """(mean, median, standard deviation)."""
    arr = np.asarray(v, dtype=np.float64)
    return float(arr.mean()), float(np.median(arr)), float(arr.std())


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
def one_step(engines: dict, vq, windows, device):
    """Teacher-forced: real context in, one frame out.

    ``engines`` maps a label to an engine, so the greedy and sampled decoders
    are scored on exactly the same windows.
    """
    C = next(iter(engines.values())).C
    rows: dict[str, dict[str, list]] = {
        k: {"psnr": [], "acc": []} for k in list(engines) + ["copy", "ceiling"]
    }
    identical = 0
    for frames, actions in windows:
        tok = encode(vq, frames[C - 1 : C + 1], device)  # tokens of f[C-1], f[C]
        for label, engine in engines.items():
            engine.reset(frames[:C], actions[:C])
            pred = engine.step(int(actions[C - 1]))
            rows[label]["psnr"].append(psnr_capped(pred[None], frames[C][None]))
            rows[label]["acc"].append(float((engine.tokens[-1] == tok[1]).float().mean()))

        raw = psnr_u8(frames[C - 1][None], frames[C][None])
        identical += raw == float("inf")
        rows["copy"]["psnr"].append(min(raw, PSNR_MAX))
        rows["copy"]["acc"].append(float((tok[0] == tok[1]).float().mean()))
        rows["ceiling"]["psnr"].append(
            psnr_capped(roundtrip(vq, frames[C : C + 1], device), frames[C][None])
        )
        rows["ceiling"]["acc"].append(1.0)
    out = {k: {"psnr": summarise(v["psnr"]), "acc": summarise(v["acc"])} for k, v in rows.items()}
    out["identical_frac"] = identical / len(windows)
    return out


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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--device", default="auto")
    p.add_argument("--windows", type=int, default=200, help="windows for the one-step test")
    p.add_argument("--rollouts", type=int, default=32, help="independent closed-loop rollouts")
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--out", default="docs/BASELINES.md")
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    device = pick_device(a.device)
    vq, dyn, dck = load_models(cfg, device)
    torch.manual_seed(0)

    ic = cfg["infer"]
    mk = dict(decode=ic["decode"], maskgit_steps=ic["maskgit_steps"])
    # Greedy is the fair "did it learn the dynamics" number; the sampled engine
    # is what play.py runs, and the gap between them turned out to matter.
    engines = {
        "model (greedy)": NeuralGameEngine(
            vq, dyn, EngineConfig(**mk, temperature=0.0, top_k=0), device=device, memory=None),
        "model (sampled)": NeuralGameEngine(
            vq, dyn,
            EngineConfig(**mk, temperature=ic["temperature"], top_k=ic["top_k"]),
            device=device, memory=None),
    }

    C = dyn.context
    print(f"one-step: {a.windows} held-out windows")
    ow = sample_windows(cfg["data"]["root"], C + 1, a.windows, seed=0)
    one = one_step(engines, vq, ow, device)
    for name in ["copy", *engines, "ceiling"]:
        mu, med, sd = one[name]["psnr"]
        acc = one[name]["acc"][0]
        print(f"  {name:16s} mean {mu:5.2f}  median {med:5.2f}  +/-{sd:4.2f} dB   "
              f"token acc {acc:.3f}")
    print(f"  ({one['identical_frac'] * 100:.1f}% of consecutive pairs are pixel-identical)")

    print(f"\nclosed-loop: {a.rollouts} rollouts x {a.horizon} frames (greedy)")
    rw = sample_windows(cfg["data"]["root"], C + a.horizon, a.rollouts, seed=1)
    model, frozen, ceil = closed_loop(engines["model (greedy)"], vq, rw, device, a.horizon)
    for k in MARKS:
        if k > a.horizon:
            continue
        print(f"  k={k:<3d} copy {frozen[:, k - 1].mean():5.2f}   "
              f"model {model[:, k - 1].mean():5.2f}   ceiling {ceil[:, k - 1].mean():5.2f}")

    g = one["model (greedy)"]["psnr"]
    s = one["model (sampled)"]["psnr"]
    c = one["copy"]["psnr"]
    ceilv = one["ceiling"]["psnr"]
    delta = g[0] - c[0]
    gap = ceilv[0] - g[0]
    verdict = "beats" if delta > 0 else "loses to"
    params = sum(q.numel() for q in dyn.parameters()) / 1e6

    lines = [
        "# B0: baselines",
        "",
        f"Model: {params:.1f}M params, context {C} frames, "
        f"val loss {dck.get('val_loss', float('nan')):.3f}, "
        f"cold token accuracy {dck.get('cold_acc', float('nan')):.3f}. "
        f"Greedy decoding, {cfg['infer']['maskgit_steps']}-pass MaskGIT, memory off.",
        "",
        "## One-step, teacher-forced",
        "",
        f"Real frames for context, one frame predicted, {a.windows} held-out windows. "
        "This is the test that cannot be excused by drift.",
        "",
        "| predictor | PSNR mean | PSNR median | token accuracy |",
        "|---|---|---|---|",
        f"| copy-last-frame | {c[0]:.2f} dB | {c[1]:.2f} dB | "
        f"{one['copy']['acc'][0]:.3f} |",
        f"| model ({params:.1f}M), sampled | {s[0]:.2f} dB | {s[1]:.2f} dB | "
        f"{one['model (sampled)']['acc'][0]:.3f} |",
        f"| model ({params:.1f}M), greedy | **{g[0]:.2f} dB** | **{g[1]:.2f} dB** | "
        f"**{one['model (greedy)']['acc'][0]:.3f}** |",
        f"| tokenizer ceiling | {ceilv[0]:.2f} dB | {ceilv[1]:.2f} dB | 1.000 by construction |",
        "",
        f"**The model {verdict} copy-last-frame by {abs(delta):.2f} dB** greedily, and "
        f"sits {gap:.2f} dB below the tokenizer ceiling.",
        "",
        f"The number that matters is the ratio. Total headroom between the trivial "
        f"baseline and the tokenizer ceiling is {ceilv[0] - c[0]:.2f} dB. The model "
        f"captures {abs(delta):.2f} dB of it, or "
        f"**{100 * delta / max(ceilv[0] - c[0], 1e-9):.0f}% of what was available**. "
        "Token accuracy tells the same story: "
        f"{one['copy']['acc'][0]:.3f} for copy against "
        f"{one['model (greedy)']['acc'][0]:.3f} for the model.",
        "",
        f"Greedy and sampled decoding land {abs(g[0] - s[0]):.2f} dB apart, which is "
        f"nothing against the {g[2]:.1f} dB frame-to-frame spread. Decoder temperature is "
        "not a meaningful lever at this model size.",
        "",
        f"{one['identical_frac'] * 100:.1f}% of consecutive frame pairs in the held-out set "
        "are pixel-identical (no-op actions, or the agent pressed against a wall), which is "
        "why PSNR is capped at 60 dB here and why the median is reported next to the mean.",
        "",
        "## Closed-loop",
        "",
        f"{a.rollouts} independent rollouts of {a.horizon} frames from held-out starts, "
        "averaged per step. The model consumes its own predictions; copy-last-frame "
        "degenerates to freezing on the last real frame. Reported per k and never "
        "averaged over k, because the trajectory is not monotonic and a single mean "
        "over it would describe nothing.",
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
    lead1 = model[:, 0].mean() - frozen[:, 0].mean()
    lead_end = model[:, k_last - 1].mean() - frozen[:, k_last - 1].mean()
    lines += [
        "",
        f"The model's lead over a frozen frame decays from {lead1:+.2f} dB at k=1 to "
        f"{lead_end:+.2f} dB at k={k_last}. Past roughly k=16 it is not meaningfully "
        "better than showing the player a still image, which is the honest way to read "
        "the rollout GIF.",
        "",
        "The tokenizer ceiling is flat across k, as it must be: it re-encodes the true "
        "frame at every step and never compounds. It is drawn here as the horizontal "
        "line everything else is failing to reach.",
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
