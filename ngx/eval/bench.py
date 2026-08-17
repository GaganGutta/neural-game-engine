"""Stage 4: measure fps and memory before and after every optimisation.

    python -m ngx.eval.bench --config configs/small.yaml

Rows are cumulative -- each one adds a single change to the row above -- so the
table reads as a changelog with numbers attached. Writes docs/BENCHMARKS.md.

Protocol, because a benchmark nobody can reproduce is decoration:

* Greedy decoding (temperature 0). Sampling would make every row produce a
  different rollout and the comparison meaningless.
* Identical seed frames and identical action sequence for every row.
* Warmup frames are discarded, which matters enormously for ``torch.compile``.
* Each row reports PSNR against the fp32 reference rollout, so a row that buys
  speed by wrecking the output cannot hide.
* A row that fails (no compiler, wrong device) is reported as unavailable
  rather than silently skipped, and the sweep continues from the last good
  configuration.
* A change that turns out to be *slower* is reported and then reverted, so
  later rows build on the best configuration rather than dragging a known
  regression along. Both outcomes appear in the table; not every optimisation
  survives contact with a given machine.
"""

from __future__ import annotations

import argparse
import os
import platform
import time

import numpy as np
import torch

from ..config import load_config, pick_device
from ..envs import make_env
from ..infer.engine import EngineConfig, NeuralGameEngine
from ..infer.load import load_models
from ..infer.quantize import weight_bytes

# (label, EngineConfig deltas). Each is applied on top of the best configuration
# found so far. int8 forces fp32 back on: PyTorch's dynamic quantised kernels
# take float activations, and feeding them autocast output raises outright.
STEPS: list[tuple[str, dict]] = [
    ("raster AR, no KV cache", dict(decode="raster", use_cache=False, dtype="fp32")),
    ("+ KV cache", dict(use_cache=True)),
    ("+ MaskGIT parallel decode", dict(decode="maskgit")),
    ("+ bf16 autocast", dict(dtype="bf16")),
    ("+ torch.compile", dict(compile=True)),
    ("+ int8 dynamic quant", dict(int8=True, dtype="fp32")),
]


def _rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return None


def seed_frames(cfg: dict, context: int, n_burn: int = 40) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Real frames to prime the context, plus a fixed action sequence."""
    env = make_env(cfg["data"]["env"], frame_size=64, frame_skip=cfg["data"]["frame_skip"], seed=0)
    rng = np.random.default_rng(0)
    try:
        frame = env.reset()
        frames, actions = [], []
        for i in range(n_burn + context):
            a = int(rng.integers(env.num_actions))
            frames.append(frame)
            actions.append(a)
            frame, done = env.step(a)
            if done:
                frame = env.reset()
        plan = [int(rng.integers(env.num_actions)) for _ in range(4096)]
    finally:
        env.close()
    return np.asarray(frames[-context:]), np.asarray(actions[-context:]), plan


def run_row(
    vq, dyn, ecfg: EngineConfig, device, seeds, acts, plan, frames: int, warmup: int, budget: float
):
    engine = NeuralGameEngine(vq, dyn, ecfg, device=device, memory=None)
    engine.reset(seeds, acts)
    for i in range(warmup):
        engine.step(plan[i])

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    rss0 = _rss_mb()

    out, t0, n = [], time.perf_counter(), 0
    for i in range(frames):
        out.append(engine.step(plan[warmup + i]))
        n += 1
        if time.perf_counter() - t0 > budget:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    rss1 = _rss_mb()
    peak = (
        torch.cuda.max_memory_allocated() / 1e6
        if device.type == "cuda"
        else (max(rss0 or 0, rss1 or 0) if rss1 is not None else None)
    )
    passes = dyn.L if ecfg.decode == "raster" else ecfg.maskgit_steps
    return {
        "fps": n / dt,
        "ms": 1000 * dt / n,
        "frames": n,
        "passes": passes,
        "weights_mb": weight_bytes(engine.model) / 1e6,
        "mem_mb": peak,
        "rollout": np.asarray(out),
    }


def psnr_u8(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    mse = np.mean((a[:n].astype(np.float64) - b[:n].astype(np.float64)) ** 2)
    return float("inf") if mse < 1e-9 else 10 * np.log10(255.0**2 / mse)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--device", default="auto")
    p.add_argument("--frames", type=int, default=30, help="timed frames per row")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--budget", type=float, default=90.0, help="seconds cap per row")
    p.add_argument("--out", default="docs/BENCHMARKS.md")
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    device = pick_device(a.device)
    vq, dyn, dck = load_models(cfg, device)
    torch.manual_seed(0)

    ctx = dyn.context
    seeds, acts, plan = seed_frames(cfg, ctx)

    # Greedy so rows are comparable; MaskGIT step count comes from the config.
    base = EngineConfig(
        temperature=0.0, top_k=0, maskgit_steps=cfg.get("infer", {}).get("maskgit_steps", 8)
    )
    cur = dict(base.__dict__)

    print(f"benchmarking on {device} ({platform.processor() or platform.machine()})")
    rows, ref, best_cfg, best_fps = [], None, dict(cur), 0.0
    for label, delta in STEPS:
        trial = {**best_cfg, **delta}
        try:
            r = run_row(vq, dyn, EngineConfig(**trial), device, seeds, acts, plan,
                        a.frames, a.warmup, a.budget)
        except Exception as e:  # noqa: BLE001 - a failed row is a result, not a crash
            msg = str(e).split("\n")[0][:70]
            print(f"  {label:28s} unavailable: {msg}")
            rows.append({"label": label, "error": msg})
            continue
        if ref is None:
            ref = r["rollout"]
        r["psnr"] = psnr_u8(r["rollout"], ref)
        r["label"] = label
        # Keep the change only if it actually made things faster. Later rows
        # then build on the best configuration rather than on a regression.
        r["kept"] = r["fps"] >= best_fps
        if r["kept"]:
            best_cfg, best_fps = trial, r["fps"]
        rows.append(r)
        print(
            f"  {label:28s} {r['fps']:7.2f} fps  {r['ms']:8.1f} ms  "
            f"{r['passes']:3d} passes  psnr {r['psnr']:5.1f} dB  "
            f"{'kept' if r['kept'] else 'REVERTED (slower)'}  ({r['frames']} frames)"
        )

    first_fps = next((r["fps"] for r in rows if "fps" in r), 1.0)
    lines = [
        "# Benchmarks",
        "",
        f"- device: `{device}` ({platform.processor() or platform.machine()})",
        f"- torch: `{torch.__version__}`, platform: `{platform.platform()}`",
        f"- model: {sum(p.numel() for p in dyn.parameters()) / 1e6:.1f}M params, "
        f"context {ctx} frames, {dyn.L} tokens/frame",
        f"- greedy decoding, {a.warmup} warmup frames discarded, "
        f"up to {a.frames} timed frames per row (cap {a.budget:g}s)",
        "",
        "Each row applies one change on top of the fastest configuration so far. A "
        "change that measures slower is reverted, and says so. PSNR is against the "
        "first row's rollout, so a change that buys speed by degrading the output "
        "cannot hide in this table.",
        "",
        "| step | fps | ms/frame | passes/frame | vs. row 1 | weights | peak mem | PSNR vs ref | |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['label']} | unavailable | | | | | | | _{r['error']}_ |")
            continue
        mem = f"{r['mem_mb']:.0f} MB" if r["mem_mb"] else "n/a"
        psnr = "reference" if r["psnr"] == float("inf") else f"{r['psnr']:.1f} dB"
        lines.append(
            f"| {r['label']} | **{r['fps']:.2f}** | {r['ms']:.1f} | {r['passes']} | "
            f"{r['fps'] / first_fps:.1f}x | {r['weights_mb']:.1f} MB | {mem} | {psnr} | "
            f"{'kept' if r['kept'] else 'reverted'} |"
        )
    lines += [
        "",
        "`peak mem` is process RSS on CPU and peak allocated VRAM on CUDA; on CPU it "
        "includes the interpreter and both models, so treat it as an envelope rather "
        "than a model footprint. `weights` is the dynamics model's parameter bytes.",
        "",
        "A PSNR of `reference` on the KV-cache row is the point of that row: caching "
        "the prefix is an exact transformation, and the identical rollout is the proof.",
        "",
        "Regenerate with `python -m ngx.eval.bench --config configs/small.yaml`.",
        "",
    ]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
