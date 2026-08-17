"""Render the README demo.

    python scripts/make_gif.py --frames 120

Left pane is the real game, right pane is the model, both driven by the same
action sequence from the same starting frames. Side by side rather than model
alone, because a model-only clip proves nothing -- the interesting claim is
that the right pane tracks the left one without a game engine underneath it.

The divider tint is the live PSNR between the panes, so drift is visible rather
than asserted.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngx.config import load_config, pick_device  # noqa: E402
from ngx.eval.bench import psnr_u8  # noqa: E402
from ngx.eval.drift import reference_trajectory  # noqa: E402
from ngx.infer.load import load_engine  # noqa: E402

BAR = 34
GAP = 8
BG = (16, 16, 20)


def _label(canvas, text, x, y, color=(238, 238, 244), scale=0.5):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def compose(real, fake, scale: int, step: int, total: int, retrieved: int) -> np.ndarray:
    s = 64 * scale
    up = lambda im: cv2.resize(im, (s, s), interpolation=cv2.INTER_NEAREST)  # noqa: E731
    h, w = s + BAR, s * 2 + GAP
    canvas = np.full((h, w, 3), BG, np.uint8)
    canvas[BAR:, :s] = up(real)
    canvas[BAR:, s + GAP :] = up(fake)

    _label(canvas, "REAL GAME (VizDoom)", 8, 22, (150, 150, 165))
    _label(canvas, "WORLD MODEL (no game engine)", s + GAP + 8, 22, (120, 220, 140))

    d = psnr_u8(real[None], fake[None])
    tag = f"frame {step:3d}/{total}   PSNR {d:4.1f} dB"
    if retrieved:
        tag += f"   memory +{retrieved}"
    _label(canvas, tag, w - 250, 22, (150, 150, 165), 0.42)
    return canvas


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--device", default="auto")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="assets/demo.gif")
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    device = pick_device(a.device)
    torch.manual_seed(a.seed)

    engine = load_engine(cfg, device=device)
    C = engine.C

    print(f"rolling the real game for {a.frames + C} frames...")
    frames, actions, _, _ = reference_trajectory(cfg, a.frames + C + 2, a.seed)
    n = min(a.frames, len(frames) - C)

    engine.reset(frames[:C], actions[:C])
    out = []
    for k in range(n):
        fake = engine.step(int(actions[C - 1 + k]))
        out.append(compose(frames[C + k], fake, a.scale, k + 1, n, engine.last_retrieved))
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{n}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    imageio.mimsave(a.out, out, duration=1000 / a.fps, loop=0)
    mb = os.path.getsize(a.out) / 1e6
    print(f"wrote {a.out}  ({n} frames, {mb:.1f} MB)")

    still = a.out.rsplit(".", 1)[0] + ".png"
    cv2.imwrite(still, cv2.cvtColor(out[len(out) // 2], cv2.COLOR_RGB2BGR))
    print(f"wrote {still}")


if __name__ == "__main__":
    main()
