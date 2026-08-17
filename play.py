"""Play inside the world model.

    python play.py

Controls: W / arrow-up to walk, A and D (or left/right) to turn, hold W with a
turn to strafe around a corner. R reseeds from the real game, M toggles
retrieval memory, TAB switches between MaskGIT and raster decoding, ESC quits.

The real game is used for exactly one thing: supplying the handful of frames
that prime the context at startup (and on R). After that every pixel on screen
came out of the network.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ngx.config import load_config, pick_device  # noqa: E402
from ngx.infer.load import load_engine  # noqa: E402
from ngx.infer.seed import get_seed  # noqa: E402

BG = (12, 12, 16)
FG = (235, 235, 240)
DIM = (130, 130, 145)
HOT = (120, 220, 140)


def action_from_keys(keys, names: tuple[str, ...]) -> int:
    """Map held keys to the closest action this world actually offers."""
    idx = {n: i for i, n in enumerate(names)}
    fwd = keys[pygame.K_w] or keys[pygame.K_UP]
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
    fire = keys[pygame.K_SPACE]

    order: list[str] = []
    if fire:
        order.append("fire")
    if fwd and left:
        order += ["fwd+L", "fwd"]
    elif fwd and right:
        order += ["fwd+R", "fwd"]
    elif fwd:
        order.append("fwd")
    elif left:
        order += ["turn L", "strafe L"]
    elif right:
        order += ["turn R", "strafe R"]
    order.append("-")
    for name in order:
        if name in idx:
            return idx[name]
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--device", default="auto")
    p.add_argument("--scale", type=int, default=8, help="pixel scale (64x64 -> 512x512)")
    p.add_argument("--seed-from", default="env", choices=("env", "data"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps-cap", type=float, default=0.0, help="0 = run as fast as it can")
    p.add_argument("--max-frames", type=int, default=0, help="quit after N frames (0 = never)")
    p.add_argument("--auto", action="store_true",
                   help="drive with random actions instead of the keyboard; with "
                        "SDL_VIDEODRIVER=dummy this makes the play loop testable headless")
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    device = pick_device(a.device)
    print(f"loading engine on {device}...")
    engine = load_engine(cfg, device=device)
    names = engine.action_names
    print(f"context {engine.C} frames, {engine.L} tokens/frame, decode={engine.cfg.decode}")

    seeds, seed_acts = get_seed(cfg, engine.C, a.seed_from, a.seed)
    engine.reset(seeds, seed_acts)

    pygame.init()
    pygame.display.set_caption("neural game engine -- no game engine running")
    S = 64 * a.scale
    HUD = 96
    screen = pygame.display.set_mode((S, S + HUD))
    font = pygame.font.SysFont("consolas,menlo,monospace", 15)
    big = pygame.font.SysFont("consolas,menlo,monospace", 19, bold=True)
    clock = pygame.time.Clock()

    recent = deque(maxlen=30)
    frame = engine.frame
    running, step = True, 0
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    seeds, seed_acts = get_seed(cfg, engine.C, a.seed_from, a.seed + step + 1)
                    engine.reset(seeds, seed_acts)
                    frame = engine.frame
                elif ev.key == pygame.K_m:
                    engine.memory = None if engine.memory else _fresh_memory(cfg, engine, device)
                elif ev.key == pygame.K_TAB:
                    engine.cfg.decode = "raster" if engine.cfg.decode == "maskgit" else "maskgit"

        if a.auto:
            action = int(np.random.default_rng(step).integers(len(names)))
        else:
            action = action_from_keys(pygame.key.get_pressed(), names)
        t0 = time.perf_counter()
        frame = engine.step(action)
        recent.append(time.perf_counter() - t0)
        step += 1

        surf = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        screen.fill(BG)
        screen.blit(pygame.transform.scale(surf, (S, S)), (0, 0))

        fps = 1.0 / max(np.mean(recent), 1e-9)
        passes = engine.L if engine.cfg.decode == "raster" else engine.cfg.maskgit_steps
        y = S + 10
        screen.blit(big.render(f"{fps:5.1f} fps", True, FG), (12, y))
        screen.blit(font.render(f"action: {names[action]:8s}", True, HOT), (140, y + 3))
        mem = "on" if engine.memory else "off"
        hit = engine.last_retrieved
        screen.blit(
            font.render(
                f"decode {engine.cfg.decode} ({passes} passes)   memory {mem}"
                + (f"  <- recalled {hit}" if hit else ""),
                True, HOT if hit else DIM,
            ),
            (12, y + 30),
        )
        screen.blit(
            font.render("W/A/D move   R reseed   M memory   TAB decode   ESC quit", True, DIM),
            (12, y + 56),
        )
        pygame.display.flip()
        if a.fps_cap > 0:
            clock.tick(a.fps_cap)
        if a.max_frames and step >= a.max_frames:
            running = False

    pygame.quit()
    print(f"played {step} frames entirely inside the model")


def _fresh_memory(cfg, engine, device):
    from ngx.infer.load import build_memory

    return build_memory(
        {**cfg, "memory": {**cfg.get("memory", {}), "enabled": True}},
        engine.vq, engine.model, device,
    )


if __name__ == "__main__":
    main()
