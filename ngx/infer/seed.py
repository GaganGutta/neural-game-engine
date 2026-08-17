"""Getting the first few real frames into the model's context.

The engine predicts frame ``t+1`` from the previous ``C`` frames, so something
has to supply the first ``C``. Two sources:

``env``
    step the real game briefly and hand over. Needs VizDoom installed.
``data``
    lift a window out of a collected dataset. Lets the model be played on a
    machine that has the checkpoints but not the game.
"""

from __future__ import annotations

import os

import numpy as np


def seed_from_env(cfg: dict, context: int, burn_in: int = 30, seed: int = 0):
    """Play the real game for a moment, return its last ``context`` frames."""
    from ..envs import make_env

    env = make_env(
        cfg["data"]["env"], frame_size=64,
        frame_skip=cfg["data"]["frame_skip"], seed=seed,
    )
    rng = np.random.default_rng(seed)
    try:
        frame = env.reset()
        frames, actions = [], []
        for _ in range(burn_in + context):
            # Mostly walk forward so the handover happens mid-corridor rather
            # than facing a blank wall.
            a = 3 if rng.random() < 0.6 else int(rng.integers(env.num_actions))
            frames.append(frame)
            actions.append(a)
            frame, done = env.step(a)
            if done:
                frame = env.reset()
    finally:
        env.close()
    return np.asarray(frames[-context:]), np.asarray(actions[-context:])


def seed_from_data(root: str, context: int, seed: int = 0):
    """Take a window from a collected dataset, not crossing an episode."""
    frames = np.load(os.path.join(root, "frames.npy"), mmap_mode="r")
    actions = np.load(os.path.join(root, "actions.npy"), mmap_mode="r")
    episodes = np.load(os.path.join(root, "episodes.npy"))
    rng = np.random.default_rng(seed)
    for _ in range(200):
        i = int(rng.integers(0, len(frames) - context))
        if episodes[i] == episodes[i + context - 1]:
            return (
                np.asarray(frames[i : i + context]),
                np.asarray(actions[i : i + context]),
            )
    raise RuntimeError(f"no clean {context}-frame window found in {root}")


def get_seed(cfg: dict, context: int, source: str = "env", seed: int = 0):
    if source == "env":
        return seed_from_env(cfg, context, seed=seed)
    if source == "data":
        return seed_from_data(cfg["data"]["root"], context, seed=seed)
    raise ValueError(f"unknown seed source {source!r} (want 'env' or 'data')")
