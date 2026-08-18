"""Roll out policies and write (frame, action) pairs to disk.

Layout of an output directory::

    frames.npy      (N, S, S, 3) uint8   -- memmapped, the bulk of the bytes
    actions.npy     (N,)         uint8   -- action taken AT frame i
    episodes.npy    (N,)         int32   -- episode id, windows must not cross
    poses.npy       (N, 3)       float32 -- ground-truth (x, y, angle), eval only
    meta.json                            -- env name, action names, counts

Convention that the rest of the codebase depends on: ``actions[i]`` is the
action applied to ``frames[i]`` to produce ``frames[i + 1]``.

Workers write disjoint slices of the same preallocated memmap, so collection
scales across cores without a merge step.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from numpy.lib.format import open_memmap

from ..envs import make_env
from .policies import make_policy

#: flush the memmaps this often, in frames (~250 MB of frame data)
FLUSH_EVERY = 20_000

ARRAYS = {
    "frames": ("uint8", lambda n, s: (n, s, s, 3)),
    "actions": ("uint8", lambda n, s: (n,)),
    "episodes": ("int32", lambda n, s: (n,)),
    "poses": ("float32", lambda n, s: (n, 3)),
}


def _alloc(out: str, n: int, size: int) -> None:
    os.makedirs(out, exist_ok=True)
    for name, (dtype, shape) in ARRAYS.items():
        arr = open_memmap(
            os.path.join(out, f"{name}.npy"), mode="w+", dtype=dtype, shape=shape(n, size)
        )
        arr.flush()
        del arr


def _worker(
    out: str,
    env_name: str,
    start: int,
    count: int,
    size: int,
    frame_skip: int,
    explore_frac: float,
    seed: int,
    ep_offset: int,
    verbose: bool,
) -> int:
    frames = np.load(os.path.join(out, "frames.npy"), mmap_mode="r+")
    actions = np.load(os.path.join(out, "actions.npy"), mmap_mode="r+")
    episodes = np.load(os.path.join(out, "episodes.npy"), mmap_mode="r+")
    poses = np.load(os.path.join(out, "poses.npy"), mmap_mode="r+")

    rng = np.random.default_rng(seed)
    env = make_env(env_name, frame_size=size, frame_skip=frame_skip, seed=seed)
    policies = {k: make_policy(k, env, rng) for k in ("explore", "random")}

    i, ep = 0, ep_offset
    t0 = time.time()
    try:
        while i < count:
            pol = policies["explore" if rng.random() < explore_frac else "random"]
            pol.reset()
            frame = env.reset()
            done = False
            while not done and i < count:
                pose = env.pose()
                a = pol.act(pose)
                j = start + i
                frames[j] = frame
                actions[j] = a
                episodes[j] = ep
                poses[j] = pose if pose is not None else (np.nan,) * 3
                frame, done = env.step(a)
                i += 1
                # Flush periodically. frames.npy runs to tens of GB, so its
                # dirty pages sit in the page cache while the small arrays stay
                # resident. A hard kill then loses frame data while episode ids
                # survive, which produces a file that looks complete and is
                # silently full of black frames. Paying a flush every ~250 MB
                # is far cheaper than discovering that later.
                if i % FLUSH_EVERY == 0:
                    for arr in (frames, actions, episodes, poses):
                        arr.flush()
                if verbose and i % 5000 == 0:
                    fps = i / max(time.time() - t0, 1e-9)
                    print(f"  [w{seed}] {i}/{count} frames  {fps:6.0f} fps", flush=True)
            ep += 1
    finally:
        env.close()
        for a in (frames, actions, episodes, poses):
            a.flush()
    return ep - ep_offset


def verify(out: str, samples: int = 2000, chunks: int = 32) -> dict:
    """Check that what landed on disk is actually usable data.

    Collection is long and the output is huge, so the failure mode worth
    guarding is not a crash but a file that *looks* complete. Writes go to
    memory-mapped pages; a hard kill can flush the small arrays and lose the
    big one, leaving arrays that disagree with each other. Both halves of that
    have now happened here, so both are checked:

    ``black``
        fraction of sampled frames that are entirely black. A real frame from
        this game always has some non-zero pixel, so all-black means the write
        never reached disk.
    ``dead_action_chunks``
        contiguous regions holding only a single action value. This is the
        nastier failure, because it is invisible: the frames show an agent
        moving while the action labels say it pressed nothing. Training on that
        teaches the model to ignore its controller, which is the one thing a
        playable world model cannot afford, and no amount of looking at frames
        would reveal it.
    """
    frames = np.load(os.path.join(out, "frames.npy"), mmap_mode="r")
    actions = np.load(os.path.join(out, "actions.npy"), mmap_mode="r")
    idx = np.linspace(0, len(frames) - 1, min(samples, len(frames))).astype(int)
    black = sum(int(np.asarray(frames[i]).max()) == 0 for i in idx) / len(idx)

    bounds = np.linspace(0, len(actions), chunks + 1).astype(int)
    dead = [
        (int(bounds[c]), int(bounds[c + 1]))
        for c in range(chunks)
        if len(np.unique(np.asarray(actions[bounds[c] : bounds[c + 1]]))) <= 1
    ]
    return {"black": black, "dead_action_chunks": dead}


def collect(
    out: str,
    env_name: str = "my_way_home",
    num_frames: int = 120_000,
    size: int = 64,
    frame_skip: int = 4,
    explore_frac: float = 0.75,
    workers: int = 1,
    seed: int = 0,
) -> None:
    probe = make_env(env_name, frame_size=size, frame_skip=frame_skip)
    num_actions, action_names = probe.num_actions, probe.action_names
    probe.close()

    print(
        f"collecting {num_frames:,} frames from {env_name} "
        f"({num_actions} actions, {size}x{size}, skip={frame_skip}) "
        f"-> {out}  [{workers} worker(s)]"
    )
    _alloc(out, num_frames, size)

    # Each worker owns a contiguous slice and a disjoint episode-id range.
    bounds = np.linspace(0, num_frames, workers + 1).astype(int)
    jobs = [
        (
            out,
            env_name,
            int(bounds[w]),
            int(bounds[w + 1] - bounds[w]),
            size,
            frame_skip,
            explore_frac,
            seed + w,
            w * 1_000_000,
            w == 0,
        )
        for w in range(workers)
    ]

    t0 = time.time()
    if workers == 1:
        _worker(*jobs[0])
    else:
        import multiprocessing as mp

        with mp.Pool(workers) as pool:
            pool.starmap(_worker, jobs)
    dt = time.time() - t0

    meta = {
        "env": env_name,
        "num_frames": int(num_frames),
        "frame_size": int(size),
        "frame_skip": int(frame_skip),
        "num_actions": int(num_actions),
        "action_names": list(action_names),
        "explore_frac": explore_frac,
        "seed": seed,
        "collect_seconds": round(dt, 1),
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    gb = num_frames * size * size * 3 / 1e9
    print(
        f"done in {dt / 60:.1f} min  ({num_frames / max(dt, 1e-9):,.0f} frames/s, {gb:.2f} GB)"
    )
    report = verify(out)
    problems = []
    if report["black"] > 0.001:
        problems.append(f"{100 * report['black']:.1f}% of sampled frames are all black")
    if report["dead_action_chunks"]:
        problems.append(
            f"{len(report['dead_action_chunks'])} region(s) hold a single action value, "
            f"first at frames {report['dead_action_chunks'][0]}"
        )
    if problems:
        raise RuntimeError(
            "; ".join(problems) + f". Writes did not reach disk, so {out} is not usable. "
            "Re-collect rather than training on it."
        )
    print(f"integrity check passed ({100 * report['black']:.2f}% black frames, "
          f"no single-action regions)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default="data/my_way_home")
    p.add_argument("--env", default="my_way_home")
    p.add_argument("--frames", type=int, default=120_000)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument(
        "--explore-frac",
        type=float,
        default=0.75,
        help="fraction of episodes driven by the explorer rather than sticky-random",
    )
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    collect(
        a.out, a.env, a.frames, a.size, a.frame_skip, a.explore_frac, a.workers, a.seed
    )


if __name__ == "__main__":
    main()
