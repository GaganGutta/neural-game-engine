"""B2: what states and actions does the collected data actually contain?

    python -m ngx.eval.coverage --roots data/my_way_home --labels before
    python -m ngx.eval.coverage --roots data/a data/b --labels before after

A world model can only learn dynamics that appear in its data, so "how much of
the map did the policy see, and did it press every button" is a prerequisite
question rather than a nice-to-have.

Raw counts mislead here. A policy that spends 90% of its steps spinning in one
corner still racks up a large number of *distinct* cells over a long run, so
the headline metric is **effective cells**: ``exp(entropy)`` of the visit
distribution, which is the number of equally-visited cells that would produce
the same spread. Ten cells visited uniformly scores 10; a thousand cells where
one holds 99% of the visits scores barely above 1.

The same reasoning applies to actions, reported as normalised entropy so that
1.00 is a perfectly uniform controller and 0.00 is one button held forever.
Neither extreme is the goal: uniform actions are not what a player does, but a
near-zero score means whole behaviours are missing from the data.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load(root: str, cell: float):
    poses = np.load(os.path.join(root, "poses.npy"))
    actions = np.load(os.path.join(root, "actions.npy"))
    episodes = np.load(os.path.join(root, "episodes.npy"))
    meta = json.load(open(os.path.join(root, "meta.json")))
    ok = np.isfinite(poses).all(1)
    cells = np.floor(poses[ok][:, :2] / cell).astype(np.int64)
    return cells, actions, episodes, meta


def entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0].astype(np.float64)
    p /= p.sum()
    return float(-(p * np.log(p)).sum())


def stats(root: str, cell: float) -> dict:
    cells, actions, episodes, meta = load(root, cell)
    _, counts = np.unique(cells, axis=0, return_counts=True)
    order = np.sort(counts)[::-1]
    top = max(1, len(order) // 10)
    a_counts = np.bincount(actions, minlength=meta["num_actions"]).astype(np.float64)
    return {
        "root": root,
        "frames": int(len(actions)),
        "episodes": int(len(np.unique(episodes))),
        "cells": int(len(counts)),
        "eff_cells": float(np.exp(entropy(counts))),
        "top10_share": float(order[:top].sum() / order.sum()),
        "action_counts": a_counts,
        "action_entropy": float(entropy(a_counts) / np.log(len(a_counts))),
        "action_min_share": float((a_counts / a_counts.sum()).min()),
        "cells_xy": cells,
        "action_names": meta["action_names"],
    }


def plot(all_stats: list[dict], labels: list[str], path: str, cell: float) -> None:
    n = len(all_stats)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 9), squeeze=False)
    for i, (s, label) in enumerate(zip(all_stats, labels)):
        xy = s["cells_xy"]
        x0, y0 = xy.min(0)
        grid = np.zeros((xy[:, 1].max() - y0 + 1, xy[:, 0].max() - x0 + 1))
        np.add.at(grid, (xy[:, 1] - y0, xy[:, 0] - x0), 1)

        ax = axes[0][i]
        im = ax.imshow(np.log10(grid + 1), origin="lower", cmap="magma", aspect="equal")
        ax.set_title(f"{label}: map-cell visitation\n"
                     f"{s['cells']} cells, {s['eff_cells']:.0f} effective",
                     fontsize=11)
        ax.set_xlabel(f"x / {cell:g} map units")
        ax.set_ylabel(f"y / {cell:g} map units")
        fig.colorbar(im, ax=ax, label="log10(visits + 1)", fraction=0.046)

        ax = axes[1][i]
        share = s["action_counts"] / s["action_counts"].sum()
        ax.bar(range(len(share)), share, color="#4C8BF5")
        ax.set_xticks(range(len(share)))
        ax.set_xticklabels(s["action_names"], rotation=30, ha="right", fontsize=9)
        ax.axhline(1 / len(share), ls="--", c="#888", lw=1, label="uniform")
        ax.set_ylim(0, max(share.max() * 1.2, 1.4 / len(share)))
        ax.set_title(f"{label}: action distribution\n"
                     f"normalised entropy {s['action_entropy']:.3f}", fontsize=11)
        ax.set_ylabel("share of steps")
        ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--roots", nargs="+", default=["data/my_way_home"])
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--cell", type=float, default=64.0, help="map units per grid cell")
    p.add_argument("--plot", default="assets/coverage.png")
    p.add_argument("--out", default="docs/COVERAGE.md")
    a = p.parse_args()

    labels = a.labels or [os.path.basename(r) for r in a.roots]
    if len(labels) != len(a.roots):
        raise SystemExit("--labels must match --roots in length")

    all_stats = [stats(r, a.cell) for r in a.roots]
    for s, label in zip(all_stats, labels):
        print(f"{label}: {s['frames']:,} frames, {s['episodes']} episodes")
        print(f"  cells {s['cells']}   effective {s['eff_cells']:.1f}   "
              f"top-10% share {s['top10_share']:.3f}")
        print(f"  action entropy {s['action_entropy']:.3f}   "
              f"rarest action {s['action_min_share']:.3f}")
        print("  " + "  ".join(
            f"{n}:{c / s['action_counts'].sum():.3f}"
            for n, c in zip(s["action_names"], s["action_counts"])))

    plot(all_stats, labels, a.plot, a.cell)
    print(f"wrote {a.plot}")

    lines = [
        "# B2: data coverage",
        "",
        f"Grid cell = {a.cell:g} map units. **Effective cells** is `exp(entropy)` of the "
        "visit distribution: the number of equally-visited cells that would produce the "
        "same spread. It is the honest version of 'how much of the map did we see', "
        "because distinct-cell counts reward a policy that touches a thousand cells once "
        "and then spins in a corner.",
        "",
        "| metric | " + " | ".join(labels) + " |",
        "|---" * (len(labels) + 1) + "|",
    ]
    rows = [
        ("frames", lambda s: f"{s['frames']:,}"),
        ("episodes", lambda s: f"{s['episodes']}"),
        ("distinct cells", lambda s: f"{s['cells']}"),
        ("**effective cells**", lambda s: f"**{s['eff_cells']:.1f}**"),
        ("top-10% of cells hold", lambda s: f"{100 * s['top10_share']:.1f}% of visits"),
        ("**action entropy** (1.0 = uniform)", lambda s: f"**{s['action_entropy']:.3f}**"),
        ("rarest action share", lambda s: f"{100 * s['action_min_share']:.1f}%"),
    ]
    for name, fn in rows:
        lines.append(f"| {name} | " + " | ".join(fn(s) for s in all_stats) + " |")
    lines += [
        "",
        f"![coverage]({os.path.relpath(a.plot, os.path.dirname(a.out))})",
        "",
        "Top row is visitation on a log colour scale, so a uniform-looking map is genuinely "
        "uniform and a few bright cells against a dark field is a policy that parked. "
        "Bottom row is the action distribution against the uniform line.",
        "",
        "Regenerate with `python -m ngx.eval.coverage --roots "
        + " ".join(a.roots) + " --labels " + " ".join(labels) + "`.",
        "",
    ]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
