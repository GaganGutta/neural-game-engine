"""Cap checkpoint retention in a pulled run directory.

    python scripts/prune_ckpts.py runs_gpu --keep 3 [--dry-run]

The trainer already prunes on the machine it runs on (see
``ngx.train.train_dynamics.prune_checkpoints``), but a sync loop that pulls the
run directory every few minutes accumulates every checkpoint that was ever
current on the far side. This walks the given roots, and in every directory
holding step-numbered ``ckpt_*.pt`` files keeps only the newest ``--keep`` of
them. ``dynamics.pt`` (best-by-val-loss) and ``final.pt`` are never touched.

Run it after each pull; it is idempotent and prints what it freed.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

STEP_CKPT = re.compile(r"^ckpt_\d+\.pt$")


def prune(roots: list[str], keep: int, dry_run: bool) -> None:
    freed = 0
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            cks = sorted(f for f in files if STEP_CKPT.match(f))
            for name in cks[:-keep] if keep > 0 else cks:
                path = Path(dirpath) / name
                size = path.stat().st_size
                freed += size
                print(f"{'would delete' if dry_run else 'delete'}  {path}  "
                      f"({size / 1e6:.0f} MB)")
                if not dry_run:
                    path.unlink()
    print(f"{'would free' if dry_run else 'freed'} {freed / 1e9:.2f} GB")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("roots", nargs="+", help="directories to walk (e.g. runs_gpu)")
    p.add_argument("--keep", type=int, default=3, help="step checkpoints to keep per run")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    prune(a.roots, a.keep, a.dry_run)


if __name__ == "__main__":
    main()
