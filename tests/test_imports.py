"""Every module in the package must import, and every entry point must exist.

This exists because of a real shipped bug. `.gitignore` carried a bare `data/`
to keep datasets out of the repo, and git matches that at any depth, so
`ngx/data/` -- collection, tokenization, the Dataset classes -- was silently
never committed. The public repo looked fine: `play.py` ran and the whole test
suite passed from a fresh clone, because neither touches `ngx.data`. Every
training and evaluation entry point was broken.

Walking the package and importing everything catches a missing file, a syntax
error, and a broken relative import in one cheap test.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ngx  # noqa: E402

#: modules exposing a CLI the README or scripts/reproduce.sh invokes
ENTRY_POINTS = [
    "ngx.data.collect",
    "ngx.data.tokenize",
    "ngx.train.train_vqvae",
    "ngx.train.train_dynamics",
    "ngx.train.overfit",
    "ngx.eval.bench",
    "ngx.eval.drift",
    "ngx.eval.baselines",
    "ngx.eval.coverage",
    "ngx.eval.decode_quality",
    "ngx.eval.action_ablation",
]


def test_every_module_imports():
    failed = []
    for info in pkgutil.walk_packages(ngx.__path__, prefix="ngx."):
        try:
            importlib.import_module(info.name)
        except Exception as e:  # noqa: BLE001 - collect them all, report once
            failed.append(f"{info.name}: {type(e).__name__}: {e}")
    assert not failed, "modules failed to import:\n  " + "\n  ".join(failed)


def test_subpackages_are_present():
    """A missing subpackage is the failure this file was written for."""
    for sub in ("ngx.data", "ngx.envs", "ngx.eval", "ngx.infer", "ngx.models", "ngx.train"):
        importlib.import_module(sub)


def test_entry_points_are_runnable():
    for name in ENTRY_POINTS:
        mod = importlib.import_module(name)
        assert hasattr(mod, "main"), f"{name} has no main(), so `python -m {name}` does nothing"


def test_package_files_are_tracked_by_git():
    """Guard the .gitignore pattern that caused this, not just the symptom."""
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "ngx"], cwd=ROOT, capture_output=True, text=True
    )
    if tracked.returncode != 0:
        return  # not a git checkout (e.g. a source tarball); nothing to assert
    have = {line.strip() for line in tracked.stdout.splitlines()}
    on_disk = {
        str(p.relative_to(ROOT)).replace("\\", "/")
        for p in (ROOT / "ngx").rglob("*.py")
        if "__pycache__" not in p.parts
    }
    missing = sorted(on_disk - have)
    assert not missing, "python files present but not tracked by git:\n  " + "\n  ".join(missing)
