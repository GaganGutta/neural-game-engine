"""The held-out evaluation scores every rung on the same frames, reproducibly."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ngx.data.dataset import load_meta  # noqa: E402
from ngx.eval import heldout  # noqa: E402
from test_train_resume import _setup, _train  # noqa: E402


def test_targets_have_full_history_inside_one_validation_episode(tmp_path):
    cfg = _setup(tmp_path)
    root = str(tmp_path / "data")
    t = heldout.target_frames(root, 16)
    episodes = np.load(Path(root) / "episodes.npy")
    assert len(t) == 16
    assert (episodes[t] % 20 == 0).all(), "a target frame is not in a validation episode"
    assert (episodes[t - heldout.MAX_CONTEXT] == episodes[t]).all(), "history crosses an episode"
    assert load_meta(root)["num_actions"] == 3 and cfg.exists()


def test_scores_are_finite_and_reproducible(tmp_path, monkeypatch):
    cfg = _setup(tmp_path)
    _train(monkeypatch, cfg)
    outs = []
    for k in range(2):
        out = tmp_path / f"h{k}.json"
        monkeypatch.setattr(sys, "argv", ["heldout", "--config", str(cfg), "--device", "cpu",
                                          "--targets", "40", "--out", str(out)])
        heldout.main()
        outs.append(json.loads(out.read_text()))
    a, b = outs
    for key in ("loss", "last_loss", "cold_loss", "cold_acc"):
        assert np.isfinite(a[key]) and a[key] == b[key], key
    assert 0.0 <= a["cold_acc"] <= 1.0
    assert a["step"] == 8 and a["ckpt"].endswith("final.pt")
