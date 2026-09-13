"""Resuming a training run from its newest step checkpoint.

The larger ladder rungs are many hours on a rented GPU. A run that cannot pick
up where it stopped turns any interruption into paying for the whole run again,
so resume is tested end to end on a tiny synthetic dataset.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ngx.models.vqvae import VQVAE  # noqa: E402
from ngx.train import train_dynamics  # noqa: E402
from ngx.train.common import cosine_warmup, load_ckpt  # noqa: E402

STEPS, BATCH, CONTEXT, WARMUP, LR = 8, 4, 2, 2, 1e-3


def _setup(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    rng = np.random.default_rng(0)
    n = 400
    np.save(root / "tokens.npy", rng.integers(0, 16, size=(n, 64)).astype(np.uint16))
    np.save(root / "actions.npy", rng.integers(0, 3, size=n).astype(np.uint8))
    # episode 20 is held out (every 20th), 21-23 train
    np.save(root / "episodes.npy", np.repeat(np.arange(20, 24), n // 4).astype(np.int32))
    (root / "meta.json").write_text(json.dumps({"num_actions": 3, "action_names": ["-", "a", "b"]}))

    tc = {"ch": 8, "embed_dim": 8, "num_codes": 16, "n_res": 1}
    vq = VQVAE(**tc)
    torch.save({"model": vq.state_dict(), "cfg": {"tokenizer": tc}}, tmp_path / "vq.pt")

    cfg = {
        "name": "resume", "seed": 0, "amp": False, "num_workers": 0,
        "log_every": 1, "eval_every": 3, "run_root": str(tmp_path / "runs"),
        "data": {"root": str(root), "preload": True},
        "tokenizer_ckpt": str(tmp_path / "vq.pt"),
        "dynamics": {
            "context": CONTEXT, "d_model": 32, "n_layers": 1, "n_heads": 2,
            "dropout": 0.0, "pos_encoding": "rope", "batch_size": BATCH, "lr": LR,
            "warmup": WARMUP, "steps": STEPS, "checkpoint_every": 3, "keep_checkpoints": 5,
        },
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _train(monkeypatch, cfg_path: Path, *extra: str) -> None:
    monkeypatch.setattr(sys, "argv", ["train_dynamics", "--config", str(cfg_path),
                                      "--device", "cpu", *extra])
    train_dynamics.main()


def test_step_checkpoints_carry_what_resume_needs(tmp_path, monkeypatch):
    cfg = _setup(tmp_path)
    _train(monkeypatch, cfg)
    out = tmp_path / "runs" / "resume" / "dynamics"
    ck = load_ckpt(str(out / "ckpt_0000006.pt"))
    assert ck["step"] == 6
    # Written after the step-3 eval, so it already knows that eval's loss.
    assert np.isfinite(load_ckpt(str(out / "ckpt_0000003.pt"))["best_val"])
    assert ck["optimizer"]["state"], "optimizer moments were not saved"
    assert np.isfinite(ck["best_val"])
    assert "evals_since_best" in ck
    assert not list(out.glob("*.tmp")), "an atomic save left its temp file behind"


def test_resume_continues_the_planned_schedule(tmp_path, monkeypatch, capsys):
    cfg = _setup(tmp_path)
    _train(monkeypatch, cfg)
    out = tmp_path / "runs" / "resume" / "dynamics"
    # Simulate an interruption right after the step-3 checkpoint.
    for f in ("ckpt_0000006.pt", "dynamics.pt", "final.pt"):
        (out / f).unlink()
    capsys.readouterr()

    _train(monkeypatch, cfg, "--resume")
    log = capsys.readouterr().out
    assert "resumed from" in log and "at step 4/8" in log

    # The first resumed step sits on the original cosine, not a fresh warmup.
    lr4 = float(re.search(r"step\s+4/8 .* lr ([0-9.e+-]+)", log).group(1))
    assert abs(lr4 - cosine_warmup(4, STEPS, WARMUP, LR, LR * 0.05)) < 1e-9 + 1e-3 * lr4
    assert not re.search(r"step\s+[0-3]/8 ", log), "resume replayed steps it already had"

    final = load_ckpt(str(out / "final.pt"))
    assert final["step"] == STEPS
    assert final["tokens_seen"] == STEPS * BATCH * CONTEXT * 65
    assert (out / "ckpt_0000006.pt").exists()


def test_resume_without_a_checkpoint_starts_fresh(tmp_path, monkeypatch, capsys):
    cfg = _setup(tmp_path)
    _train(monkeypatch, cfg, "--resume")
    log = capsys.readouterr().out
    assert "resumed from" not in log
    assert re.search(r"step\s+0/8 ", log)
    assert load_ckpt(str(tmp_path / "runs" / "resume" / "dynamics" / "final.pt"))["step"] == STEPS


class _Stop(Exception):
    pass


def test_resume_restores_weights_optimizer_and_plateau_state(tmp_path, monkeypatch, capsys):
    """Stop the resumed run before its first update and compare with the file."""
    cfg = _setup(tmp_path)
    _train(monkeypatch, cfg)
    out = tmp_path / "runs" / "resume" / "dynamics"
    for f in ("ckpt_0000006.pt", "dynamics.pt", "final.pt"):
        (out / f).unlink()
    ck = load_ckpt(str(out / "ckpt_0000003.pt"))

    models, opts = [], []

    class RecordingModel(train_dynamics.DynamicsTransformer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            models.append(self)

    class RecordingAdamW(torch.optim.AdamW):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            opts.append(self)

    def stop_before_first_batch(_loader):
        raise _Stop
        yield  # pragma: no cover - makes this a generator

    monkeypatch.setattr(train_dynamics, "DynamicsTransformer", RecordingModel)
    monkeypatch.setattr(torch.optim, "AdamW", RecordingAdamW)
    monkeypatch.setattr(train_dynamics, "infinite", stop_before_first_batch)
    capsys.readouterr()
    with pytest.raises(_Stop):
        _train(monkeypatch, cfg, "--resume")
    assert "best val inf" not in capsys.readouterr().out

    sd = models[-1].state_dict()
    for k, v in ck["model"].items():
        assert torch.equal(sd[k].cpu(), v), f"weight {k} not restored"
    st = opts[-1].state_dict()["state"]
    assert set(st) == set(ck["optimizer"]["state"])
    for i, s in st.items():
        assert torch.equal(s["exp_avg"].cpu(), ck["optimizer"]["state"][i]["exp_avg"]), i
        assert s["step"].device.type == "cpu"
