"""Every context length is scored on the same target frames.

Without this, a 24-frame rung and a 6-frame rung would be compared on different
windows and different revisit pairs, and a difference between them could come
from which frames were drawn rather than from the model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ngx.eval.heldout import MAX_CONTEXT  # noqa: E402
from ngx.eval.ladder import align, shift_pairs  # noqa: E402

HORIZON = 5


def _window():
    n = MAX_CONTEXT + HORIZON
    frames = np.broadcast_to(np.arange(n)[:, None, None, None], (n, 2, 2, 3)).copy()
    return frames, np.arange(n)


def test_every_context_predicts_the_same_frames():
    for C in (6, 12, MAX_CONTEXT):
        (f, a), = align([_window()], C)
        # one_step/closed_loop read context f[:C], target f[C + k], action a[C - 1 + k]
        assert f[C, 0, 0, 0] == MAX_CONTEXT
        assert a[C - 1] == MAX_CONTEXT - 1
        assert f[C + HORIZON - 1, 0, 0, 0] == MAX_CONTEXT + HORIZON - 1
        assert (f[:C, 0, 0, 0] == np.arange(MAX_CONTEXT - C, MAX_CONTEXT)).all()


def test_revisit_pairs_land_on_the_same_rollout_steps():
    pairs = [(MAX_CONTEXT + 10, MAX_CONTEXT + 90), (MAX_CONTEXT + 3, MAX_CONTEXT + 200)]
    seen = set()
    for C in (6, 12, MAX_CONTEXT):
        off = MAX_CONTEXT - C
        # drift.evaluate indexes predictions as i - C on the sliced trajectory
        seen.add(tuple((i - C, j - C) for i, j in shift_pairs(pairs, off)))
    assert len(seen) == 1


def test_context_longer_than_the_history_is_refused():
    try:
        align([_window()], MAX_CONTEXT + 1)
    except ValueError:
        return
    raise AssertionError("a context longer than MAX_CONTEXT was silently accepted")
