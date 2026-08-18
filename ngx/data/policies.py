"""Behaviour policies for data collection.

The world model can only learn dynamics that appear in the data. IID-uniform
random actions look like a good null hypothesis and are actually a trap: in a
maze they produce a jittering agent that barely translates, so the model learns
that the world is mostly static and rotating. Both policies here therefore hold
an action for a while before resampling.
"""

from __future__ import annotations

import numpy as np


class Policy:
    def reset(self) -> None:
        pass

    def act(self, pose: tuple[float, float, float] | None) -> int:
        raise NotImplementedError


class StickyRandom(Policy):
    """Uniform over actions, but each choice is held for a few steps.

    ``hold`` is the mean number of steps an action persists.
    """

    def __init__(self, num_actions: int, hold: float = 4.0, rng=None) -> None:
        self.num_actions = num_actions
        self.p_switch = 1.0 / max(hold, 1.0)
        self.rng = rng or np.random.default_rng()
        self._a = 0

    def reset(self) -> None:
        self._a = int(self.rng.integers(self.num_actions))

    def act(self, pose=None) -> int:
        if self.rng.random() < self.p_switch:
            self._a = int(self.rng.integers(self.num_actions))
        return self._a


class Explorer(Policy):
    """Forward-biased wanderer that turns when it stops making progress.

    Alternates between a FORWARD mode and a TURN mode. It leaves FORWARD early
    if the ground-truth pose says it has stopped moving, which is what happens
    when you walk into a wall. The result is trajectories that actually traverse
    the maze and revisit rooms -- the revisits are what the drift evaluation
    later measures against.

    Pose is used only to drive the behaviour policy. It is not recorded as a
    model input.
    """

    def __init__(
        self,
        action_names: tuple[str, ...],
        rng=None,
        stuck_window: int = 6,
        stuck_eps: float = 6.0,
    ) -> None:
        self.rng = rng or np.random.default_rng()
        self.stuck_window = stuck_window
        self.stuck_eps = stuck_eps

        def find(pred, fallback=0):
            for i, n in enumerate(action_names):
                if pred(n):
                    return i
            return fallback

        self.a_fwd = find(lambda n: n == "fwd")
        self.a_left = find(lambda n: n == "turn L")
        self.a_right = find(lambda n: n == "turn R")
        self.a_fwd_l = find(lambda n: n == "fwd+L", self.a_fwd)
        self.a_fwd_r = find(lambda n: n == "fwd+R", self.a_fwd)
        self.num_actions = len(action_names)
        self.reset()

    def reset(self) -> None:
        self._mode = "fwd"
        self._left = int(self.rng.integers(8, 30))
        self._turn = self.a_left
        self._hist: list[tuple[float, float]] = []

    def act(self, pose=None) -> int:
        if pose is not None:
            self._hist.append((pose[0], pose[1]))
            if len(self._hist) > self.stuck_window:
                self._hist.pop(0)

        self._left -= 1
        if self._mode == "fwd" and self._stuck():
            self._left = 0

        if self._left <= 0:
            if self._mode == "fwd":
                self._mode = "turn"
                self._turn = self.a_left if self.rng.random() < 0.5 else self.a_right
                self._left = int(self.rng.integers(3, 12))
            else:
                self._mode = "fwd"
                self._left = int(self.rng.integers(10, 40))
            self._hist.clear()

        if self._mode == "turn":
            return self._turn
        # Curve occasionally so the dataset is not all axis-aligned motion.
        r = self.rng.random()
        if r < 0.12:
            return self.a_fwd_l
        if r < 0.24:
            return self.a_fwd_r
        return self.a_fwd

    def _stuck(self) -> bool:
        if len(self._hist) < self.stuck_window:
            return False
        xs = np.array(self._hist)
        return float(np.linalg.norm(xs[-1] - xs[0])) < self.stuck_eps


def make_policy(kind: str, env, rng=None) -> Policy:
    if kind == "random":
        return StickyRandom(env.num_actions, rng=rng)
    if kind == "explore":
        return Explorer(env.action_names, rng=rng)
    raise ValueError(f"unknown policy {kind!r}")
