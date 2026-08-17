"""The contract every world shares.

Deliberately not ``gym.Env``. We need exactly two things out of a world -- a
uint8 RGB frame and whether the episode ended -- and a discrete action index
in. Gym's step tuple has grown three times in as many years; this hasn't.
"""

from __future__ import annotations

import numpy as np


class WorldEnv:
    """Discrete actions in, uint8 RGB frames out."""

    #: number of discrete actions the world accepts
    num_actions: int = 0
    #: human-readable name per action index, used by ``play.py`` for the HUD
    action_names: tuple[str, ...] = ()
    #: side length of the square frames this world emits
    frame_size: int = 64

    def reset(self) -> np.ndarray:
        """Start a new episode. Returns the first frame, ``(S, S, 3)`` uint8."""
        raise NotImplementedError

    def step(self, action: int) -> tuple[np.ndarray, bool]:
        """Apply ``action``. Returns ``(frame, done)``."""
        raise NotImplementedError

    def pose(self) -> tuple[float, float, float] | None:
        """Ground-truth ``(x, y, angle_degrees)``, or None if the world has no
        notion of position.

        Only ever used by the *evaluation* code -- to decide when a trajectory
        has genuinely returned to a place it visited before -- and by scripted
        data-collection policies. The world model never sees it.
        """
        return None

    def close(self) -> None:
        pass

    def __enter__(self) -> "WorldEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
