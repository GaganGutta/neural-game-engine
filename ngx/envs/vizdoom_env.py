"""VizDoom worlds.

``my_way_home`` is the default and the one the drift work is built around: it
is a maze of rooms with visually distinct wall textures, so "did walking back
into a room give you that room" is a question with a ground-truth answer.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import vizdoom as vzd

from .base import WorldEnv

# Each entry maps a discrete action index to a set of simultaneously-held
# buttons. Buttons are named rather than positional because the cfg file owns
# the ordering and it differs per scenario.
ACTION_SETS: dict[str, tuple[tuple[str, ...], ...]] = {
    "my_way_home": (
        (),
        ("TURN_LEFT",),
        ("TURN_RIGHT",),
        ("MOVE_FORWARD",),
        ("MOVE_FORWARD", "TURN_LEFT"),
        ("MOVE_FORWARD", "TURN_RIGHT"),
    ),
    "health_gathering": (
        (),
        ("TURN_LEFT",),
        ("TURN_RIGHT",),
        ("MOVE_FORWARD",),
        ("MOVE_FORWARD", "TURN_LEFT"),
        ("MOVE_FORWARD", "TURN_RIGHT"),
    ),
    "deadly_corridor": (
        (),
        ("TURN_LEFT",),
        ("TURN_RIGHT",),
        ("MOVE_FORWARD",),
        ("MOVE_LEFT",),
        ("MOVE_RIGHT",),
        ("ATTACK",),
    ),
    "defend_the_center": ((), ("TURN_LEFT",), ("TURN_RIGHT",), ("ATTACK",)),
    "basic": ((), ("MOVE_LEFT",), ("MOVE_RIGHT",), ("ATTACK",)),
}

# Short labels for the play.py HUD.
_PRETTY = {
    (): "-",
    ("TURN_LEFT",): "turn L",
    ("TURN_RIGHT",): "turn R",
    ("MOVE_FORWARD",): "fwd",
    ("MOVE_FORWARD", "TURN_LEFT"): "fwd+L",
    ("MOVE_FORWARD", "TURN_RIGHT"): "fwd+R",
    ("MOVE_LEFT",): "strafe L",
    ("MOVE_RIGHT",): "strafe R",
    ("ATTACK",): "fire",
}


class VizDoomEnv(WorldEnv):
    def __init__(
        self,
        scenario: str = "my_way_home",
        frame_size: int = 64,
        frame_skip: int = 4,
        clean_render: bool = True,
        seed: int | None = None,
        episode_timeout: int | None = None,
    ) -> None:
        if scenario not in ACTION_SETS:
            raise ValueError(
                f"unknown scenario {scenario!r}; known: {sorted(ACTION_SETS)}"
            )
        cfg = os.path.join(vzd.scenarios_path, f"{scenario}.cfg")
        if not os.path.exists(cfg):
            raise FileNotFoundError(f"vizdoom scenario cfg not found: {cfg}")

        self.scenario = scenario
        self.frame_size = frame_size
        self.frame_skip = frame_skip

        game = vzd.DoomGame()
        game.load_config(cfg)
        # Render small: we downscale to 64x64 anyway, and the renderer is the
        # bottleneck during collection.
        game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_window_visible(False)
        game.set_mode(vzd.Mode.PLAYER)
        if clean_render:
            # Fewer moving parts the tokenizer has to spend codebook entries on.
            game.set_render_hud(False)
            game.set_render_crosshair(False)
            game.set_render_decals(False)
            game.set_render_particles(False)
        # Pose is for evaluation and scripted policies only, never for the model.
        for var in (
            vzd.GameVariable.POSITION_X,
            vzd.GameVariable.POSITION_Y,
            vzd.GameVariable.ANGLE,
        ):
            game.add_available_game_variable(var)
        if episode_timeout is not None:
            # The stock scenarios cut episodes at 2100 tics, which at frame_skip
            # 4 is ~525 steps -- too short to measure drift over 1000 frames.
            game.set_episode_timeout(episode_timeout)
        if seed is not None:
            game.set_seed(seed)
        game.init()
        self.game = game

        specs = ACTION_SETS[scenario]
        buttons = [b.name for b in game.get_available_buttons()]
        missing = {b for spec in specs for b in spec} - set(buttons)
        if missing:
            raise RuntimeError(
                f"{scenario} action set needs buttons {sorted(missing)} "
                f"which the cfg does not expose (has {buttons})"
            )
        # Precompute one button vector per discrete action.
        self._action_vecs = [
            [1.0 if b in spec else 0.0 for b in buttons] for spec in specs
        ]
        self.num_actions = len(specs)
        self.action_names = tuple(_PRETTY.get(s, "+".join(s) or "-") for s in specs)
        self._last_frame = np.zeros((frame_size, frame_size, 3), np.uint8)

    # -- WorldEnv -----------------------------------------------------------
    def reset(self) -> np.ndarray:
        self.game.new_episode()
        return self._grab()

    def step(self, action: int) -> tuple[np.ndarray, bool]:
        self.game.make_action(self._action_vecs[action], self.frame_skip)
        done = self.game.is_episode_finished()
        # On the terminal tick VizDoom drops the state, so reuse the last frame
        # rather than emitting a black one the model would have to learn.
        return (self._last_frame if done else self._grab()), done

    def pose(self) -> tuple[float, float, float] | None:
        if self.game.is_episode_finished():
            return None
        state = self.game.get_state()
        if state is None:
            return None
        gv = state.game_variables
        if len(gv) < 3:
            return None
        return float(gv[-3]), float(gv[-2]), float(gv[-1])

    def close(self) -> None:
        try:
            self.game.close()
        except Exception:
            pass

    # -- internals ----------------------------------------------------------
    def _grab(self) -> np.ndarray:
        state = self.game.get_state()
        if state is None:
            return self._last_frame
        buf = state.screen_buffer  # (H, W, 3) uint8
        # INTER_AREA is the right downsampler here: it box-filters, so thin
        # wall textures alias far less than they do under bilinear.
        self._last_frame = cv2.resize(
            buf, (self.frame_size, self.frame_size), interpolation=cv2.INTER_AREA
        )
        return self._last_frame
