"""World registry."""

from __future__ import annotations

from .base import WorldEnv


def make_env(name: str = "my_way_home", **kw) -> WorldEnv:
    """Build a world by name.

    Names are VizDoom scenarios (``my_way_home``, ``basic``,
    ``defend_the_center``, ``health_gathering``, ``deadly_corridor``).
    """
    from .vizdoom_env import ACTION_SETS, VizDoomEnv

    if name in ACTION_SETS:
        return VizDoomEnv(scenario=name, **kw)
    raise ValueError(f"unknown world {name!r}; known: {sorted(ACTION_SETS)}")


__all__ = ["WorldEnv", "make_env"]
