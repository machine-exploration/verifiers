"""The World: a deterministic state machine that agents act on.

Same state + same action => same result, always. Randomness comes only from `rng(...)`,
which is seeded by the scenario seed and an explicit key, so replays recompute every
transition exactly. Agent memory belongs here too (not in policies), so it is logged,
inspectable and survives a policy swap in a fork.
"""

from __future__ import annotations

import copy
import random
from typing import Any, ClassVar


class Rejected(Exception):
    """The world refused an action (bad arguments, rule violation). Logged, not fatal."""


def action(fn):
    fn._agora_action = True
    return fn


class World:
    snapshots: ClassVar[dict[str, dict]] = {}
    """Named starting states; a scenario starts from one of them."""

    def __init__(self, state: dict, seed: int):
        self.state = copy.deepcopy(state)
        self.seed = seed

    @classmethod
    def from_snapshot(cls, name: str, seed: int) -> World:
        return cls(cls.snapshots[name], seed)

    @classmethod
    def action_names(cls) -> list[str]:
        return sorted(
            name
            for name in dir(cls)
            if getattr(getattr(cls, name), "_agora_action", False)
        )

    def rng(self, *key: Any) -> random.Random:
        return random.Random(":".join(map(str, (self.seed, *key))))

    def apply(self, actor: str, name: str, args: dict) -> dict:
        if name not in self.action_names():
            raise Rejected(f"unknown action {name!r}")
        result = getattr(self, name)(actor, **args)
        return {} if result is None else result

    def view(self, actor: str) -> dict:
        """What `actor` may observe. Override for partial observability."""
        return copy.deepcopy(self.state)

    def end_tick(self, tick: int) -> dict:
        """World dynamics after every agent has acted this tick."""
        return {}

    def measures(self) -> dict[str, float]:
        return {}
