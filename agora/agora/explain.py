"""Causal attribution for collective outcomes.

Which agent's decision, at which tick, moved the outcome? For every (tick, actor) we
fork the recorded run at that tick twice per seed: once unchanged (control) and once
with that one decision ablated (the agent sits the turn out). Both forks share the
exact history before the tick and the same resampled future after it, so the paired
difference in the outcome measure is the causal effect of that decision, with a
bootstrap interval over seeds. Activation patching, applied to a society.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from agora.log import Log
from agora.mediator import Interceptor
from agora.scenario import fork, measures, scenarios_of
from agora.stats import bootstrap_ci


class Ablate(Interceptor):
    """Blocks every action `actor` attempts at `tick`."""

    def __init__(self, actor: str, tick: int):
        self.actor, self.tick = actor, tick

    def inspect(self, tick, actor, boundary, request, world) -> str | None:
        if boundary == "action" and tick == self.tick and actor == self.actor:
            return "ablated"
        return None


@dataclass(frozen=True)
class Effect:
    tick: int
    actor: str
    decision: dict | None
    """What the agent actually did at that tick in the recorded run."""
    effect: float
    """Mean change in the measure when this decision is removed."""
    low: float
    high: float

    @property
    def significant(self) -> bool:
        return self.low > 0 or self.high < 0


def attribute(
    log: Log,
    measure: str,
    seeds: int = 8,
    ticks: list[int] | None = None,
    actors: list[str] | None = None,
) -> list[Effect]:
    """Effects of removing each (tick, actor) decision, largest first."""
    start, forked, _ = scenarios_of(log)
    if forked is not None:
        raise ValueError("attribute effects on an original run, not a fork")
    ticks = list(range(start.horizon)) if ticks is None else ticks
    actors = [a.name for a in start.agents] if actors is None else actors
    decisions = {
        (e.tick, e.actor): e.payload for e in reversed(log.of_kind("action.request"))
    }
    effects = []
    for t in ticks:
        control = {s: measures(fork(log, t, seed=s))[measure] for s in range(seeds)}
        for actor in actors:
            ablate = {"use": "agora.explain:Ablate", "actor": actor, "tick": t}
            diffs = [
                measures(
                    fork(
                        log,
                        t,
                        start.replace(
                            seed=s, interceptors=(*start.interceptors, ablate)
                        ),
                    )
                )[measure]
                - control[s]
                for s in range(seeds)
            ]
            low, high = bootstrap_ci(diffs)
            effects.append(
                Effect(
                    t,
                    actor,
                    decisions.get((t, actor)),
                    statistics.fmean(diffs),
                    low,
                    high,
                )
            )
    return sorted(effects, key=lambda e: -abs(e.effect))
