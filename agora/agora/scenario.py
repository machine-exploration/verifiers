"""Scenarios and the run loop.

A Scenario is the reproducible unit: world + starting snapshot + population +
interceptors + horizon + seed, identified by a hash. `run` records a log; `replay`
re-executes a log and checks it event by event; `fork` replays a log up to a tick, then
continues under a different scenario (another policy, interceptor or seed).
"""

from __future__ import annotations

import dataclasses
import importlib
import random
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from agora.log import Log, canonical, digest
from agora.mediator import Capability, Interceptor, Mediator
from agora.policy import Policy, Turn
from agora.world import World

SETUP_TICK = -1


def load_object(path: str):
    module, _, name = path.partition(":")
    return getattr(importlib.import_module(module), name)


def _build(spec: dict):
    """`{"use": "module:Class", **kwargs}` -> instance."""
    kwargs = {k: v for k, v in spec.items() if k != "use"}
    return load_object(spec["use"])(**kwargs)


@dataclass(frozen=True)
class AgentSpec:
    name: str
    policy: str
    params: dict = field(default_factory=dict)
    scopes: tuple[str, ...] = ("*",)
    model: dict | None = None


@dataclass(frozen=True)
class Scenario:
    world: str
    snapshot: str
    agents: tuple[AgentSpec, ...]
    horizon: int
    seed: int = 0
    interceptors: tuple[dict, ...] = ()
    model: dict | None = None
    """Default model for every agent; an agent's own `model` overrides it."""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @property
    def hash(self) -> str:
        return digest(canonical(self.to_dict()))

    def replace(self, **changes) -> Scenario:
        return dataclasses.replace(self, **changes)

    @classmethod
    def from_dict(cls, d: dict) -> Scenario:
        agents = tuple(
            AgentSpec(**{**a, "scopes": tuple(a.get("scopes", ("*",)))})
            for a in d["agents"]
        )
        return cls(
            **{
                **d,
                "agents": agents,
                "interceptors": tuple(d.get("interceptors", ())),
            }
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> Scenario:
        with open(path, "rb") as f:
            return cls.from_dict(tomllib.load(f))


@dataclass
class Population:
    policies: dict[str, Policy]
    caps: dict[str, Capability]
    interceptors: list[Interceptor]
    models: dict


def populate(scenario: Scenario) -> Population:
    models = {}
    for agent in scenario.agents:
        spec = agent.model or scenario.model
        if spec is not None:
            models[agent.name] = _build(spec)
    return Population(
        policies={a.name: load_object(a.policy)(**a.params) for a in scenario.agents},
        caps={a.name: Capability(a.name, frozenset(a.scopes)) for a in scenario.agents},
        interceptors=[_build(spec) for spec in scenario.interceptors],
        models=models,
    )


def execute(
    scenario: Scenario,
    *,
    prefix: Scenario | None = None,
    switch_tick: int | None = None,
    recorded: Log | None = None,
    replay_until: int = 0,
) -> Log:
    """Run `prefix` (or `scenario`) until `switch_tick`, then `scenario`. Events before
    `replay_until` must match `recorded` exactly; model replies there are read back."""
    first = prefix or scenario
    world: World = load_object(first.world).from_snapshot(first.snapshot, first.seed)
    log = Log(blobs=recorded.blobs if recorded is not None else None)
    pop = populate(first)
    m = Mediator(
        world, log, pop.caps, pop.interceptors, pop.models, recorded, replay_until
    )
    m.emit(
        SETUP_TICK,
        "system",
        "scenario",
        {"hash": first.hash, "scenario": first.to_dict()},
    )
    active = first
    for tick in range(scenario.horizon):
        if prefix is not None and tick == switch_tick:
            active, pop = scenario, populate(scenario)
            m.caps, m.interceptors, m.models = pop.caps, pop.interceptors, pop.models
            m.emit(
                tick,
                "system",
                "fork",
                {"hash": scenario.hash, "scenario": scenario.to_dict()},
            )
        order = sorted(pop.policies)
        random.Random(f"{active.seed}:order:{tick}").shuffle(order)
        for actor in order:
            turn = Turn(
                actor=actor,
                tick=tick,
                view=world.view(actor),
                actions=world.action_names(),
                rng=random.Random(f"{active.seed}:policy:{actor}:{tick}"),
                mediator=m,
            )
            pop.policies[actor].act(turn)
        m.end_tick(tick)
    m.emit(scenario.horizon, "system", "measures", world.measures())
    return log


def scenarios_of(log: Log) -> tuple[Scenario, Scenario | None, int | None]:
    """(starting scenario, forked-to scenario or None, fork tick or None)."""
    start = Scenario.from_dict(log.events[0].payload["scenario"])
    forks = log.of_kind("fork")
    if not forks:
        return start, None, None
    return start, Scenario.from_dict(forks[0].payload["scenario"]), forks[0].tick


def run(scenario: Scenario) -> Log:
    return execute(scenario)


def replay(log: Log) -> Log:
    """Re-execute a recorded run; raises Divergence at the first mismatching event."""
    start, forked, tick = scenarios_of(log)
    again = execute(
        forked or start,
        prefix=start if forked else None,
        switch_tick=tick,
        recorded=log,
        replay_until=len(log.events),
    )
    assert again.root == log.root
    return again


def fork(log: Log, at_tick: int, scenario: Scenario | None = None, **changes) -> Log:
    """Same history up to `at_tick`, then continue under `scenario` (default: the
    original scenario with `changes`, e.g. `seed=7`)."""
    start, forked, _ = scenarios_of(log)
    if forked is not None:
        raise ValueError("fork the original run, not a fork")
    new = (scenario or start).replace(**changes)
    return execute(
        new,
        prefix=start,
        switch_tick=at_tick,
        recorded=log,
        replay_until=log.first_seq_at_tick(at_tick),
    )


def measures(log: Log) -> dict[str, float]:
    return log.of_kind("measures")[-1].payload
