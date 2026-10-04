"""End-to-end acceptance tests for the kernel: replay, forks, interceptors, models."""

from pathlib import Path

import pytest
from agora.log import Event
from agora.models import FakeModel

from agora import Divergence, Log, Scenario, fork, measures, replay, run

SCENARIOS = Path(__file__).parent.parent / "scenarios"


def load(name: str) -> Scenario:
    return Scenario.from_toml(SCENARIOS / f"{name}.toml")


def rechain(events: list[Event]) -> list[Event]:
    """Re-hash a (tampered) event list so only replay, not `verify`, can catch it."""
    out, parent = [], "0" * 64
    for e in events:
        out.append(Event.make(e.seq, e.tick, e.actor, e.kind, e.payload, parent))
        parent = out[-1].hash
    return out


def test_replay_is_bit_exact(tmp_path):
    log = run(load("market_tacit"))
    log.save(tmp_path)
    assert replay(Log.load(tmp_path)).root == log.root


def test_replay_catches_a_forged_world_transition():
    log = run(load("market_tacit"))
    events = list(log.events)
    i = next(e.seq for e in events if e.kind == "tick" and e.payload)
    forged = events[i]
    events[i] = Event.make(
        forged.seq, forged.tick, forged.actor, forged.kind, {"low": 99.0}, forged.parent
    )
    with pytest.raises(Divergence):
        replay(Log(rechain(events), log.blobs))


def test_fork_shares_history_and_blobs():
    log = run(load("market_tacit"))
    same = fork(log, 10)
    k = log.first_seq_at_tick(10)
    assert [e.hash for e in same.events[:k]] == [e.hash for e in log.events[:k]]
    assert same.blobs is log.blobs
    assert measures(same) == measures(log)  # unchanged scenario => same future
    assert measures(fork(log, 10, seed=7)) != measures(log)  # resampled future


def test_interceptor_changes_outcome_without_touching_world_or_policies():
    log = run(load("market_tacit"))
    capped = fork(log, 10, load("market_capped"))
    assert capped.of_kind("action.modified")
    assert measures(capped)["collusion_index"] < measures(log)["collusion_index"]
    assert replay(capped).root == capped.root


def test_model_replies_are_replayed_not_resampled():
    log = run(load("market_llm"))
    assert log.of_kind("model.result")
    before = FakeModel.calls
    replay(log)
    assert FakeModel.calls == before


def test_attribution_ablates_exactly_one_decision():
    from agora.explain import attribute

    log = run(load("market_tacit"))
    [effect] = attribute(log, "collusion_index", seeds=4, ticks=[6], actors=["a"])
    assert effect.low <= effect.effect <= effect.high
    assert effect.decision == next(
        e.payload
        for e in log.of_kind("action.request")
        if (e.tick, e.actor) == (6, "a")
    )
    ablated = fork(
        log,
        6,
        load("market_tacit").replace(
            interceptors=({"use": "agora.explain:Ablate", "actor": "a", "tick": 6},)
        ),
    )
    denied = ablated.of_kind("action.denied")
    assert [(e.tick, e.actor) for e in denied] == [(6, "a")]
