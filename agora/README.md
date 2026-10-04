# agora

Reproducible, forkable multi-agent worlds. Every model call and every action passes one
mediator and lands in one hash-chained event log, so any run can be replayed bit for
bit, forked at any tick, and compared under different oversight mechanisms.

No dependencies beyond the Python standard library.

## Quickstart

```bash
uv run agora run scenarios/market_tacit.toml --out runs/tacit      # record a run
uv run agora replay runs/tacit                                     # re-execute, verify every event
uv run agora fork runs/tacit --at 10 --scenario scenarios/market_capped.toml   # same history, then a regulator
uv run agora fork runs/tacit --at 10 --seed 7                      # same history, resampled future
uv run agora study scenarios/market_tacit.toml --seeds 30          # means with 95% bootstrap CIs
uv run --group dev pytest                                          # acceptance tests
```

## Concepts

| Concept | What it is | File |
| --- | --- | --- |
| Log | Hash-chained events + content-addressed blobs; `fork(at)` shares blobs | `agora/log.py` |
| World | Deterministic state machine: `@action` methods, `view`, `end_tick`, `measures`, named `snapshots` | `agora/world.py` |
| Mediator | Capability check, interceptors (allow / deny / modify), execute or replay, append | `agora/mediator.py` |
| Policy | Stateless; acts through `turn.do(...)` and `turn.ask(...)`. Memory lives in the world | `agora/policy.py` |
| Scenario | World + snapshot + agents + interceptors + horizon + seed, identified by its hash | `agora/scenario.py` |

A world is a few dozen lines. See `agora/worlds/market.py`: a Bertrand market whose
`collusion_index` is 0 at the competitive price and 1 at the monopoly price.

## LLM agents

`agora.policy:LLMPolicy` asks a model for one JSON action per turn. `scenarios/market_llm.toml`
runs offline with `FakeModel`; point its `[model]` table at any OpenAI-compatible endpoint
(`agora.models:OpenAICompatible`). Model replies are stored as blobs and read back on
replay, so replays never call the model.
