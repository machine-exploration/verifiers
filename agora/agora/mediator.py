"""The mediator: the single point every model call and every action passes through.

Each request is checked against the actor's capabilities, shown to every interceptor
(oversight mechanisms and protocols), then executed or replayed, and the outcome is
appended to the log. While replaying, every event is compared with the recorded one,
so any divergence is caught at the exact step it happens; model replies are read back
from the recording instead of calling the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agora.log import Event, Log, canonical
from agora.world import Rejected, World


class Denied(Exception):
    pass


class Divergence(Exception):
    pass


@dataclass(frozen=True)
class Capability:
    actor: str
    scopes: frozenset[str]

    def allows(self, scope: str) -> bool:
        family = scope.split(":", 1)[0]
        return bool({"*", scope, f"{family}:*"} & self.scopes)


class Interceptor:
    """Sees every request before it runs. Return None to allow it, a reason (str) to
    deny it, or a replacement request (dict) to modify it."""

    def inspect(
        self, actor: str, boundary: str, request: dict, world: World
    ) -> str | dict | None:
        return None


class ModelClient(Protocol):
    def complete(self, messages: list[dict]) -> str: ...


class Mediator:
    def __init__(
        self,
        world: World,
        log: Log,
        caps: dict[str, Capability],
        interceptors: list[Interceptor],
        models: dict[str, ModelClient],
        recorded: Log | None = None,
        replay_until: int = 0,
    ):
        self.world = world
        self.log = log
        self.caps = caps
        self.interceptors = interceptors
        self.models = models
        self.recorded = recorded
        self.replay_until = replay_until if recorded is not None else 0

    def replaying(self) -> bool:
        return len(self.log.events) < self.replay_until

    def emit(self, tick: int, actor: str, kind: str, payload: dict) -> Event:
        event = self.log.next(tick, actor, kind, payload)
        if self.replaying():
            expected = self.recorded.events[event.seq]
            if expected.hash != event.hash:
                raise Divergence(
                    f"seq {event.seq}: recorded {expected.kind} {expected.payload}, "
                    f"got {kind} {payload}"
                )
        return self.log.append(event)

    def _gate(
        self, tick: int, actor: str, boundary: str, scope: str, request: dict
    ) -> dict:
        """Capability check, then every interceptor in order; returns the request to run."""
        if not self.caps[actor].allows(scope):
            self._deny(tick, actor, boundary, request, f"no capability for {scope}")
        for interceptor in self.interceptors:
            verdict = interceptor.inspect(actor, boundary, request, self.world)
            if isinstance(verdict, str):
                self._deny(tick, actor, boundary, request, verdict)
            if isinstance(verdict, dict):
                self.emit(
                    tick,
                    actor,
                    f"{boundary}.modified",
                    {"from": request, "to": verdict},
                )
                request = verdict
        return request

    def _deny(self, tick, actor, boundary, request, reason) -> None:
        self.emit(
            tick, actor, f"{boundary}.denied", {"request": request, "reason": reason}
        )
        raise Denied(reason)

    def act(self, tick: int, actor: str, name: str, args: dict) -> dict:
        request = self._gate(
            tick, actor, "action", f"action:{name}", {"action": name, "args": args}
        )
        self.emit(tick, actor, "action.request", request)
        try:
            result = self.world.apply(actor, request["action"], request["args"])
        except (Rejected, TypeError, ValueError) as error:
            result = {"rejected": str(error)}
        self.emit(tick, actor, "action.result", result)
        return result

    def model(self, tick: int, actor: str, messages: list[dict]) -> str:
        request = self._gate(tick, actor, "model", "model", {"messages": messages})
        self.emit(
            tick,
            actor,
            "model.request",
            {"blob": self.log.put_blob(canonical(request))},
        )
        if self.replaying():
            recorded = self.recorded.events[len(self.log.events)]
            if recorded.kind != "model.result":
                raise Divergence(f"seq {recorded.seq}: expected a model reply")
            text = self.recorded.get_blob(recorded.payload["blob"]).decode()
        else:
            if actor not in self.models:
                raise RuntimeError(f"no model configured for {actor!r}")
            text = self.models[actor].complete(request["messages"])
        self.emit(
            tick, actor, "model.result", {"blob": self.log.put_blob(text.encode())}
        )
        return text

    def end_tick(self, tick: int) -> None:
        self.emit(tick, "world", "tick", self.world.end_tick(tick))
