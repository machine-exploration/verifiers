"""Policies turn an observation into requests. They are stateless by design: whatever an
agent remembers lives in the world (see `World`), so a policy can be swapped mid-run in a
fork without losing the agent's history. Frontier, distilled, scripted: same interface.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from agora.mediator import Denied, Mediator


@dataclass
class Turn:
    """One agent's turn: what it sees and the only two ways it can affect anything."""

    actor: str
    tick: int
    view: dict
    actions: list[str]
    rng: random.Random
    mediator: Mediator

    def do(self, name: str, **args) -> dict:
        try:
            return self.mediator.act(self.tick, self.actor, name, args)
        except Denied as denied:
            return {"denied": str(denied)}

    def ask(self, messages: list[dict]) -> str:
        return self.mediator.model(self.tick, self.actor, messages)


class Policy:
    def act(self, turn: Turn) -> None:
        raise NotImplementedError


class LLMPolicy(Policy):
    """Shows the model its view and the action list; expects one JSON action back."""

    def __init__(self, system: str = "You are an agent acting in a shared world."):
        self.system = system

    def act(self, turn: Turn) -> None:
        prompt = (
            f"Tick {turn.tick}. You are {turn.actor}.\n"
            f"What you can see:\n{json.dumps(turn.view, sort_keys=True)}\n\n"
            f"Actions: {', '.join(turn.actions)}.\n"
            'Reply with exactly one JSON object: {"action": "<name>", "args": {...}}'
        )
        try:
            reply = turn.ask(
                [
                    {"role": "system", "content": self.system},
                    {"role": "user", "content": prompt},
                ]
            )
        except Denied:
            return
        try:
            choice = json.loads(reply[reply.index("{") : reply.rindex("}") + 1])
            turn.do(choice["action"], **choice.get("args", {}))
        except (ValueError, KeyError, TypeError):
            pass  # the unparseable reply is already in the log
