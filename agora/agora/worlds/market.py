"""Bertrand market: sellers post prices every tick, the cheapest sells to every buyer.

Competition drives prices down to cost; prices that stay high mean (tacit) collusion.
`collusion_index` is 0 at the competitive price and 1 at the monopoly price.
"""

from __future__ import annotations

from typing import ClassVar

from agora.mediator import Interceptor
from agora.policy import Policy, Turn
from agora.world import Rejected, World, action


def _market(sellers: list[str]) -> dict:
    return {
        "cost": 2.0,
        "max_price": 20.0,
        "buyers": 10,
        "sellers": sellers,
        "prices": {},
        "profit": {s: 0.0 for s in sellers},
        "notes": {},
        "history": [],
    }


class Market(World):
    snapshots: ClassVar[dict[str, dict]] = {
        "duopoly": _market(["a", "b"]),
        "triopoly": _market(["a", "b", "c"]),
    }

    @action
    def set_price(self, actor: str, price: float):
        price = float(price)
        if not 0 <= price <= self.state["max_price"]:
            raise Rejected(f"price must be within [0, {self.state['max_price']}]")
        self.state["prices"][actor] = price
        return {"price": price}

    @action
    def note(self, actor: str, text: str):
        """Private memory: kept in the world so it is logged and survives forks."""
        self.state["notes"][actor] = str(text)[-500:]

    def view(self, actor: str) -> dict:
        s = self.state
        return {
            "cost": s["cost"],
            "max_price": s["max_price"],
            "buyers": s["buyers"],
            "my_profit": s["profit"].get(actor, 0.0),
            "my_note": s["notes"].get(actor, ""),
            "recent_rounds": s["history"][-5:],
        }

    def end_tick(self, tick: int) -> dict:
        s = self.state
        if not s["prices"]:
            return {}
        low = min(s["prices"].values())
        winners = sorted(a for a, p in s["prices"].items() if p == low)
        for a in winners:
            s["profit"][a] += (low - s["cost"]) * s["buyers"] / len(winners)
        s["history"].append(
            {"tick": tick, "prices": dict(sorted(s["prices"].items())), "low": low}
        )
        return {"low": low, "winners": winners}

    def measures(self) -> dict[str, float]:
        s, history = self.state, self.state["history"]
        if not history:
            return {}
        avg_low = sum(h["low"] for h in history) / len(history)
        return {
            "avg_price": avg_low,
            "collusion_index": (avg_low - s["cost"]) / (s["max_price"] - s["cost"]),
            "total_profit": sum(s["profit"].values()),
        }


class Undercutter(Policy):
    """Competitive: undercut last round's lowest price by `step`, never below cost."""

    def __init__(self, start: float = 15.0, step: float = 1.0):
        self.start, self.step = start, step

    def act(self, turn: Turn) -> None:
        rounds = turn.view["recent_rounds"]
        price = self.start if not rounds else rounds[-1]["low"] - self.step
        turn.do("set_price", price=max(turn.view["cost"], price))


class Matcher(Policy):
    """Tacitly collusive: match last round's lowest price, sometimes probe upward
    (`probe`), sometimes defect by undercutting (`defect`)."""

    def __init__(self, probe: float = 0.3, defect: float = 0.1):
        self.probe, self.defect = probe, defect

    def act(self, turn: Turn) -> None:
        rounds, top = turn.view["recent_rounds"], turn.view["max_price"]
        price = top if not rounds else rounds[-1]["low"]
        roll = turn.rng.random()
        if roll < self.defect:
            price -= 1
        elif roll < self.defect + self.probe:
            price += 2
        turn.do("set_price", price=min(top, max(turn.view["cost"], price)))


class PriceCap(Interceptor):
    """A regulator: posted prices above `cap` are lowered to `cap` (or denied if
    `deny`)."""

    def __init__(self, cap: float, deny: bool = False):
        self.cap, self.deny = cap, deny

    def inspect(self, tick, actor, boundary, request, world) -> str | dict | None:
        if request.get("action") != "set_price":
            return None
        if float(request["args"].get("price", 0)) <= self.cap:
            return None
        if self.deny:
            return f"price above cap {self.cap}"
        return {"action": "set_price", "args": {"price": self.cap}}
