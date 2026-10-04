"""The event log: the only source of truth for a run.

Events are hash-chained (each carries its parent's hash), so a run is identified by its
root hash and any tampering or divergence is detectable. Large payloads (model requests
and replies) live in a content-addressed blob store shared across forks: a fork copies
a list prefix, never the blobs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

GENESIS = "0" * 64


def canonical(obj: object) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Event:
    seq: int
    tick: int
    actor: str
    kind: str
    payload: dict
    parent: str
    hash: str

    @staticmethod
    def make(
        seq: int, tick: int, actor: str, kind: str, payload: dict, parent: str
    ) -> Event:
        body = {
            "seq": seq,
            "tick": tick,
            "actor": actor,
            "kind": kind,
            "payload": payload,
            "parent": parent,
        }
        return Event(**body, hash=digest(canonical(body)))


class Log:
    def __init__(
        self, events: list[Event] | None = None, blobs: dict[str, bytes] | None = None
    ):
        self.events = list(events or [])
        self.blobs = {} if blobs is None else blobs

    @property
    def root(self) -> str:
        return self.events[-1].hash if self.events else GENESIS

    def next(self, tick: int, actor: str, kind: str, payload: dict) -> Event:
        return Event.make(len(self.events), tick, actor, kind, payload, self.root)

    def append(self, event: Event) -> Event:
        if event.parent != self.root or event.seq != len(self.events):
            raise ValueError(f"event {event.seq} does not extend the log head")
        self.events.append(event)
        return event

    def put_blob(self, data: bytes) -> str:
        key = digest(data)
        self.blobs.setdefault(key, data)
        return key

    def get_blob(self, key: str) -> bytes:
        return self.blobs[key]

    def fork(self, at: int) -> Log:
        """The first `at` events, sharing this log's (immutable) blobs."""
        return Log(self.events[:at], self.blobs)

    def first_seq_at_tick(self, tick: int) -> int:
        return next((e.seq for e in self.events if e.tick >= tick), len(self.events))

    def of_kind(self, kind: str) -> list[Event]:
        return [e for e in self.events if e.kind == kind]

    def verify(self) -> None:
        parent = GENESIS
        for i, e in enumerate(self.events):
            again = Event.make(e.seq, e.tick, e.actor, e.kind, e.payload, parent)
            if e.seq != i or again.hash != e.hash:
                raise ValueError(f"log corrupted at event {i}")
            parent = e.hash

    def save(self, path: str | Path) -> None:
        path = Path(path)
        (path / "blobs").mkdir(parents=True, exist_ok=True)
        with open(path / "events.jsonl", "w") as f:
            f.writelines(
                json.dumps(asdict(e), sort_keys=True) + "\n" for e in self.events
            )
        for key, data in self.blobs.items():
            blob = path / "blobs" / key
            if not blob.exists():
                blob.write_bytes(data)

    @classmethod
    def load(cls, path: str | Path) -> Log:
        path = Path(path)
        with open(path / "events.jsonl") as f:
            events = [Event(**json.loads(line)) for line in f if line.strip()]
        blobs = {p.name: p.read_bytes() for p in (path / "blobs").glob("*")}
        log = cls(events, blobs)
        log.verify()
        return log
