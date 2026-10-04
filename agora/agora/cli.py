"""agora run | replay | fork | study | explain"""

from __future__ import annotations

import argparse
import json
import statistics

from agora.explain import attribute
from agora.log import Log
from agora.scenario import Scenario, fork, measures, replay, run
from agora.stats import bootstrap_ci


def _report(log: Log, out: str | None) -> None:
    if out:
        log.save(out)
    print(
        json.dumps(
            {"root": log.root, "events": len(log.events), "measures": measures(log)},
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="agora")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a scenario and record its log")
    r.add_argument("scenario")
    r.add_argument("--seed", type=int)
    r.add_argument("--out")

    rp = sub.add_parser("replay", help="re-execute a recorded log and verify it")
    rp.add_argument("log")

    f = sub.add_parser("fork", help="same history up to --at, then a changed scenario")
    f.add_argument("log")
    f.add_argument("--at", type=int, required=True)
    f.add_argument("--scenario", help="scenario to continue with (default: original)")
    f.add_argument("--seed", type=int)
    f.add_argument("--out")

    s = sub.add_parser("study", help="run a scenario over many seeds, report 95%% CIs")
    s.add_argument("scenario")
    s.add_argument("--seeds", type=int, default=20)

    e = sub.add_parser("explain", help="which decisions caused the outcome")
    e.add_argument("log")
    e.add_argument("--measure", required=True)
    e.add_argument("--seeds", type=int, default=8)
    e.add_argument("--top", type=int, default=10)

    args = p.parse_args(argv)
    if args.cmd == "run":
        scenario = Scenario.from_toml(args.scenario)
        if args.seed is not None:
            scenario = scenario.replace(seed=args.seed)
        _report(run(scenario), args.out)
    elif args.cmd == "replay":
        log = replay(Log.load(args.log))
        print(f"replay ok: {len(log.events)} events, root {log.root}")
    elif args.cmd == "fork":
        changes = {} if args.seed is None else {"seed": args.seed}
        new = Scenario.from_toml(args.scenario) if args.scenario else None
        _report(fork(Log.load(args.log), args.at, new, **changes), args.out)
    elif args.cmd == "study":
        scenario = Scenario.from_toml(args.scenario)
        runs = [measures(run(scenario.replace(seed=k))) for k in range(args.seeds)]
        for key in runs[0]:
            values = [m[key] for m in runs]
            lo, hi = bootstrap_ci(values)
            print(
                f"{key:>16}: {statistics.fmean(values):.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (n={len(values)})"
            )
    elif args.cmd == "explain":
        effects = attribute(Log.load(args.log), args.measure, seeds=args.seeds)
        print(f"Removing this decision changes {args.measure} by:")
        for x in effects[: args.top]:
            star = "*" if x.significant else " "
            print(
                f"{star} tick {x.tick:>3}  {x.actor:<6} {x.effect:+.3f}  "
                f"[{x.low:+.3f}, {x.high:+.3f}]  did: {x.decision}"
            )


if __name__ == "__main__":
    main()
