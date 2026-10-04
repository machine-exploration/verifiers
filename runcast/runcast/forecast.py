"""Forecast how a run ends from how it starts, and measure how forecast error scales
with the number of runs in the training corpus.

Target: log10 final test loss (step 400). Inputs: the run's config and its first 10%
(steps 0-40). Methods:
  last value      - assume the loss stops improving (no corpus)
  power law       - fit log L = a + b log t on the early curve, extrapolate (no corpus)
  config only     - learned from the corpus, config without the early curve
  learned         - learned from the corpus, config + early curve
  hybrid          - learned + the power-law extrapolation as an extra feature
Two test sets: in-distribution (width <= 64, like the corpus) and extrapolation to
wider networks (width >= 128), never seen in the corpus.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

CUT = 4  # checkpoint index of step 40 = 10% of the run
SIZES = [30, 100, 300, 1000, 3000]
REPEATS = 5


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def power_law(run: dict) -> float:
    t = np.array([10, 20, 30, 40])
    y = np.log10(run["curve"][1 : CUT + 1])
    slope, intercept = np.polyfit(np.log10(t), y, 1)
    return float(intercept + slope * np.log10(400))


def features(run: dict, kind: str) -> list[float]:
    cfg = [
        np.log2(run["width"]),
        np.log10(run["lr"]),
        np.log2(run["n_train"]),
        np.log2(run["dim"]),
        run["noise"],
    ]
    if kind == "config":
        return cfg
    early = list(np.log10(run["curve"][: CUT + 1])) + list(
        np.log10(np.array(run["grad_norms"][:CUT]) + 1e-12)
    )
    out = cfg + early
    if kind == "hybrid":
        out.append(power_law(run))
    return out


def evaluate(runs: list[dict]) -> dict:
    rng = np.random.default_rng(1)
    target = np.array([np.log10(r["final"]) for r in runs])
    width = np.array([r["width"] for r in runs])
    narrow = np.flatnonzero(width <= 64)
    wide = np.flatnonzero(width >= 128)
    narrow = rng.permutation(narrow)
    test_sets = {"in-distribution": narrow[:1000], "wider networks": wide[:1000]}
    pool = narrow[1000:]

    sizes = [n for n in SIZES if n <= len(pool)]
    results: dict = {"sizes": sizes, "n_pool": len(pool), "tests": {}}
    for name, test in test_sets.items():
        y = target[test]
        results["tests"][name] = {
            "last value": float(
                np.mean(np.abs(np.log10([runs[i]["curve"][CUT] for i in test]) - y))
            ),
            "power law": float(
                np.mean(np.abs(np.array([power_law(runs[i]) for i in test]) - y))
            ),
        }
    for kind, label in [
        ("config", "config only"),
        ("early", "learned"),
        ("hybrid", "hybrid"),
    ]:
        x_all = np.array([features(r, kind) for r in runs])
        errs = {name: [] for name in test_sets}
        for n in sizes:
            per = {name: [] for name in test_sets}
            for rep in range(REPEATS):
                train = np.random.default_rng(100 * n + rep).choice(
                    pool, size=n, replace=False
                )
                model = HistGradientBoostingRegressor(
                    max_iter=300, learning_rate=0.05, min_samples_leaf=5
                )
                model.fit(x_all[train], target[train])
                for name, test in test_sets.items():
                    err = np.abs(model.predict(x_all[test]) - target[test])
                    per[name].append(float(np.mean(err)))
            for name in test_sets:
                errs[name].append(per[name])
        for name in test_sets:
            results["tests"][name][label] = errs[name]
    return results


if __name__ == "__main__":
    out = evaluate(load(sys.argv[1]))
    with open(sys.argv[2], "w") as f:
        json.dump(out, f, indent=2)
    for test, res in out["tests"].items():
        print(f"\n{test}: mean absolute error in log10 final loss (dex)")
        for method, value in res.items():
            if isinstance(value, float):
                print(f"  {method:<12} {value:.3f}  (no corpus)")
            else:
                means = "  ".join(
                    f"N={n}:{np.mean(v):.3f}" for n, v in zip(out["sizes"], value)
                )
                print(f"  {method:<12} {means}")
