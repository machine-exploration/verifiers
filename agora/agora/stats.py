"""Small statistics helpers: every reported number comes with an interval."""

from __future__ import annotations

import random
import statistics


def bootstrap_ci(values: list[float], n: int = 2000) -> tuple[float, float]:
    """95% percentile-bootstrap interval of the mean (deterministic)."""
    rng = random.Random(0)
    means = sorted(
        statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n)
    )
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]
