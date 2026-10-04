"""Forecast error vs corpus size, log-log, one panel per test set."""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"learned": "#2a78d6", "hybrid": "#eb6834", "config only": "#1baf7a"}


def main(results_path: str, out_path: str) -> None:
    with open(results_path) as f:
        res = json.load(f)
    sizes = np.array(res["sizes"])
    tests = list(res["tests"])
    fig, axes = plt.subplots(1, len(tests), figsize=(11, 4.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, test in zip(axes, tests):
        r = res["tests"][test]
        ax.set_facecolor(SURFACE)
        ends = {label: float(np.mean(r[label][-1])) for label in SERIES}
        label_y, prev = {}, None
        for label in sorted(ends, key=ends.get):  # nudge end labels apart
            y = np.log10(ends[label])
            if prev is not None and y - prev < 0.045:
                y = prev + 0.045
            label_y[label], prev = 10**y, y
        for label, color in SERIES.items():
            vals = np.array(r[label])
            mean = vals.mean(1)
            ax.fill_between(
                sizes, vals.min(1), vals.max(1), color=color, alpha=0.15, lw=0
            )
            ax.plot(
                sizes,
                mean,
                color=color,
                lw=2,
                marker="o",
                ms=6,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label=label,
            )
            ax.annotate(
                label,
                (sizes[-1], label_y[label]),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                fontsize=10,
                color=INK,
            )
        for label in ["power law", "last value"]:
            ax.axhline(r[label], color=MUTED, lw=1, ls=(0, (4, 3)))
            ax.annotate(
                f"{label} (no corpus)",
                (sizes[0], r[label]),
                xytext=(0, 4),
                textcoords="offset points",
                fontsize=9,
                color=MUTED,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(test, loc="left", fontsize=12, color=INK)
        ax.set_xlabel("runs in training corpus", color=MUTED)
        ax.grid(True, which="major", color=GRID, lw=0.8)
        ax.set_xlim(sizes[0] * 0.8, sizes[-1] * 3.2)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED)
    axes[0].set_ylabel("forecast error (mean |Δ log10 loss|)", color=MUTED)
    axes[0].legend(frameon=False, loc="lower left", fontsize=10)
    fig.suptitle(
        "Forecasting the final loss from the first 10% of a run",
        x=0.01,
        ha="left",
        fontsize=14,
        color=INK,
    )
    fig.text(
        0.01,
        0.905,
        "Band: min-max over 5 resampled corpora. Dashed: baselines that use no corpus.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
