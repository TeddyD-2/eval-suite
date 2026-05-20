"""Per-cell statistics — Wilson CIs and per-axis aggregation.

**In plain words.** When a paper reports "60% success" you can't tell
whether that came from 6 wins out of 10 trials (very noisy) or 60 out
of 100 (much more solid). This file is what turns a count of wins and
trials into an honest 95% confidence interval — the bracket that says
"the true success rate is somewhere in this range." Every per-cell
number the suite reports goes through this file so the reader can see
how shaky or solid each number actually is.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from ._types import CellId, CellResult


def wilson_ci(successes: int, n_trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson 95% confidence interval for a binomial proportion.

    Returns (low, high). Edge cases: n_trials == 0 returns (0.0, 1.0).

    The z=1.959963984540054 default is the two-sided 95% normal quantile
    to higher precision than 1.96 (which produces visibly different CI
    widths at small N). statsmodels uses the same constant.
    """
    if n_trials < 0 or successes < 0 or successes > n_trials:
        raise ValueError(f"invalid inputs: {successes=} {n_trials=}")
    if n_trials == 0:
        return (0.0, 1.0)
    p = successes / n_trials
    denom = 1.0 + (z * z) / n_trials
    center = (p + (z * z) / (2 * n_trials)) / denom
    half = (z * math.sqrt(p * (1 - p) / n_trials + (z * z) / (4 * n_trials * n_trials))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def cell_result_from_rollouts(cell_slug_to_successes: Iterable[bool], cell: CellId, /) -> CellResult:
    """Build a CellResult from a stream of per-seed booleans.

    `cell` is the CellId (passed positionally only to keep the iterable
    arg in the natural first position).
    """
    successes = [bool(x) for x in cell_slug_to_successes]
    n = len(successes)
    s = sum(successes)
    low, high = wilson_ci(s, n)
    return CellResult(
        cell=cell,
        n_trials=n,
        successes=s,
        wilson_ci_low=low,
        wilson_ci_high=high,
        per_seed_success=successes,
    )


def per_axis_means(cells: Iterable[CellResult]) -> dict[str, dict[str, float]]:
    """For each axis, compute the mean success rate across its levels.

    Returns {axis_name: {level: mean_success_rate}}. Each level's value is
    the mean across all cells where that axis takes that level.
    """
    out: dict[str, dict[str, list[float]]] = {}
    for cr in cells:
        for axis, level in cr.cell.axes.items():
            out.setdefault(axis, {}).setdefault(level, []).append(cr.success_rate)
    return {axis: {level: (sum(vs) / len(vs)) for level, vs in levels.items()} for axis, levels in out.items()}


def worst_axis(per_axis: Mapping[str, Mapping[str, float]]) -> tuple[str, float]:
    """Returns (axis_name, min_level_mean) — the worst-axis ranking metric.

    For each axis, takes the *mean* across its levels. The worst axis is
    the one with the lowest level-mean. The level-count bias and the v1.0
    deployment-relevance-weighting fix are discussed in
    `takehome/EXTENSION.md` §3.
    """
    if not per_axis:
        raise ValueError("per_axis is empty")
    axis_means: dict[str, float] = {}
    for axis, levels in per_axis.items():
        vals = list(levels.values())
        axis_means[axis] = sum(vals) / len(vals) if vals else 0.0
    worst = min(axis_means.items(), key=lambda kv: kv[1])
    return worst
