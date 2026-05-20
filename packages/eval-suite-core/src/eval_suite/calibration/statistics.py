"""Sim-to-real calibration statistics.

Two metrics, two regimes:

  - **Outcome-level Pearson r** (per-task-model): correlation of
    sim-success-rate vs. real-success-rate across the cells where
    paired numbers exist. Reported with a 95% bootstrap CI; tier-A in
    EXTENSION.md §4 requires ≥10 paired cells.

  - **MMRV** (Maximum Mean Relative Velocity error): per paired
    (sim, real) trajectory, the mean velocity error normalized by the
    real trajectory's velocity scale, taken as the max over the
    trajectory. SIMPLER-style trajectory-level metric for evaluating
    whether the sim physics matches reality at the rollout level —
    not just the outcome level.

Inputs are deliberately concrete numpy arrays and dataclasses, not
sim-specific. The OXEReplayPolicy from Phase 1 supplies the recorded
real-side trajectory; the existing rollout artifact (`trajectory.npz`)
supplies the sim-side trajectory. The Manifest + calibration registry
supply the paired outcome rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._types import CellId


@dataclass(frozen=True)
class PearsonCI:
    """Pearson correlation coefficient with a stratified-bootstrap 95% CI."""

    r: float
    ci_low: float
    ci_high: float
    n_pairs: int


def pearson_r_with_bootstrap_ci(
    sim: list[float] | np.ndarray[Any, Any],
    real: list[float] | np.ndarray[Any, Any],
    *,
    n_bootstrap: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> PearsonCI:
    """Pearson r between paired (sim, real) success rates + bootstrap CI.

    Cell-stratified percentile bootstrap: resample (sim_i, real_i)
    pairs *with replacement* `n_bootstrap` times; report the
    [alpha/2, 1-alpha/2] quantiles of the resampled r distribution.

    Edge cases:
      - <3 paired cells → r=NaN, CI=(NaN, NaN).
      - Zero-variance sim or real → r=NaN (Pearson is undefined).
    """
    sim_arr = np.asarray(sim, dtype=np.float64)
    real_arr = np.asarray(real, dtype=np.float64)
    if sim_arr.shape != real_arr.shape:
        raise ValueError(f"sim/real shape mismatch: {sim_arr.shape} vs {real_arr.shape}")
    n = len(sim_arr)
    if n < 3:
        return PearsonCI(r=float("nan"), ci_low=float("nan"), ci_high=float("nan"), n_pairs=n)
    r = _pearson(sim_arr, real_arr)
    rng = np.random.default_rng(seed)
    rs = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        rs[i] = _pearson(sim_arr[idx], real_arr[idx])
    finite = rs[np.isfinite(rs)]
    if len(finite) < 2:
        return PearsonCI(r=r, ci_low=float("nan"), ci_high=float("nan"), n_pairs=n)
    lo, hi = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return PearsonCI(r=float(r), ci_low=float(lo), ci_high=float(hi), n_pairs=n)


def _pearson(x: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> float:
    xv = x - x.mean()
    yv = y - y.mean()
    denom = float(np.sqrt((xv * xv).sum()) * np.sqrt((yv * yv).sum()))
    if denom == 0.0:
        return float("nan")
    return float((xv * yv).sum() / denom)


def mmrv(
    sim_traj: np.ndarray[Any, Any],
    real_traj: np.ndarray[Any, Any],
    *,
    dt: float = 0.05,
) -> float:
    """Maximum Mean Relative Velocity error.

    `sim_traj` and `real_traj` are (T, D) arrays of positions (joint
    angles or EEF xyz — pick one). The metric:

      1. Compute per-step velocity: v[t] = (x[t+1] - x[t]) / dt.
      2. Per-step error: e[t] = ||v_sim[t] - v_real[t]||.
      3. Per-step normalizer: s[t] = max(||v_real[t]||, eps).
      4. Per-step relative error: r[t] = e[t] / s[t].
      5. **mean** over D dimensions → per-step scalar; **max** over T.

    Returns a scalar; lower is better. 0.0 means sim and real
    trajectories have identical velocities.
    """
    if sim_traj.shape != real_traj.shape:
        raise ValueError(f"sim/real trajectory shape mismatch: {sim_traj.shape} vs {real_traj.shape}")
    if sim_traj.ndim != 2 or sim_traj.shape[0] < 2:
        raise ValueError(f"trajectories must be 2D with T>=2; got shape {sim_traj.shape}")
    v_sim = np.diff(sim_traj, axis=0) / dt
    v_real = np.diff(real_traj, axis=0) / dt
    err = np.linalg.norm(v_sim - v_real, axis=1)
    scale = np.maximum(np.linalg.norm(v_real, axis=1), 1e-6)
    return float(np.max(err / scale))


def paired_cell_data(
    *,
    manifest_cells: list[Any],
    registry_entries: list[dict[str, Any]],
    task_key: str,
    model_key: str,
) -> tuple[list[CellId], list[float], list[float]]:
    """Pair the sim-side per-cell rates from a Manifest with the
    real-side per-cell rates from the calibration registry.

    A registry entry pairs with a manifest cell when:
      - entry.task_key == task_key
      - entry.model_key == model_key
      - entry.cell_axes is a sub-mapping of the manifest cell's axes
        (so a cell with axes {orientation: vertical, lighting: darker}
        matches both a registry entry for {orientation: vertical} and
        one for {orientation: vertical, lighting: darker} — the
        cell-axis match is "all entry keys present and equal").

    Returns three same-length lists: (cell_ids, sim_rates, real_rates).
    Order is the manifest's cell order, filtered to those with a
    matching registry entry.
    """
    matching_entries = [
        e for e in registry_entries
        if e.get("task_key") == task_key and e.get("model_key") == model_key and "cell_axes" in e
    ]
    out_cells: list[CellId] = []
    out_sim: list[float] = []
    out_real: list[float] = []
    for cell in manifest_cells:
        axes = dict(cell.axes)
        for entry in matching_entries:
            entry_axes = entry.get("cell_axes") or {}
            if all(axes.get(k) == v for k, v in entry_axes.items()):
                cid = CellId(embodiment="", task=task_key, axes=axes)
                sim_rate = float(cell.successes) / float(max(cell.n_trials, 1))
                out_cells.append(cid)
                out_sim.append(sim_rate)
                out_real.append(float(entry["value"]))
                break  # first match wins
    return out_cells, out_sim, out_real
