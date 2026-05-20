"""Sim-to-real calibration statistics tests.

**In plain words.** If this test ever fails, the headline
calibration number the suite reports (`Pearson r = +0.875 [+0.733,
+0.970]`) is suspect — the math producing it is broken.

Pearson r + bootstrap CI + MMRV — the substrate the tier-A and
tier-A+ rules in EXTENSION.md §4 rest on.
"""

from __future__ import annotations

import numpy as np
import pytest
from eval_suite.calibration import (
    mmrv,
    paired_cell_data,
    pearson_r_with_bootstrap_ci,
)


def test_pearson_r_against_perfectly_correlated_data() -> None:
    sim = np.array([0.10, 0.30, 0.50, 0.70, 0.90, 0.20, 0.40, 0.60, 0.80, 0.55])
    real = sim.copy()  # perfect correlation
    result = pearson_r_with_bootstrap_ci(sim, real, n_bootstrap=1000, seed=42)
    assert result.r == pytest.approx(1.0, abs=1e-9)
    assert result.n_pairs == 10


def test_pearson_r_against_anti_correlated_data() -> None:
    sim = np.linspace(0.1, 0.9, 10)
    real = sim[::-1].copy()  # perfect anti-correlation
    result = pearson_r_with_bootstrap_ci(sim, real, n_bootstrap=1000, seed=42)
    assert result.r == pytest.approx(-1.0, abs=1e-9)


def test_pearson_r_known_correlation_in_ci() -> None:
    """For noisy linear data, the recovered r should be high and the
    bootstrap CI must contain the point estimate."""
    rng = np.random.default_rng(0)
    n = 50
    sim = rng.uniform(0, 1, n)
    real = 0.6 * sim + 0.4 * rng.uniform(0, 1, n)
    result = pearson_r_with_bootstrap_ci(sim, real, n_bootstrap=5000, seed=7)
    assert 0.4 < result.r < 0.95
    assert result.ci_low <= result.r <= result.ci_high
    # Bootstrap CI should be reasonably tight for n=50.
    assert (result.ci_high - result.ci_low) < 0.5


def test_pearson_r_fewer_than_three_pairs_returns_nan() -> None:
    result = pearson_r_with_bootstrap_ci([0.5, 0.7], [0.4, 0.6])
    assert np.isnan(result.r)
    assert result.n_pairs == 2


def test_pearson_r_zero_variance_returns_nan() -> None:
    sim = np.zeros(5)
    real = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    result = pearson_r_with_bootstrap_ci(sim, real, n_bootstrap=100, seed=0)
    assert np.isnan(result.r)


def test_pearson_r_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        pearson_r_with_bootstrap_ci([0.1, 0.2, 0.3], [0.1, 0.2])


def test_mmrv_zero_for_identical_trajectories() -> None:
    traj = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
    assert mmrv(traj, traj.copy(), dt=0.1) == pytest.approx(0.0, abs=1e-12)


def test_mmrv_increases_with_velocity_difference() -> None:
    t = np.linspace(0, 1, 20)[:, None]
    real = np.hstack([t, t])                                  # slope 1 in both dims
    sim_close = np.hstack([t * 1.1, t * 1.1])                  # 10% velocity error
    sim_far = np.hstack([t * 2.0, t * 2.0])                    # 100% velocity error
    assert mmrv(sim_close, real, dt=0.05) < mmrv(sim_far, real, dt=0.05)


def test_mmrv_shape_mismatch_raises() -> None:
    a = np.zeros((10, 3))
    b = np.zeros((10, 4))
    with pytest.raises(ValueError, match="shape mismatch"):
        mmrv(a, b)


def test_mmrv_too_short_trajectory_raises() -> None:
    a = np.zeros((1, 3))
    with pytest.raises(ValueError, match="T>=2"):
        mmrv(a, a)


def test_paired_cell_data_matches_axis_subset() -> None:
    """A registry entry whose cell_axes is a subset of the manifest cell's
    axes should pair; an entry referencing an unrelated axis-value should not."""

    class _Cell:
        def __init__(self, axes: dict[str, str], successes: int, n_trials: int) -> None:
            self.axes = axes
            self.successes = successes
            self.n_trials = n_trials

    cells = [
        _Cell({"orientation": "vertical", "lighting": "darker"}, 12, 20),
        _Cell({"orientation": "horizontal", "lighting": "darker"}, 18, 20),
    ]
    entries = [
        {"task_key": "T", "model_key": "M", "cell_axes": {"orientation": "vertical"}, "value": 0.60},
        {"task_key": "T", "model_key": "M", "cell_axes": {"orientation": "horizontal"}, "value": 0.90},
        {"task_key": "T", "model_key": "M", "cell_axes": {"orientation": "diagonal"}, "value": 0.50},
    ]
    out_cells, sim_rates, real_rates = paired_cell_data(
        manifest_cells=cells, registry_entries=entries, task_key="T", model_key="M",
    )
    assert len(out_cells) == 2
    assert sim_rates == [0.6, 0.9]
    assert real_rates == [0.6, 0.9]


def test_paired_cell_data_skips_entries_missing_cell_axes() -> None:
    """The legacy aggregate registry entries (no cell_axes) must NOT match."""

    class _Cell:
        def __init__(self, axes: dict[str, str], successes: int, n_trials: int) -> None:
            self.axes = axes
            self.successes = successes
            self.n_trials = n_trials

    cells = [_Cell({"orientation": "vertical"}, 10, 20)]
    entries = [
        {"task_key": "T", "model_key": "M", "value": 0.30},  # aggregate, no cell_axes
    ]
    out_cells, sim_rates, real_rates = paired_cell_data(
        manifest_cells=cells, registry_entries=entries, task_key="T", model_key="M",
    )
    assert out_cells == []
