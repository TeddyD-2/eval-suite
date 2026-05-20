"""paired_trials.json sidecar contract tests.

**In plain words.** Pins down the shape of the on-disk artifact a
partner lab produces when they pair a sim rollout with a real
trajectory. If this test ever fails, partner-contributed
trajectory-level calibration data can't be ingested safely.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from eval_suite._types import CellId
from eval_suite.calibration import (
    PairedTrial,
    paired_trials_sidecar_path,
    record_paired_trial,
)


def test_paired_trial_from_trajectories_records_mmrv() -> None:
    cell = CellId(embodiment="emb", task="task", axes={"axis": "v"})
    traj = np.linspace(0.0, 1.0, 50)[:, None]
    sim = traj * 1.5  # 50% velocity error
    pt = PairedTrial.from_trajectories(
        cell=cell, sim_run_id="abc", sim_seed=7,
        sim_traj=sim, real_traj=traj, real_episode_ref="oxe:test#0",
    )
    assert pt.cell_id_slug == cell.slug
    assert pt.sim_seed == 7
    assert pt.real_episode_ref == "oxe:test#0"
    assert pt.mmrv_score > 0.0
    assert pt.n_steps_compared == 50


def test_paired_trial_truncates_to_shorter_traj() -> None:
    cell = CellId(embodiment="e", task="t", axes={"a": "v"})
    sim = np.linspace(0.0, 1.0, 100)[:, None]
    real = np.linspace(0.0, 1.0, 50)[:, None]
    pt = PairedTrial.from_trajectories(
        cell=cell, sim_run_id="r", sim_seed=0,
        sim_traj=sim, real_traj=real, real_episode_ref="x",
    )
    assert pt.n_steps_compared == 50


def test_paired_trials_sidecar_round_trip(tmp_path: Path) -> None:
    sidecar = paired_trials_sidecar_path(tmp_path)
    cell = CellId(embodiment="e", task="t", axes={"axis": "v1"})
    pt = PairedTrial.from_trajectories(
        cell=cell, sim_run_id="run1", sim_seed=0,
        sim_traj=np.linspace(0, 1, 10)[:, None],
        real_traj=np.linspace(0, 1, 10)[:, None],
        real_episode_ref="oxe:test#0",
    )
    record_paired_trial(
        sidecar_path=sidecar, manifest_run_id="run1",
        task_key="t", model_key="m", paired_trial=pt,
    )
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["manifest_run_id"] == "run1"
    assert payload["task_key"] == "t"
    assert payload["model_key"] == "m"
    assert len(payload["pairs"]) == 1
    assert payload["pairs"][0]["sim_seed"] == 0


def test_paired_trials_sidecar_appends_multiple_pairs(tmp_path: Path) -> None:
    sidecar = paired_trials_sidecar_path(tmp_path)
    cell = CellId(embodiment="e", task="t", axes={"a": "v"})
    sim = np.linspace(0, 1, 10)[:, None]
    real = np.linspace(0, 1, 10)[:, None]
    for seed in range(3):
        pt = PairedTrial.from_trajectories(
            cell=cell, sim_run_id="run1", sim_seed=seed,
            sim_traj=sim, real_traj=real, real_episode_ref=f"oxe:t#{seed}",
        )
        record_paired_trial(
            sidecar_path=sidecar, manifest_run_id="run1",
            task_key="t", model_key="m", paired_trial=pt,
        )
    payload = json.loads(sidecar.read_text())
    assert len(payload["pairs"]) == 3


def test_paired_trials_sidecar_refuses_to_mix_manifest_run_ids(tmp_path: Path) -> None:
    sidecar = paired_trials_sidecar_path(tmp_path)
    cell = CellId(embodiment="e", task="t", axes={"a": "v"})
    sim = np.linspace(0, 1, 10)[:, None]
    real = np.linspace(0, 1, 10)[:, None]
    pt1 = PairedTrial.from_trajectories(
        cell=cell, sim_run_id="run-A", sim_seed=0, sim_traj=sim, real_traj=real,
        real_episode_ref="x",
    )
    record_paired_trial(
        sidecar_path=sidecar, manifest_run_id="run-A",
        task_key="t", model_key="m", paired_trial=pt1,
    )
    pt2 = PairedTrial.from_trajectories(
        cell=cell, sim_run_id="run-B", sim_seed=0, sim_traj=sim, real_traj=real,
        real_episode_ref="y",
    )
    with pytest.raises(ValueError, match="refusing to mix"):
        record_paired_trial(
            sidecar_path=sidecar, manifest_run_id="run-B",
            task_key="t", model_key="m", paired_trial=pt2,
        )
