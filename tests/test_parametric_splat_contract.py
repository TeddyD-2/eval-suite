"""ParametricSplatTask contract tests (CI-runnable via the mock factory).

**In plain words.** Pins down the v1 splat-task generator end to
end: any combination of axes + predicates produces a sweepable
task, the predicate flows into the manifest as part of the
fingerprint, and changing the predicate really does produce a
different run_id. This is the test that protects the factory-
engineer wedge.

Covers Protocol compliance, cell-grid decode, canonical-axis mapping,
success-criterion serialization, sweep-end-to-end through the existing
`MujocoPlaygroundAdapter`, and the substrate determinism property
(changing the predicate changes `run_id`).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import get_args

import pytest
from eval_suite._types import CanonicalDim, CellId
from eval_suite.adapters.mujoco_playground import MujocoPlaygroundAdapter
from eval_suite.contracts import Task
from eval_suite.manifest import Manifest
from eval_suite.policies.random_locomotion import RandomLocomotionPolicy
from eval_suite.sweep import CSV_COLUMNS, run_sweep
from eval_suite.tasks._success_predicates import (
    MaintainedClearance,
    RobotReachedRegion,
    predicate_from_dict,
)
from eval_suite.tasks.parametric_splat import (
    mock_tnt_truck_splat_task_factory,
)


def test_parametric_splat_task_satisfies_protocol() -> None:
    task = mock_tnt_truck_splat_task_factory()
    assert isinstance(task, Task)
    assert task.name == "tnt_truck_splat_v01"
    assert task.embodiment == "unitree_go1"
    assert task.n_cells == 9  # 3 lighting × 3 camera
    cell0 = task.cell_id(0)
    assert isinstance(cell0, CellId)
    assert cell0.embodiment == "unitree_go1"
    assert set(cell0.axes.keys()) == {"lighting", "camera"}
    assert task.max_episode_steps > 0


def test_n_cells_equals_product_of_axes() -> None:
    """ParametricSplatTask must decode the axes dict into n_cells =
    prod(len(level_list))."""
    task = mock_tnt_truck_splat_task_factory()
    # 3 × 3 = 9 from the standard axes.
    assert task.n_cells == 3 * 3


def test_cell_id_round_trip_deterministic() -> None:
    """cell_id(i) returns the same (axes-dict) value on every call AND
    every cell maps to a unique (lighting, camera) pair."""
    task = mock_tnt_truck_splat_task_factory()
    n = task.n_cells
    seen: set[tuple[tuple[str, str], ...]] = set()
    for i in range(n):
        c1 = task.cell_id(i)
        c2 = task.cell_id(i)
        assert c1 == c2
        axes_tup = tuple(sorted(c1.axes.items()))
        assert axes_tup not in seen, (
            f"cell_id({i}) collided with a previous cell: {c1.axes}"
        )
        seen.add(axes_tup)
    assert len(seen) == n


def test_cell_id_out_of_range_raises_index_error() -> None:
    task = mock_tnt_truck_splat_task_factory()
    with pytest.raises(IndexError):
        task.cell_id(-1)
    with pytest.raises(IndexError):
        task.cell_id(task.n_cells)


def test_canonical_axis_map_projects_to_valid_canonical_dims() -> None:
    task = mock_tnt_truck_splat_task_factory()
    valid = set(get_args(CanonicalDim))
    for axis_name, dim in task.canonical_axis_map.items():
        assert dim in valid, (
            f"axis {axis_name!r} maps to {dim!r}, which is not in CanonicalDim {valid}"
        )


def test_success_criterion_property_round_trips_through_predicate_registry() -> None:
    """The task's `success_criterion` is bound into the manifest as a
    dict; predicate_from_dict must reconstruct an equivalent predicate
    object. This is the load-bearing property that makes "configure,
    don't code" work — the manifest is self-describing, the predicate
    is reconstructible."""
    task = mock_tnt_truck_splat_task_factory()
    crit = task.success_criterion
    assert crit["kind"] == "robot_reached_region"
    rebuilt = predicate_from_dict(crit)
    assert isinstance(rebuilt, RobotReachedRegion)
    assert rebuilt.region_name == "behind_truck"
    assert rebuilt.tolerance == 0.5


def test_sweep_through_mujoco_adapter_produces_verifying_manifest(tmp_path: Path) -> None:
    """End-to-end through the existing MujocoPlaygroundAdapter (unchanged):
    9 cells × 3 seeds = 27 trial rows; sealed manifest; verify() True;
    `manifest.success_criterion` populated."""
    task = mock_tnt_truck_splat_task_factory(max_episode_steps=10)
    policy = RandomLocomotionPolicy()
    adapter = MujocoPlaygroundAdapter()
    manifest = run_sweep(
        policy=policy,
        task=task,
        adapter=adapter,
        seeds=[0, 1, 2],
        output_dir=tmp_path,
        code_sha="test-sha",
    )
    assert manifest.verify() is True
    assert manifest.schema_version == "0.3.0"
    assert manifest.success_criterion == {
        "kind": "robot_reached_region",
        "params": {"region_name": "behind_truck", "tolerance": 0.5},
    }
    assert manifest.canonical_axis_map == {"lighting": "visuals", "camera": "visuals"}

    # 9 cells × 3 seeds = 27 rows.
    csv_path = tmp_path / "trials.csv"
    assert csv_path.is_file()
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 27
    assert set(rows[0].keys()) == set(CSV_COLUMNS)
    for r in rows:
        axes = json.loads(r["cell_axes"])
        assert set(axes.keys()) == {"lighting", "camera"}


def test_changing_predicate_changes_run_id(tmp_path: Path) -> None:
    """The whole point of the schema 0.3.0 substrate change: two sweeps
    of the same scene with different predicates must produce distinct
    run_ids. Without this, a factory engineer who swaps the goal on
    their captured warehouse would collide on run_id with their
    previous evaluation."""
    task_a = mock_tnt_truck_splat_task_factory(
        max_episode_steps=5,
        success_predicate=RobotReachedRegion(region_name="behind_truck", tolerance=0.5),
    )
    task_b = mock_tnt_truck_splat_task_factory(
        max_episode_steps=5,
        success_predicate=MaintainedClearance(region_name="behind_truck", min_distance=0.3),
    )
    policy = RandomLocomotionPolicy()
    adapter = MujocoPlaygroundAdapter()
    m_a = run_sweep(
        policy=policy, task=task_a, adapter=adapter,
        seeds=[0], output_dir=tmp_path / "a", code_sha="fixed-test-sha",
    )
    m_b = run_sweep(
        policy=policy, task=task_b, adapter=adapter,
        seeds=[0], output_dir=tmp_path / "b", code_sha="fixed-test-sha",
    )
    assert m_a.verify() and m_b.verify()
    assert m_a.run_id != m_b.run_id, (
        "Predicate change did NOT alter run_id — schema 0.3.0 binding broken."
    )
    # And the criterion dicts in the two manifests differ:
    assert m_a.success_criterion != m_b.success_criterion


def test_manifest_round_trips_through_json(tmp_path: Path) -> None:
    """Sealed manifests with the new field must round-trip through to_json /
    from_json and re-verify."""
    task = mock_tnt_truck_splat_task_factory(max_episode_steps=5)
    policy = RandomLocomotionPolicy()
    adapter = MujocoPlaygroundAdapter()
    m = run_sweep(
        policy=policy, task=task, adapter=adapter,
        seeds=[0, 1], output_dir=tmp_path, code_sha="round-trip-sha",
    )
    j = m.to_json()
    parsed = json.loads(j)
    assert "success_criterion" in parsed  # to_json includes the key (even when None)
    loaded = Manifest.from_json(j)
    assert loaded.verify() is True
    assert loaded.run_id == m.run_id
    assert loaded.success_criterion == m.success_criterion
