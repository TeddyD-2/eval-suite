"""Adapter-compatibility test: the existing MujocoPlaygroundAdapter
accepts a ParametricSplatTask unmodified.

This is the proof that the v1 splat path satisfies the eval-suite's
"absorb new task families through one new Task + one new sidecar with
zero changes to contracts/sweep/adapter" claim from EXTENSION.md §2.
"""

from __future__ import annotations

from eval_suite.adapters.mujoco_playground import MujocoPlaygroundAdapter
from eval_suite.policies.random_locomotion import RandomLocomotionPolicy
from eval_suite.tasks.parametric_splat import mock_tnt_truck_splat_task_factory


def test_adapter_unmodified_accepts_parametric_splat_task() -> None:
    """Instantiate the existing MujocoPlaygroundAdapter (no splat-aware
    changes), call rollout against the mock task, assert no errors and
    a well-formed RolloutResult."""
    task = mock_tnt_truck_splat_task_factory(max_episode_steps=5)
    policy = RandomLocomotionPolicy()
    adapter = MujocoPlaygroundAdapter()
    result = adapter.rollout(policy=policy, task=task, cell=0, seed=0)
    assert isinstance(result.success, bool)
    assert result.num_steps > 0
    assert result.elapsed_wall_seconds >= 0.0
    # Adapter sets RolloutResult.cell to the task's cell_id(0) — same
    # contract as Namaqualand and SimplerEnv tasks.
    assert result.cell == task.cell_id(0)
