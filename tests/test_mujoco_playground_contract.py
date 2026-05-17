"""End-to-end v0 contract: MujocoPlaygroundAdapter + Go1 Task +
RandomLocomotionPolicy + Manifest, with a MockGo1Env so it runs on CI.

Asserts:
1. JointAction flattens to a 12-dim ndarray and passes env.step()'s shape check.
2. run_sweep() drives the joint-space rollout, emits CSV with same schema.
3. Manifest seals and verifies for a legged embodiment.
4. The Go1 Task's optional hooks (instruction_for, extract_image) work.
5. Cell IDs from the Go1 catalog parse cleanly (terrain × camera × perturbation).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from eval_suite._types import Action, JointAction, Observation
from eval_suite.adapters import GymAdapter, MujocoPlaygroundAdapter
from eval_suite.policies import RandomLocomotionPolicy
from eval_suite.sweep import run_sweep
from eval_suite.tasks.unitree_go1 import GO1_ACTION_DIM, UnitreeGo1Joystick, mock_go1_task_factory


def test_joint_action_flatten() -> None:
    """Action and JointAction both flatten through the same Adapter helper."""
    eef = Action(
        world_vector=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        rot_axangle=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        gripper=np.array([1.0], dtype=np.float32),
    )
    joint = JointAction(vector=np.arange(12, dtype=np.float32) * 0.01)
    assert GymAdapter._flatten_action(eef).shape == (7,)
    assert GymAdapter._flatten_action(joint).shape == (12,)
    assert MujocoPlaygroundAdapter._flatten_action(joint).shape == (12,)


def test_random_locomotion_policy_emits_jointaction() -> None:
    pol = RandomLocomotionPolicy(action_dim=12, seed=42)
    pol.reset("test instruction")
    obs = Observation(image=np.zeros((1, 1, 3), dtype=np.uint8), instruction="test")
    a = pol.step(obs)
    assert isinstance(a, JointAction)
    assert a.vector.shape == (GO1_ACTION_DIM,)
    assert a.vector.dtype == np.float32
    # Determinism: same instruction → same first action.
    pol.reset("test instruction")
    a2 = pol.step(obs)
    assert np.array_equal(a.vector, a2.vector)


def test_go1_task_cell_catalog() -> None:
    task = UnitreeGo1Joystick()
    assert task.embodiment == "unitree_go1"
    assert task.action_dim == GO1_ACTION_DIM
    # 3 task families × 4 vary-one-axis cells = 12.
    assert task.n_cells == 12
    cell0 = task.cell_id(0)
    assert set(cell0.axes.keys()) == {"task_family", "terrain", "camera", "perturbation"}
    # Three baseline cells (one per task family): flat/front/none.
    baselines = [task.cell_id(i) for i in range(task.n_cells)
                 if task.cell_id(i).axes["terrain"] == "flat"
                 and task.cell_id(i).axes["camera"] == "front"
                 and task.cell_id(i).axes["perturbation"] == "none"]
    assert len(baselines) == 3


def test_go1_task_hooks_present() -> None:
    task = UnitreeGo1Joystick()
    assert callable(getattr(task, "instruction_for", None))
    assert callable(getattr(task, "extract_image", None))


def test_mujoco_adapter_full_rollout(tmp_path: Path) -> None:
    """Drive a real run_sweep through the v0 Adapter against MockGo1Env."""
    task = mock_go1_task_factory(n_cells=2, max_episode_steps=5)
    policy = RandomLocomotionPolicy(action_dim=GO1_ACTION_DIM, seed=7)
    adapter = MujocoPlaygroundAdapter()

    manifest = run_sweep(
        policy=policy,
        task=task,
        adapter=adapter,
        seeds=[0, 1, 2],
        output_dir=tmp_path,
        code_sha="v0-contract-test",
    )

    assert manifest.verify()
    assert manifest.embodiment == "unitree_go1"
    assert manifest.task_name == "unitree_go1_joystick"
    assert manifest.trials_per_cell == 3
    assert len(manifest.cells) == 2

    # CSV checks
    rows = list(csv.DictReader((tmp_path / "trials.csv").open()))
    assert len(rows) == 6  # 2 cells × 3 seeds
    assert all(r["embodiment"] == "unitree_go1" for r in rows)
    assert all(r["model_name"].startswith("random-locomotion") for r in rows)
    for r in rows:
        # JointAction → 12-dim env_action → MockGo1Env's shape assert passes
        # (otherwise we'd have got "Go1 action must be 12-dim" mid-sweep).
        axes = json.loads(r["cell_axes"])
        assert "task_family" in axes


def test_go1_task_real_build_env_errors_clearly() -> None:
    """Without mujoco_playground installed, the real build_env raises a
    clear error pointing the user at the workaround."""
    task = UnitreeGo1Joystick()
    try:
        task.build_env(0)
    except ImportError as e:
        assert "mujoco_playground" in str(e)
        assert "mock_go1_task_factory" in str(e) or "pip install" in str(e)
        return
    # If it didn't raise, mujoco_playground happens to be installed on
    # this runner — that's fine, just not the path this test covers.
