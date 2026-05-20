"""MuJoCo Playground Adapter — drives a Policy against an MJX-shaped env.

**In plain words.** The sibling of `GymAdapter`, but for legged
robots. Legs need joint-level commands (not the 7-DoF end-effector
deltas an arm sim uses), so the suite needs a different driver for
them. This file is that driver. Same rollout loop; different action
shape and observation flavor. Its presence is the proof the
contract isn't accidentally arm-only.


The v0 sibling of `GymAdapter`. The split exists because legged
embodiments emit joint-space actions (12-DoF for Go1, not 7-DoF EEF),
and because MJX envs use a slightly different observation contract
than gymnasium (`obs` is typically a flat ndarray of proprio + IMU,
not a dict with an "image" key — rendering is a separate hook).

Like `GymAdapter`, this Adapter only needs the env's `reset` and
`step` methods plus two optional Task-level hooks
(`instruction_for(env)` and `extract_image(env, obs)`). The Task is
responsible for wrapping MJX into a gymnasium-shaped interface (5-tuple
step, dict-or-array obs); see `eval_suite/tasks/unitree_go1.py` for
the reference shim.

This module is import-cheap. MuJoCo Playground is lazy-imported only
when a real Task constructs an env. Tests that drive this Adapter use
a MockEnv to keep CI fast and dep-free.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .._types import Action, ActionLike, CellId, JointAction, NDArrayF32, NDArrayU8, Observation, RolloutResult
from ..contracts import Policy, Task
from ..rollout_data import (
    RolloutMetadata,
    Trajectory,
    joint_action_components,
    write_rollout_artifacts,
)

_VIDEO_FPS = 30  # legged sims typically run higher fps than manipulation


class MujocoPlaygroundAdapter:
    """Drives one (cell, seed) rollout through an MJX-shaped env.

    Differences from `GymAdapter` worth knowing:
    - Action type: defaults to `JointAction` for legged. Accepts `Action`
      for compatibility, but legged Policies should emit `JointAction`.
    - Observation: MJX envs frequently emit flat ndarrays; the default
      `extract_image` hook tries the standard MJX render path
      (`env.render()` when available) rather than walking an obs dict.
    - Video FPS: defaults to 30 (legged is per-step PD-control at 50Hz;
      we downsample on capture).
    """

    def __init__(
        self,
        *,
        max_steps_override: int | None = None,
        videos_dir: str | Path | None = None,
        video_fps: int = _VIDEO_FPS,
    ) -> None:
        self._max_steps_override = max_steps_override
        self._rollouts_dir: Path | None = Path(videos_dir) if videos_dir is not None else None
        self._video_fps = video_fps

    @property
    def name(self) -> str:
        return "mujoco-playground-adapter"

    def rollout(self, policy: Policy, task: Task, cell: int, seed: int) -> RolloutResult:
        env = task.build_env(cell)
        try:
            return self._rollout_with_env(env, policy, task, task.cell_id(cell), task.max_episode_steps, seed)
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def _rollout_with_env(
        self,
        env: Any,
        policy: Policy,
        task: Task,
        cell: CellId,
        env_max_steps: int,
        seed: int,
    ) -> RolloutResult:
        max_steps = self._max_steps_override or env_max_steps
        start = time.monotonic()
        capture_artifacts = self._rollouts_dir is not None
        frames: list[NDArrayU8] = []
        per_step_actions: list[NDArrayF32] = []
        per_step_rewards: list[float] = []
        per_step_success: list[bool] = []
        per_step_truncated: list[bool] = []

        try:
            obs, _info = env.reset(seed=seed)
        except TypeError:
            obs, _info = env.reset()

        instruction = self._instruction(task, env)
        policy.reset(instruction)

        image = self._image(task, env, obs)
        observation = Observation(image=image, instruction=instruction, extra={"obs": obs})
        if capture_artifacts and image is not None:
            frames.append(image)

        success = False
        truncated = False
        info: dict[str, Any] = {}
        step_idx = 0
        action_dim_seen = 0

        while step_idx < max_steps and not truncated:
            action = policy.step(observation)
            if action.terminate:
                break
            env_action = self._flatten_action(action)
            obs, reward, success, truncated, info = env.step(env_action)
            step_idx += 1
            if capture_artifacts:
                per_step_actions.append(env_action)
                per_step_rewards.append(float(reward) if reward is not None else 0.0)
                per_step_success.append(bool(success))
                per_step_truncated.append(bool(truncated))
                action_dim_seen = env_action.shape[0]

            image = self._image(task, env, obs)
            observation = Observation(image=image, instruction=instruction, extra={"obs": obs})
            if capture_artifacts and image is not None:
                frames.append(image)

            if success:
                break

        elapsed = time.monotonic() - start

        video_path: str | None = None
        if capture_artifacts and (frames or per_step_actions):
            assert self._rollouts_dir is not None
            metadata = RolloutMetadata(
                instruction=instruction,
                cell_axes=dict(cell.axes),
                cell_slug=cell.slug,
                seed=seed,
                embodiment=cell.embodiment,
                task_name=cell.task,
                model_name=policy.name,
                model_checkpoint_id=policy.checkpoint_id,
                success=bool(success),
                num_steps=step_idx,
                elapsed_wall_seconds=elapsed,
                action_dim=action_dim_seen,
                action_components=joint_action_components(action_dim_seen),
                episode_stats=dict(info.get("episode_stats", {})) if isinstance(info, dict) else {},
            )
            traj = Trajectory.from_steps(per_step_actions, per_step_rewards, per_step_success, per_step_truncated)
            target = write_rollout_artifacts(self._rollouts_dir, metadata, traj, frames, video_fps=self._video_fps)
            mp4 = target / "rollout.mp4"
            if mp4.exists():
                video_path = str(mp4.relative_to(self._rollouts_dir))

        return RolloutResult(
            cell=cell,
            seed=seed,
            success=bool(success),
            num_steps=step_idx,
            elapsed_wall_seconds=elapsed,
            episode_stats=dict(info.get("episode_stats", {})) if isinstance(info, dict) else {},
            video_path=video_path,
        )

    @staticmethod
    def _flatten_action(action: ActionLike) -> NDArrayF32:
        # Legged/aerial Policies should emit JointAction. Allow the EEF
        # case as a fallback (e.g. for a Mock policy that doesn't know
        # the embodiment).
        if isinstance(action, JointAction):
            return action.vector.astype(np.float32)
        if isinstance(action, Action):
            return np.concatenate(
                [action.world_vector, action.rot_axangle, action.gripper]
            ).astype(np.float32)
        raise TypeError(f"unknown action type: {type(action).__name__}")

    @staticmethod
    def _instruction(task: Task, env: Any) -> str:
        hook = getattr(task, "instruction_for", None)
        if callable(hook):
            try:
                return str(hook(env)) or ""
            except Exception:
                return ""
        return ""

    @staticmethod
    def _image(task: Task, env: Any, obs: Any) -> NDArrayU8:
        hook = getattr(task, "extract_image", None)
        if callable(hook):
            try:
                return np.asarray(hook(env, obs), dtype=np.uint8)
            except Exception:
                pass
        # Default: try env.render() (MJX standard).
        render = getattr(env, "render", None)
        if callable(render):
            try:
                arr = np.asarray(render(), dtype=np.uint8)
                if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                    return arr
            except Exception:
                pass
        # Last resort: return a 1x1 black frame so downstream code doesn't
        # crash. Legged sims without rendering still produce CSV + manifest.
        return np.zeros((1, 1, 3), dtype=np.uint8)

