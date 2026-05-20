"""GymAdapter — drives a Policy against any gymnasium-shaped env.

**In plain words.** This is the workhorse that handles every
arm-style simulator. It takes a robot model + a task + a seed,
resets the simulator, asks the model for an action, steps the
simulator forward, repeats until the episode ends, and reports the
result. Same code drives both SimplerEnv (Google Robot, WidowX) and
LIBERO — that's the in-tree proof that the suite isn't bound to one
sim.


The same Adapter works for SimplerEnv and LIBERO (both gym-shaped, both
7-DoF EEF action). Sim-specific quirks live in the `Task` via two
optional hooks:

- `Task.instruction_for(env) -> str` — pull the language instruction.
  SimplerEnv: `env.get_language_instruction()`. LIBERO: `env.language_instruction`.
  Defaults: try `env.get_language_instruction()`, return "" if absent.
- `Task.extract_image(env, obs) -> NDArrayU8` — pull the primary RGB.
  Defaults: import SimplerEnv's helper if available, else walk the obs
  dict for any (H, W, 3|4) uint8 array.

A different action shape (legged / aerial) needs a sibling Adapter; the
shared 7-DoF EEF assumption is what makes SimplerEnv↔LIBERO trivial.

**v0 storage upgrade:** when `videos_dir` is set, the Adapter writes
a per-rollout directory containing `rollout.mp4` + `trajectory.npz` +
`metadata.json` (see `eval_suite/rollout_data.py`).
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
    gym_action_components,
    write_rollout_artifacts,
)

_VIDEO_FPS = 5


class GymAdapter:
    """Drives one (cell, seed) rollout through a gymnasium-shaped env.

    Pass `videos_dir` to enable per-rollout artifact capture: an mp4
    video plus a trajectory.npz plus a metadata.json. Set to None to
    skip (default; CI keeps it None so the mock loop stays fast).
    """

    def __init__(
        self,
        *,
        max_steps_override: int | None = None,
        videos_dir: str | Path | None = None,
    ) -> None:
        self._max_steps_override = max_steps_override
        self._rollouts_dir: Path | None = Path(videos_dir) if videos_dir is not None else None

    @property
    def name(self) -> str:
        return "gym-adapter"

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
                    pass  # best-effort; env.close() can be flaky across resets

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

        reset_kwargs: dict[str, Any] = {}
        try:
            obs, _info = env.reset(seed=seed)
        except TypeError:
            obs, _info = env.reset(**reset_kwargs)

        instruction = self._instruction(task, env)
        policy.reset(instruction)

        image = self._image(task, env, obs)
        observation = Observation(image=image, instruction=instruction, extra={"obs": obs})
        if capture_artifacts:
            frames.append(image)

        success = False
        truncated = False
        info: dict[str, Any] = {}
        step_idx = 0
        is_final_subtask = self._is_final_subtask(env)

        while step_idx < max_steps and not truncated:
            action = policy.step(observation)
            if action.terminate:
                if not is_final_subtask:
                    self._advance_to_next_subtask(env)
                    is_final_subtask = self._is_final_subtask(env)
                else:
                    break

            env_action = self._flatten_action(action)
            obs, reward, success, truncated, info = env.step(env_action)
            step_idx += 1

            if capture_artifacts:
                per_step_actions.append(env_action)
                per_step_rewards.append(float(reward) if reward is not None else 0.0)
                per_step_success.append(bool(success))
                per_step_truncated.append(bool(truncated))

            new_instruction = self._instruction(task, env)
            if new_instruction and new_instruction != instruction:
                instruction = new_instruction
                policy.reset(instruction)

            image = self._image(task, env, obs)
            observation = Observation(image=image, instruction=instruction, extra={"obs": obs})
            if capture_artifacts:
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
                action_dim=7,
                action_components=gym_action_components(),
                episode_stats=dict(info.get("episode_stats", {})) if isinstance(info, dict) else {},
            )
            traj = Trajectory.from_steps(per_step_actions, per_step_rewards, per_step_success, per_step_truncated)
            target = write_rollout_artifacts(self._rollouts_dir, metadata, traj, frames, video_fps=_VIDEO_FPS)
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

    # ---- hookable extractors -------------------------------------------

    @staticmethod
    def _instruction(task: Task, env: Any) -> str:
        """Use `task.instruction_for(env)` if defined; else fall back."""
        hook = getattr(task, "instruction_for", None)
        if callable(hook):
            try:
                return str(hook(env)) or ""
            except Exception:
                return ""
        return GymAdapter._default_instruction(env)

    @staticmethod
    def _default_instruction(env: Any) -> str:
        # SimplerEnv envs expose this method.
        getter = getattr(env, "get_language_instruction", None)
        if callable(getter):
            try:
                return str(getter()) or ""
            except Exception:
                return ""
        return ""

    @staticmethod
    def _image(task: Task, env: Any, obs: Any) -> NDArrayU8:
        """Use `task.extract_image(env, obs)` if defined; else fall back."""
        hook = getattr(task, "extract_image", None)
        if callable(hook):
            return np.asarray(hook(env, obs), dtype=np.uint8)
        return GymAdapter._default_extract_image(env, obs)

    @staticmethod
    def _default_extract_image(env: Any, obs: Any) -> NDArrayU8:
        """SimplerEnv's helper first; else walk the obs dict."""
        try:
            from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
            return np.asarray(get_image_from_maniskill2_obs_dict(env, obs), dtype=np.uint8)
        except Exception:
            return GymAdapter._walk_for_image(obs)

    @staticmethod
    def _walk_for_image(obj: Any) -> NDArrayU8:
        if isinstance(obj, np.ndarray) and obj.ndim == 3 and obj.shape[-1] in (3, 4):
            return obj.astype(np.uint8)
        if isinstance(obj, dict):
            for v in obj.values():
                try:
                    return GymAdapter._walk_for_image(v)
                except ValueError:
                    continue
        raise ValueError("no RGB image in observation")

    # ---- SimplerEnv-specific subtask handling (no-op elsewhere) --------

    @staticmethod
    def _is_final_subtask(env: Any) -> bool:
        getter = getattr(env, "is_final_subtask", None)
        if callable(getter):
            try:
                return bool(getter())
            except Exception:
                return True
        return True

    @staticmethod
    def _advance_to_next_subtask(env: Any) -> None:
        advance = getattr(env, "advance_to_next_subtask", None)
        if callable(advance):
            try:
                advance()
            except Exception:
                pass

    # ---- action flattening + video encoding ----------------------------

    @staticmethod
    def _flatten_action(action: ActionLike) -> NDArrayF32:
        # 7-DoF EEF convention used by SimplerEnv, LIBERO, RoboCasa, Bridge.
        # JointAction passes its arbitrary-length vector straight through
        # (used by MujocoPlaygroundAdapter for legged etc).
        if isinstance(action, JointAction):
            return action.vector.astype(np.float32)
        if isinstance(action, Action):
            return np.concatenate(
                [action.world_vector.astype(np.float32),
                 action.rot_axangle.astype(np.float32),
                 action.gripper.astype(np.float32)]
            ).astype(np.float32)
        raise TypeError(f"unknown action type: {type(action).__name__}")
