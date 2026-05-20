"""Per-rollout data storage.

**In plain words.** Every individual rollout (one model, one
condition, one seed) writes a small self-contained folder to disk:
the video, the per-step actions and rewards, and a metadata file
describing the exact conditions. This is what lets a reviewer pull
up a single failure case in isolation — share one folder, anybody
can replay the video and inspect the trajectory without needing the
whole sweep.


Each rollout becomes a self-contained directory next to the sealed
manifest:

    rollouts/<cell-slug>/seed_NNNN/
      rollout.mp4         # the video (h.264, browser-playable)
      trajectory.npz      # per-step action / reward / done / success
      metadata.json       # exact instruction, axes, model, env config, etc.

Why this shape:
- One-dir-per-rollout means you can tar-and-send a single failure case
  to anyone without grep'ing across files.
- `trials.csv` stays as a thin index for backwards compat with the
  v0 analysis pipeline.
- `trajectory.npz` is numpy-loadable with no extra deps (no pandas /
  pyarrow). Reviewers can `np.load` it and have arrays.
- `metadata.json` is human-readable so a reviewer can `cat` a single
  rollout and see exactly what was asked of the policy.

The Adapters call `write_rollout_artifacts` after each rollout. The
notebook (and `analysis.Rollout.load`) reads them back.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

NDArrayF32 = npt.NDArray[np.float32]
NDArrayU8 = npt.NDArray[np.uint8]

_SLUG_SANITIZE = re.compile(r"[^A-Za-z0-9._=,-]+")


def safe_cell_dirname(embodiment: str, task: str, axes: dict[str, str]) -> str:
    """Deterministic short directory name for one cell. Same scheme as
    the original video path (mirrored here for backwards-compat)."""
    axis_part = ",".join(f"{k}={v}" for k, v in sorted(axes.items()))
    raw = f"{embodiment}__{task}__{axis_part}"
    return _SLUG_SANITIZE.sub("_", raw)


def rollout_dir(rollouts_root: Path, embodiment: str, task: str, axes: dict[str, str], seed: int) -> Path:
    """`<rollouts_root>/<cell-slug>/seed_NNNN/`."""
    return rollouts_root / safe_cell_dirname(embodiment, task, axes) / f"seed_{seed:04d}"


@dataclass(frozen=True)
class Trajectory:
    """Per-step rollout record. All arrays have length `num_steps`.

    `action` has shape (num_steps, action_dim). The action_dim is
    embodiment-specific: 7 for SimplerEnv arms (Action.world_vector +
    rot_axangle + gripper), 12 for Go1 (joint targets). The
    `action_components` field in `RolloutMetadata` names each column.
    """

    t: npt.NDArray[np.int32]                   # (T,) step indices 0..T-1
    action: NDArrayF32                         # (T, action_dim)
    reward: NDArrayF32                         # (T,)
    success: npt.NDArray[np.bool_]             # (T,) success-flag-after-step
    truncated: npt.NDArray[np.bool_]           # (T,) truncated-after-step

    @classmethod
    def from_steps(
        cls,
        actions: list[npt.NDArray[Any]],
        rewards: list[float],
        successes: list[bool],
        truncated_flags: list[bool],
    ) -> Trajectory:
        n = len(actions)
        if n == 0:
            # Empty rollout (policy.terminate at step 0). Still write a
            # valid (empty) trajectory so the reader doesn't crash.
            return cls(
                t=np.zeros((0,), dtype=np.int32),
                action=np.zeros((0, 0), dtype=np.float32),
                reward=np.zeros((0,), dtype=np.float32),
                success=np.zeros((0,), dtype=np.bool_),
                truncated=np.zeros((0,), dtype=np.bool_),
            )
        action_arr = np.stack([np.asarray(a, dtype=np.float32) for a in actions], axis=0)
        return cls(
            t=np.arange(n, dtype=np.int32),
            action=action_arr,
            reward=np.asarray(rewards, dtype=np.float32),
            success=np.asarray(successes, dtype=np.bool_),
            truncated=np.asarray(truncated_flags, dtype=np.bool_),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            t=self.t,
            action=self.action,
            reward=self.reward,
            success=self.success,
            truncated=self.truncated,
        )

    @classmethod
    def load(cls, path: Path) -> Trajectory:
        with np.load(path) as data:
            return cls(
                t=data["t"],
                action=data["action"],
                reward=data["reward"],
                success=data["success"],
                truncated=data["truncated"],
            )


@dataclass(frozen=True)
class RolloutMetadata:
    """Everything about a single rollout that isn't pixels or step arrays.

    Serializes to JSON. Self-contained — a reviewer reading just this
    file knows exactly what was asked of the policy, on what cell, with
    what seed, on what hardware, and what the policy answered with.
    """

    # What was asked
    instruction: str                       # the exact string passed to policy.reset()
    cell_axes: dict[str, str]              # axis → level mapping for this cell
    cell_slug: str                         # deterministic short id
    seed: int

    # Identity
    embodiment: str
    task_name: str
    model_name: str
    model_checkpoint_id: str

    # Outcome
    success: bool
    num_steps: int
    elapsed_wall_seconds: float

    # Spec
    action_dim: int
    action_components: list[str]           # names of each action column ("world_vector_x", etc.)
    episode_stats: dict[str, Any]          # whatever the env handed back at end-of-episode

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_coerce_json_safe(self._asdict()), indent=2, sort_keys=True))

    def _asdict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "cell_axes": dict(self.cell_axes),
            "cell_slug": self.cell_slug,
            "seed": self.seed,
            "embodiment": self.embodiment,
            "task_name": self.task_name,
            "model_name": self.model_name,
            "model_checkpoint_id": self.model_checkpoint_id,
            "success": self.success,
            "num_steps": self.num_steps,
            "elapsed_wall_seconds": self.elapsed_wall_seconds,
            "action_dim": self.action_dim,
            "action_components": list(self.action_components),
            "episode_stats": dict(self.episode_stats),
        }

    @classmethod
    def load(cls, path: Path) -> RolloutMetadata:
        d = json.loads(path.read_text())
        return cls(
            instruction=d["instruction"],
            cell_axes=dict(d["cell_axes"]),
            cell_slug=d["cell_slug"],
            seed=int(d["seed"]),
            embodiment=d["embodiment"],
            task_name=d["task_name"],
            model_name=d["model_name"],
            model_checkpoint_id=d["model_checkpoint_id"],
            success=bool(d["success"]),
            num_steps=int(d["num_steps"]),
            elapsed_wall_seconds=float(d["elapsed_wall_seconds"]),
            action_dim=int(d["action_dim"]),
            action_components=list(d["action_components"]),
            episode_stats=dict(d.get("episode_stats", {})),
        )


def write_rollout_artifacts(
    rollouts_root: Path,
    metadata: RolloutMetadata,
    trajectory: Trajectory,
    frames: list[NDArrayU8] | None,
    video_fps: int,
) -> Path:
    """Write `rollout.mp4` + `trajectory.npz` + `metadata.json` into the
    per-rollout directory and return that directory.

    `frames` is the optional list of RGB uint8 frames the Adapter collected
    during the rollout. When None or empty, the mp4 is skipped (used in
    contract tests where rendering is off)."""
    target = rollout_dir(rollouts_root, metadata.embodiment, metadata.task_name, metadata.cell_axes, metadata.seed)
    target.mkdir(parents=True, exist_ok=True)
    metadata.save(target / "metadata.json")
    trajectory.save(target / "trajectory.npz")
    if frames:
        _encode_h264(target / "rollout.mp4", frames, fps=video_fps)
    return target


def _encode_h264(path: Path, frames: list[NDArrayU8], *, fps: int) -> None:
    import mediapy  # lazy: not in .venv-ci's substrate-only deps

    stack: list[NDArrayU8] = []
    target_shape: tuple[int, ...] | None = None
    for f in frames:
        arr = np.ascontiguousarray(np.asarray(f, dtype=np.uint8))
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        if target_shape is None:
            target_shape = arr.shape
        elif arr.shape != target_shape:
            continue
        stack.append(arr)
    if not stack:
        return
    mediapy.write_video(str(path), stack, fps=fps, codec="h264")


# ---- Action-component naming (per-adapter) ------------------------------


def _coerce_json_safe(obj: Any) -> Any:
    """Recursively coerce numpy scalars/arrays into native Python types.

    `info["episode_stats"]` blobs returned from envs sometimes contain
    `np.bool_`, `np.int64`, `np.float32`, or 0-d / 1-d ndarrays — none
    of which `json.dumps` will serialize. This helper walks the dict /
    list / tuple structure and converts to Python natives so the
    metadata.json save path is robust against whatever the env hands us.
    """
    if isinstance(obj, np.ndarray):
        return _coerce_json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _coerce_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_coerce_json_safe(v) for v in obj]
    return obj


def gym_action_components() -> list[str]:
    """7-DoF EEF: SimplerEnv / LIBERO / RoboCasa / Bridge convention."""
    return [
        "world_vector_x", "world_vector_y", "world_vector_z",
        "rot_axangle_x", "rot_axangle_y", "rot_axangle_z",
        "gripper",
    ]


def joint_action_components(action_dim: int) -> list[str]:
    """N-DoF joint-space (Go1, etc). Generic 'joint_i' labels."""
    return [f"joint_{i}" for i in range(action_dim)]
