"""Shared value types used across the eval-suite contracts.

These are plain dataclasses (no behavior) so they're safe to import without
pulling in SimplerEnv / TF / JAX. The Protocols in `contracts.py` reference
these types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, get_args

import numpy as np
import numpy.typing as npt

NDArrayF32 = npt.NDArray[np.float32]
NDArrayU8 = npt.NDArray[np.uint8]


# ---------------------------------------------------------------------------
# Canonical generalization-axis taxonomy.
#
# The takehome prompt names four dimensions a generalist robot model is
# tested against: shifts in language, visuals, physics, and embodiment. Each
# Task in this suite optionally declares which of its cell-level axes maps
# to which canonical dim via a `canonical_axis_map: dict[str, CanonicalDim]`
# attribute (duck-typed; not in the Protocol). The analysis layer reads it
# at notebook-render time to compute the canonical generalization profile.
#
# The mapping is per-(Task, axis): the same axis name can legitimately mean
# different things in different tasks. E.g. `task_id` on a manipulation task
# might be a language shift; `task_family` on Go1 is a command-vector
# language shift; on a different sim it could be physics.
#
# Cross-task aggregation of canonical-dim scores is v1 work — pooling
# across tasks requires either deployment-relevance weighting or a
# stratified bootstrap. v0 publishes per-(model, task) canonical
# profiles only.
# ---------------------------------------------------------------------------

CanonicalDim = Literal["language", "visuals", "physics", "embodiment"]


def canonical_dims() -> tuple[CanonicalDim, ...]:
    """Tuple of the four canonical dims in their canonical order."""
    return get_args(CanonicalDim)


@dataclass(frozen=True)
class Observation:
    """A single environment observation as the Policy sees it.

    `image` is the primary input; `extra` carries everything else the env
    exposed (proprio, robot state, second camera) so policies that want more
    than RGB don't have to fight the interface.
    """

    image: NDArrayU8  # (H, W, 3) uint8 RGB
    instruction: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    """7-DoF end-effector action — the manipulation case.

    The (world_vector, rot_axangle, gripper) decomposition matches what
    SimplerEnv's reference policies emit and what SimplerEnv / LIBERO /
    RoboCasa / Bridge envs consume. The Adapter flattens this to the
    7-d action vector the env expects.

    `terminate` is the policy's request to end the episode early (RT-1 and
    Octo both have a terminate-episode head). The Adapter decides whether
    to honor it or, for multi-subtask envs, advance to the next subtask.

    For non-manipulation embodiments (legged, aerial, multi-arm), use
    `JointAction` instead — it carries an arbitrary-length joint vector.
    """

    world_vector: NDArrayF32  # (3,) float32 — delta xyz
    rot_axangle: NDArrayF32  # (3,) float32 — rotation axis-angle
    gripper: NDArrayF32  # (1,) float32 — gripper command
    terminate: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JointAction:
    """Arbitrary-length joint-space action vector.

    For embodiments where the 7-DoF EEF Action doesn't apply: legged
    (Unitree Go1 is 12-DoF joint targets), aerial (4-DoF rotor thrusts),
    multi-arm (14-DoF dual-arm EEF), etc.

    The Adapter passes `vector` straight through to env.step(). The Task
    is responsible for choosing a Policy whose action_dim matches the
    env's action_space.
    """

    vector: NDArrayF32  # (N,) float32 — joint targets / torques / forces
    terminate: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


# Union type for anything a Policy can emit. Adapter switches on isinstance.
ActionLike = Action | JointAction


@dataclass(frozen=True)
class CellId:
    """Identifies a single variant cell on the Variant Aggregation grid.

    The `axes` mapping is the source of truth for what makes this cell
    unique. `slug` is a deterministic short string derived from the axes,
    used in filenames and manifest cell IDs.
    """

    embodiment: str
    task: str
    axes: dict[str, str]

    @property
    def slug(self) -> str:
        axis_part = ",".join(f"{k}={v}" for k, v in sorted(self.axes.items()))
        return f"{self.embodiment}/{self.task}/{axis_part}"


@dataclass(frozen=True)
class RolloutResult:
    """Outcome of a single rollout (one cell, one seed)."""

    cell: CellId
    seed: int
    success: bool
    num_steps: int
    elapsed_wall_seconds: float
    episode_stats: dict[str, Any] = field(default_factory=dict)
    video_path: str | None = None  # relative path to the mp4 if videos enabled


@dataclass(frozen=True)
class CellResult:
    """Aggregated results for a single cell across N seeds."""

    cell: CellId
    n_trials: int
    successes: int
    wilson_ci_low: float
    wilson_ci_high: float
    per_seed_success: list[bool] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successes / self.n_trials if self.n_trials else 0.0
