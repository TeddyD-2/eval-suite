"""OXEReplayPolicy — replays a recorded episode as if it were a policy.

**In plain words.** Pretend a recorded real-world demonstration is a
"model" — just feed the recorded actions back through the suite as
if a live model had produced them. Two things this unlocks: it proves
the suite can ingest data in the Open-X-Embodiment (OXE) and LeRobot
dataset formats, and it gives you the sim-side rollout you need to
pair against the real recording for trajectory-level (MMRV)
calibration. Without this, you couldn't compare "what happened in
sim" against "what happened in the lab" frame-by-frame.


The point isn't to "evaluate" the dataset — its actions are by
construction the actions that produced the recorded trajectory. The
point is twofold:

1. **Interop proof.** Confirm the suite ingests an OXE/RLDS or LeRobot
   episode through the same `Policy` Protocol every other model uses —
   no parallel dataset-replay code path in the sweep loop.
2. **Sim-to-real calibration data.** Pair the recorded real-world
   action sequence against a sim rollout (Phase 3): replay the same
   actions in sim, record the resulting trajectory, compute MMRV
   between the recorded real trajectory and the sim one. Without an
   open-loop replay path, you can't generate paired sim-real
   trajectories for trajectory-level calibration.

Supports two source formats:

- **OXE/RLDS** via `tensorflow_datasets.load(dataset_id)`. Episodes
  carry a `steps` substream; `steps['action']` is the recorded action
  sequence.
- **LeRobot HF datasets** via `datasets.load_dataset(dataset_id)`.
  Episodes are flat with an `action` column.

Both are behind the `[oxe]` optional extra so the default install
stays light.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._types import Action, ActionLike, JointAction, Observation


class OXEReplayPolicy:
    """Open-loop replay of a recorded OXE / LeRobot episode."""

    def __init__(
        self,
        *,
        dataset_id: str,
        episode_id: int,
        format: str = "oxe",
        split: str = "train",
    ) -> None:
        self._dataset_id = dataset_id
        self._episode_id = int(episode_id)
        self._format = format
        self._split = split
        self._actions: np.ndarray[Any, Any] = _load_episode_actions(
            dataset_id=dataset_id, episode_id=self._episode_id, fmt=format, split=split,
        )
        self._step_idx = 0
        self._instruction = ""

    def reset(self, instruction: str) -> None:
        self._instruction = instruction
        self._step_idx = 0

    def step(self, observation: Observation) -> ActionLike:
        # Replay is open-loop: ignore the observation and emit the
        # next recorded action. When we run past the recorded length,
        # signal terminate so the rollout loop ends cleanly.
        if self._step_idx >= len(self._actions):
            return _terminate_action(self._actions.shape[1])
        action_vec = self._actions[self._step_idx].astype(np.float32)
        self._step_idx += 1
        is_last = self._step_idx >= len(self._actions)
        return _action_from_vector(action_vec, terminate=is_last)

    @property
    def name(self) -> str:
        return f"oxe-replay:{self._dataset_id}#{self._episode_id}"

    @property
    def checkpoint_id(self) -> str:
        # Re-using checkpoint_id as the "what produced these actions"
        # provenance pointer; for a dataset replay that's the dataset id
        # + episode index. Stable across reruns.
        return f"oxe:{self._dataset_id}#{self._episode_id}:{self._format}"

    @property
    def family(self) -> str:
        return "oxe_replay"


def _load_episode_actions(
    *,
    dataset_id: str,
    episode_id: int,
    fmt: str,
    split: str,
) -> np.ndarray[Any, Any]:
    if fmt == "oxe":
        try:
            import tensorflow_datasets as tfds  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "OXEReplayPolicy(format='oxe') needs the `[oxe]` extra: "
                "`pip install 'eval-suite-stdlib[oxe]'`."
            ) from e
        builder = tfds.builder(dataset_id)
        ds = builder.as_dataset(split=split)
        for i, episode in enumerate(ds):
            if i != episode_id:
                continue
            steps = list(episode["steps"].as_numpy_iterator())
            out: np.ndarray[Any, Any] = np.stack([s["action"] for s in steps]).astype(np.float32)
            return out
        raise IndexError(f"OXE dataset {dataset_id!r} has no episode {episode_id}")
    if fmt == "lerobot":
        try:
            from datasets import load_dataset  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "OXEReplayPolicy(format='lerobot') needs `datasets` from the "
                "`[lerobot]` extra: `pip install 'eval-suite-stdlib[lerobot]'`."
            ) from e
        ds = load_dataset(dataset_id, split=split)
        # LeRobot HF datasets carry `episode_index` per row; filter to one.
        rows = [r for r in ds if int(r.get("episode_index", -1)) == episode_id]
        if not rows:
            raise IndexError(
                f"LeRobot dataset {dataset_id!r} has no episode_index={episode_id}"
            )
        return np.stack([np.asarray(r["action"], dtype=np.float32) for r in rows])
    raise ValueError(f"unknown OXEReplayPolicy format: {fmt!r}")


def _action_from_vector(vec: np.ndarray[Any, Any], *, terminate: bool) -> ActionLike:
    if vec.shape[0] == 7:
        return Action(
            world_vector=vec[:3].astype(np.float32).copy(),
            rot_axangle=vec[3:6].astype(np.float32).copy(),
            gripper=vec[6:7].astype(np.float32).copy(),
            terminate=terminate,
        )
    return JointAction(vector=vec.astype(np.float32).copy(), terminate=terminate)


def _terminate_action(action_dim: int) -> ActionLike:
    if action_dim == 7:
        return Action(
            world_vector=np.zeros(3, dtype=np.float32),
            rot_axangle=np.zeros(3, dtype=np.float32),
            gripper=np.zeros(1, dtype=np.float32),
            terminate=True,
        )
    return JointAction(
        vector=np.zeros(action_dim, dtype=np.float32),
        terminate=True,
    )
