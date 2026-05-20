"""LIBERO Task implementation — proof of architecture sim-agnosticism.

**In plain words.** LIBERO is a completely different simulator from
SimplerEnv. This file wires up three LIBERO scenes as tasks so the
suite can sweep them with the exact same pipeline it uses for the
SimplerEnv ones. Its existence is the proof that the suite isn't
hidden-coupled to any particular simulator — adding a new sim is
roughly 80 lines of code.


The point of this file is to demonstrate that the *same* `GymAdapter`,
`Manifest`, `statistics`, `analysis`, and notebook pipeline absorbs a
different simulator backend with only a new Task class. ~80 LOC, no
changes outside this file required.

`LIBEROSpatial` exposes the first three tasks from the LIBERO-Spatial
suite as cells (the "axis" is task_id since LIBERO doesn't ship a
Variant Aggregation grid). The Task implements the two optional hooks
the GymAdapter looks for:

- `instruction_for(env)` — LIBERO uses `env.language_instruction` (attribute,
  not method); the default lookup would silently return "".
- `extract_image(env, obs)` — LIBERO obs is `{"agentview_image": ndarray, ...}`
  and the image arrives upside-down (robosuite convention); flip vertically
  so policies see what the workspace looks like to a human.

Robosuite uses the older 4-tuple gym signature `(obs, reward, done, info)`;
`_LIBEROGymCompatEnv` adapts to the gymnasium 5-tuple the Adapter expects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._types import CellId, NDArrayU8


@dataclass(frozen=True)
class _LiberoCellSpec:
    task_id: int
    suite: str
    task_name_short: str
    bddl_filename: str
    language: str


class _LIBEROGymCompatEnv:
    """Adapts robosuite's 4-tuple step() to gymnasium's 5-tuple.

    Also exposes `language_instruction` as an attribute on the wrapper so
    the GymAdapter's `instruction_for` hook can find it via the Task. The
    underlying ControlEnv is at `.env_inner`.
    """

    def __init__(self, env_inner: Any, language_instruction: str, horizon: int) -> None:
        self.env_inner = env_inner
        self.language_instruction = language_instruction
        self._horizon = horizon
        self._step_count = 0

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        obs = self.env_inner.reset()
        self._step_count = 0
        return obs, {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, done, info = self.env_inner.step(action)
        self._step_count += 1
        # In LIBERO/robosuite, `done` becomes True at horizon OR on success.
        # The robosuite check_success() is the authoritative success signal.
        success = False
        check_success = getattr(self.env_inner.env, "_check_success", None)
        if callable(check_success):
            try:
                success = bool(check_success())
            except Exception:
                success = False
        truncated = bool(done and not success) or self._step_count >= self._horizon
        info = dict(info) if isinstance(info, dict) else {}
        info["episode_stats"] = {"check_success": success, "raw_done": bool(done)}
        return obs, float(reward), success, truncated, info

    def close(self) -> None:
        close = getattr(self.env_inner, "close", None)
        if callable(close):
            close()


def _libero_spatial_cells(n: int = 3) -> list[_LiberoCellSpec]:
    """First `n` LIBERO-Spatial tasks as cells. Lazy-imports libero."""
    from libero.libero import benchmark
    suite_obj = benchmark.get_benchmark_dict()["libero_spatial"]()
    out: list[_LiberoCellSpec] = []
    for i in range(min(n, len(suite_obj.tasks))):
        t = suite_obj.get_task(i)
        out.append(_LiberoCellSpec(
            task_id=i,
            suite="libero_spatial",
            task_name_short=f"spatial_{i}",
            bddl_filename=t.bddl_file,
            language=t.language,
        ))
    return out


class LIBEROSpatial:
    """LIBERO-Spatial-N: pick-and-place across N spatial reasoning tasks.

    Each cell is one task. Axes: `task_id` (the only variant axis since
    LIBERO doesn't ship lighting/background/etc. variants out of the box).
    Adding LIBERO variant axes is additive: edit `_libero_spatial_cells`.
    """

    def __init__(self, *, n_tasks: int = 3) -> None:
        self._specs = _libero_spatial_cells(n_tasks)

    @property
    def name(self) -> str:
        return "libero_spatial"

    @property
    def embodiment(self) -> str:
        return "panda"  # robosuite default — could be widowx, ur5e, etc.

    @property
    def n_cells(self) -> int:
        return len(self._specs)

    def cell_id(self, cell: int) -> CellId:
        spec = self._specs[cell]
        return CellId(
            embodiment=self.embodiment,
            task=self.name,
            axes={"task_id": spec.task_name_short, "suite": spec.suite},
        )

    def build_env(self, cell: int) -> Any:
        # Import locally so eval_suite can be imported without libero installed
        # (CI doesn't have libero; this Task is only constructed in the libero
        # venv).
        import os

        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        spec = self._specs[cell]
        bddl_root = get_libero_path("bddl_files")
        bddl_path = os.path.join(bddl_root, spec.suite, spec.bddl_filename)
        env_inner = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=128,
            camera_widths=128,
        )
        return _LIBEROGymCompatEnv(env_inner, spec.language, horizon=self.max_episode_steps)

    @property
    def max_episode_steps(self) -> int:
        return 100  # robosuite default for these tasks is much higher; tight cap for v0 PoC

    # ---- GymAdapter optional hooks --------------------------------------

    def instruction_for(self, env: Any) -> str:
        # Wrapper exposes the attribute; defensive .get on a real env too.
        return getattr(env, "language_instruction", "") or ""

    def extract_image(self, env: Any, obs: Any) -> NDArrayU8:
        # Robosuite returns agentview_image upside-down (camera-frame convention).
        # Flip vertically so policies see the workspace as a human would.
        if isinstance(obs, dict):
            for key in ("agentview_image", "frontview_image", "robot0_eye_in_hand_image"):
                if key in obs:
                    img = np.asarray(obs[key], dtype=np.uint8)
                    return np.ascontiguousarray(img[::-1, :, :])
        raise ValueError("no LIBERO image in observation")
