"""MockTask — synthetic env that the CI contract test runs against.

Returns a tiny gymnasium-like env that the Adapter can drive without any
real simulator. The env counts steps and reports `success=False` after a
fixed horizon. Sufficient to verify: Adapter steps, Action flattening
works, RolloutResult is produced, Manifest contains the cell.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .._types import CellId


class _MockEnv:
    """Minimal gymnasium-shaped env used by MockTask. Internal."""

    def __init__(self, horizon: int = 5) -> None:
        self._horizon = horizon
        self._step = 0
        self._instruction = "mock instruction"

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self._step = 0
        if seed is not None:
            np.random.default_rng(seed)
        obs = self._observation()
        return obs, {}

    def step(self, action: npt.NDArray[Any]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._step += 1
        truncated = self._step >= self._horizon
        info = {"elapsed_steps": self._step, "success": False, "episode_stats": {"mock_action_mean": float(np.mean(action))}}
        return self._observation(), 0.0, False, truncated, info

    def _observation(self) -> dict[str, Any]:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        return {"image": {"camera": {"rgb": image}}, "extra": {}}

    def get_language_instruction(self) -> str:
        return self._instruction

    def is_final_subtask(self) -> bool:
        return True

    def advance_to_next_subtask(self) -> None:
        return None


class MockTask:
    """A trivial Task with a configurable number of cells.

    Each cell has axes {axis_a: levelN, axis_b: leveln_mod_2}. Useful for
    exercising the per-axis aggregation logic in tests without standing
    up the full SimplerEnv cell catalog.
    """

    def __init__(self, *, n_cells: int = 3, max_episode_steps: int = 5) -> None:
        self._n_cells = n_cells
        self._max_episode_steps = max_episode_steps

    @property
    def name(self) -> str:
        return "mock-task"

    @property
    def embodiment(self) -> str:
        return "mock"

    @property
    def n_cells(self) -> int:
        return self._n_cells

    def cell_id(self, cell: int) -> CellId:
        if not (0 <= cell < self._n_cells):
            raise IndexError(cell)
        return CellId(
            embodiment="mock",
            task="mock-task",
            axes={"axis_a": f"level{cell}", "axis_b": f"level{cell % 2}"},
        )

    def build_env(self, cell: int) -> Any:
        if not (0 <= cell < self._n_cells):
            raise IndexError(cell)
        return _MockEnv(horizon=self._max_episode_steps)

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps
