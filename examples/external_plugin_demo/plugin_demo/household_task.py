"""Household task — a reference external Task plugin.

Wraps eval-suite's `_MockEnv` (no real simulation) but with house-themed
cell labels so the v0 canonical-axis-taxonomy chart and the per-condition
view show what a real external task family would look like.

Three cells:
    cell 0: room=kitchen, object=cup,    instruction="pick up the cup"
    cell 1: room=pantry,  object=can,    instruction="grab the can"
    cell 2: room=living_room, object=remote, instruction="lift the remote"

The Task declares `canonical_axis_map` mapping `room` to visuals and
`object` to language. After running this Task, the eval-suite notebook
renders a four-bar canonical profile for the (Policy, HouseholdMockTask)
pair just like it does for the in-tree Tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eval_suite._types import CanonicalDim, CellId
from eval_suite.tasks.mock import _MockEnv


@dataclass(frozen=True)
class _HouseholdCellSpec:
    room: str
    object_label: str
    instruction: str


_HOUSEHOLD_CELLS: list[_HouseholdCellSpec] = [
    _HouseholdCellSpec(room="kitchen",     object_label="cup",    instruction="pick up the cup"),
    _HouseholdCellSpec(room="pantry",      object_label="can",    instruction="grab the can"),
    _HouseholdCellSpec(room="living_room", object_label="remote", instruction="lift the remote"),
]


class HouseholdMockTask:
    """A house-themed Task that an external plugin would publish.

    Implementation note: the underlying env is eval-suite's `_MockEnv`
    (no real simulation). What's external about this Task is the *labels*
    and the *cell catalog*. A real third-party plugin would build a real
    env in its `build_env` and the rest of the contract would stay the
    same.
    """

    canonical_axis_map: dict[str, CanonicalDim] = {
        "room":   "visuals",
        "object": "language",
    }

    def __init__(self, *, max_episode_steps: int = 5) -> None:
        self._max_episode_steps = max_episode_steps

    @property
    def name(self) -> str:
        return "household_mock"

    @property
    def embodiment(self) -> str:
        return "household_arm"

    @property
    def n_cells(self) -> int:
        return len(_HOUSEHOLD_CELLS)

    def cell_id(self, cell: int) -> CellId:
        spec = _HOUSEHOLD_CELLS[cell]
        return CellId(
            embodiment=self.embodiment,
            task=self.name,
            axes={"room": spec.room, "object": spec.object_label},
        )

    def build_env(self, cell: int) -> Any:
        spec = _HOUSEHOLD_CELLS[cell]
        env = _MockEnv(horizon=self._max_episode_steps)
        # Override the env's instruction so it surfaces in the rollout
        # metadata.json (via the Adapter's instruction_for path).
        env._instruction = spec.instruction
        return env

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    def instruction_for(self, env: Any) -> str:
        """Optional Task hook the GymAdapter looks up via getattr. Returns
        the cell-specific instruction so the rollout's metadata.json
        records it correctly."""
        getter = getattr(env, "get_language_instruction", None)
        if callable(getter):
            try:
                return str(getter())
            except Exception:
                return ""
        return ""
