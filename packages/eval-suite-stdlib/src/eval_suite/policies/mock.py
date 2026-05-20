"""MockPolicy — zero actions, used by the CI contract test.

**In plain words.** A stand-in robot model that does nothing — it
emits zero actions every step. It exists so the suite's CI tests can
exercise the full sweep pipeline (manifest, signing, analysis)
without having to load a real model. Catches plumbing bugs without
needing a GPU.


The CI test asserts that a no-op policy round-trips through the Adapter
and produces a valid Manifest. No SimplerEnv, no GPU, no real model.
Keeps the contract test fast (<5 seconds on a hosted runner) and
independent of upstream simulator versions.
"""

from __future__ import annotations

import numpy as np

from .._types import Action, Observation


class MockPolicy:
    """A Policy that returns zero deltas and never terminates the episode.

    Useful for: CI contract verification, smoke-testing the Adapter with
    a fresh env, debugging the manifest pipeline without GPU.
    """

    def __init__(self, *, terminate_after: int | None = None) -> None:
        self._terminate_after = terminate_after
        self._step_count = 0

    def reset(self, instruction: str) -> None:
        self._step_count = 0

    def step(self, observation: Observation) -> Action:
        self._step_count += 1
        do_terminate = self._terminate_after is not None and self._step_count >= self._terminate_after
        return Action(
            world_vector=np.zeros(3, dtype=np.float32),
            rot_axangle=np.zeros(3, dtype=np.float32),
            gripper=np.zeros(1, dtype=np.float32),
            terminate=do_terminate,
            raw={"step": self._step_count},
        )

    @property
    def name(self) -> str:
        return "mock-zero"

    @property
    def checkpoint_id(self) -> str:
        # No weights; use a constant sentinel so manifests are still hashable.
        return "mock:zero-action-policy:v1"

    @property
    def family(self) -> str:
        return "mock"
