"""RandomLocomotionPolicy — joint-space placeholder for v0 framework tests.

Emits random `JointAction`s within a configurable per-joint bound. Stable
seeded RNG keyed off the instruction so each episode is reproducible.
Used for:
- Contract tests against MockGo1Env (verifies the rollout loop and
  manifest pipeline absorb joint-space actions end-to-end).
- Smoke testing a real MuJoCo Playground Go1 env once the dep is
  installed (the random policy won't solve walking, but it will produce
  a valid sweep + manifest with non-trivial trajectories).

Real Go1 evaluation uses a published locomotion checkpoint — wiring up
DeepMimic / RoboPianist / mujoco_playground reference weights is named
as mechanical follow-up work in README.md.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .._types import JointAction, Observation


class RandomLocomotionPolicy:
    """Bounded random joint-space policy. CI-cheap; not for production."""

    def __init__(self, action_dim: int = 12, bound: float = 0.3, seed: int = 0) -> None:
        self._action_dim = action_dim
        self._bound = bound
        self._base_seed = seed
        self._rng = np.random.default_rng(seed)
        self._instruction = ""
        self._step_count = 0

    def reset(self, instruction: str) -> None:
        self._instruction = instruction
        # Python's built-in str hash is salted per-process (PYTHONHASHSEED),
        # which would make instruction-derived seeding non-reproducible
        # across runs. Use SHA256 for a stable digest.
        digest = hashlib.sha256(instruction.encode("utf-8")).digest()
        instr_seed = (self._base_seed + int.from_bytes(digest[:4], "big")) & 0xFFFFFFFF
        self._rng = np.random.default_rng(instr_seed)
        self._step_count = 0

    def step(self, observation: Observation) -> JointAction:
        self._step_count += 1
        vec = self._rng.uniform(-self._bound, self._bound, size=self._action_dim).astype(np.float32)
        return JointAction(
            vector=vec,
            terminate=False,
            raw={"step": self._step_count, "norm": float(np.linalg.norm(vec))},
        )

    @property
    def name(self) -> str:
        return f"random-locomotion-{self._action_dim}dof"

    @property
    def checkpoint_id(self) -> str:
        return f"mock:random-locomotion-policy:dim={self._action_dim}:bound={self._bound}:seed={self._base_seed}"

    @property
    def family(self) -> str:
        return "random-locomotion"
