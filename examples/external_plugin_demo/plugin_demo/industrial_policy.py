"""Industrial-arm mock Policy — a reference external Policy plugin.

Wraps eval-suite's `MockPolicy` (zero EEF actions) but with an
industrial-arm-themed name and a deterministic action perturbation.
What's "external" about this Policy is the label, the checkpoint id,
and the kwargs shape — exactly what a third-party plugin would publish.

A real industrial-arm policy would call into the actual model
inference here; the contract surface (Policy Protocol) is identical.
"""

from __future__ import annotations

import numpy as np

from eval_suite._types import Action, Observation


class IndustrialArmMockPolicy:
    """7-DoF EEF Policy with an industrial-arm-themed name.

    Emits zero actions for all components except a tiny constant
    perturbation on the world_vector axis (so the action_norm column in
    `metadata.json` isn't identically zero — useful when inspecting
    `trajectory.npz` from a rollout). The behaviour is deterministic;
    `verify_policy` and `roundtrip_determinism` from the conformance kit
    both pass against this class.
    """

    def __init__(self, *, perturbation: float = 0.01, model_id: str = "mock-industrial-arm-v1") -> None:
        self._perturbation = float(perturbation)
        self._model_id = model_id

    def reset(self, instruction: str) -> None:
        return None

    def step(self, observation: Observation) -> Action:
        return Action(
            world_vector=np.array([self._perturbation, 0.0, 0.0], dtype=np.float32),
            rot_axangle=np.zeros((3,), dtype=np.float32),
            gripper=np.zeros((1,), dtype=np.float32),
            terminate=False,
        )

    @property
    def name(self) -> str:
        return self._model_id

    @property
    def checkpoint_id(self) -> str:
        return f"mock:industrial-arm:{self._perturbation:.3f}"
