"""SimplerEnv-backed Policy — wraps Octo and RT-1 under one `Policy`.

Both Octo and RT-1's reference implementations in SimplerEnv expose the
same `reset(instruction)` / `step(image, instruction)` shape. This
adapter normalizes that to the `Policy` protocol and tags the wrapped
model with a stable checkpoint identifier for the manifest.

This module is import-lazy: SimplerEnv pulls in tensorflow, jax,
sapien, etc. — fine on the GPU box, but the CI contract test must
not require any of it. The actual model is constructed in
`__init__`, so importing this module is cheap.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .._types import Action, Observation
from ..hashing import sha256_dir


class SimplerEnvPolicy:
    """Wraps SimplerEnv's RT1Inference or OctoInference under our Policy.

    Construct with:
      - `family="rt1"`, ckpt_path pointing to the local TF saved-model dir.
      - `family="octo"`, ckpt_path = None and model_type in {"octo-base",
        "octo-small"} — Octo loads weights from HuggingFace at init.

    `policy_setup` is "google_robot" or "widowx_bridge" per SimplerEnv's
    convention; it selects gripper conventions and action statistics. The
    Task tells the adapter which setup to use when constructing the
    Policy for a given embodiment.
    """

    def __init__(
        self,
        *,
        family: str,
        policy_setup: str,
        ckpt_path: str | None = None,
        model_type: str = "octo-base",
        init_rng: int = 0,
    ) -> None:
        self._family = family
        self._policy_setup = policy_setup
        self._ckpt_path = ckpt_path
        self._model_type = model_type
        self._instruction: str = ""

        if family == "rt1":
            if not ckpt_path:
                raise ValueError("RT-1 requires --ckpt-path pointing to a saved_model dir")
            from simpler_env.policies.rt1.rt1_model import RT1Inference
            self._model: Any = RT1Inference(saved_model_path=ckpt_path, policy_setup=policy_setup)
            self._checkpoint_id = self._compute_rt1_checkpoint_id(ckpt_path)
        elif family == "octo":
            from simpler_env.policies.octo.octo_model import OctoInference
            self._model = OctoInference(model_type=model_type, policy_setup=policy_setup, init_rng=init_rng)
            self._checkpoint_id = f"hf:rail-berkeley/{model_type}"
        else:
            raise ValueError(f"unknown policy family: {family}")

    @staticmethod
    def _compute_rt1_checkpoint_id(ckpt_path: str) -> str:
        path = Path(ckpt_path)
        if path.is_dir():
            return f"sha256:{sha256_dir(path)}"
        return f"path:{os.path.basename(ckpt_path)}"

    def reset(self, instruction: str) -> None:
        self._instruction = instruction
        self._model.reset(instruction)

    def step(self, observation: Observation) -> Action:
        if observation.image.dtype != np.uint8:
            raise TypeError(f"SimplerEnv policies expect uint8 RGB; got {observation.image.dtype}")
        _raw_action, action_dict = self._model.step(observation.image, observation.instruction or self._instruction)
        terminate_flag = bool(np.asarray(action_dict.get("terminate_episode", [0]))[0] > 0)
        return Action(
            world_vector=np.asarray(action_dict["world_vector"], dtype=np.float32),
            rot_axangle=np.asarray(action_dict["rot_axangle"], dtype=np.float32),
            gripper=np.asarray(action_dict["gripper"], dtype=np.float32).reshape(1),
            terminate=terminate_flag,
            raw={k: np.asarray(v).tolist() for k, v in action_dict.items()},
        )

    @property
    def name(self) -> str:
        if self._family == "octo":
            return self._model_type
        return f"rt1:{os.path.basename(self._ckpt_path or '')}"

    @property
    def checkpoint_id(self) -> str:
        return self._checkpoint_id

    @property
    def family(self) -> str:
        # Surfaced into Manifest.model.family by sweep.py — used by
        # analysis to group submissions and pick the right calibration
        # reference. "rt1" / "octo".
        return self._family
