"""LeRobotPolicy — wraps a LeRobot-formatted policy as an eval-suite `Policy`.

**In plain words.** The HuggingFace LeRobot project has become the
de-facto place where robotics groups publish their pretrained policy
checkpoints — `lerobot/smolvla-base`, `lerobot/pi0`, ACT, Diffusion.
This file is what lets the suite evaluate any of them by name,
without anyone having to write a custom wrapper per checkpoint. A
user types `--policy lerobot --policy-arg repo_id=lerobot/smolvla-base`
and the suite picks the model up off the Hub and drives it through
the full sweep.


LeRobot (the HuggingFace project, not the dataset spec) distributes
pretrained policy checkpoints via the HF Hub (`lerobot/smolvla-base`,
`lerobot/pi0`, ACT/Diffusion variants, etc.). Each checkpoint is loaded
through `lerobot.policies.from_pretrained(repo_id, revision=...)` —
the loader resolves the revision SHA at load time, which is what makes
`checkpoint_id` reproducible across sessions.

The interop contract this proves: a LeRobot policy drops into the
suite **without contract changes**. `Policy.reset(instruction)` +
`Policy.step(observation)` is enough — LeRobot policies fit it
directly. The class lives in `eval-suite-stdlib` behind the
`[lerobot]` extra so the default install stays light; the lazy import
inside `__init__` means `python -m eval_suite.cli list` works even
when `lerobot` isn't installed.

What's deliberately conservative here:

  - The observation dict passed to `policy.select_action` follows the
    most common LeRobot convention (`observation.image`,
    `observation.state`, `task`). Some LeRobot model cards use
    multi-camera keys; the dispatch table in
    `_lerobot_observation_format` handles the few in-tree-supported
    shapes. New shapes are one-liners to add.
  - Action unpacking checks `action_dim`. 7-DoF EEF returns `Action`;
    anything else returns `JointAction`. Bimanual / mobile-base
    policies that don't fit either still work — the env's Adapter
    consumes the typed Action.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._types import Action, ActionLike, JointAction, NDArrayF32, Observation


class LeRobotPolicy:
    """Wraps a LeRobot Hub checkpoint under the eval-suite `Policy` Protocol.

    Construct with:

      LeRobotPolicy(repo_id="lerobot/smolvla-base")           # latest revision
      LeRobotPolicy(repo_id="lerobot/pi0", revision="abc123") # pinned

    `device` defaults to "cuda" — pass "cpu" for CI smoke tests.
    """

    def __init__(
        self,
        *,
        repo_id: str,
        revision: str | None = None,
        device: str = "cuda",
    ) -> None:
        # Lazy-import torch + lerobot only at construction so `python -m
        # eval_suite.cli list` doesn't pay the cost when neither is
        # installed. ImportError surfaces a one-line install hint.
        try:
            import torch
            from lerobot.policies import (  # type: ignore[import-not-found]
                from_pretrained as _from_pretrained,
            )
        except ImportError as e:
            raise RuntimeError(
                "LeRobotPolicy requires the `[lerobot]` extra: "
                "`pip install 'eval-suite-stdlib[lerobot]'`."
            ) from e

        self._repo_id = repo_id
        self._device = device
        self._model = _from_pretrained(repo_id, revision=revision)
        self._model.to(device)
        self._model.eval()
        self._torch = torch
        # `from_pretrained` resolves revision=None to the current main-branch
        # SHA at load time; stash whatever the loader settled on for the
        # manifest's checkpoint_id (so two reloads with revision=None still
        # produce identical ids as long as upstream main is stable).
        cfg = getattr(self._model, "config", None)
        self._resolved_revision = (
            getattr(cfg, "revision_sha", None)
            or getattr(cfg, "_commit_hash", None)
            or revision
            or "unresolved"
        )
        self._instruction: str = ""

    def reset(self, instruction: str) -> None:
        self._instruction = instruction
        reset_fn = getattr(self._model, "reset", None)
        if callable(reset_fn):
            reset_fn()

    def step(self, observation: Observation) -> ActionLike:
        obs_batch = self._build_observation_batch(observation)
        with self._torch.no_grad():
            action_tensor = self._model.select_action(obs_batch)
        action_np = action_tensor.detach().cpu().numpy().astype(np.float32).reshape(-1)
        return _action_from_vector(action_np)

    def _build_observation_batch(self, observation: Observation) -> dict[str, Any]:
        torch = self._torch
        # LeRobot expects batched tensors (B=1). Image: HWC uint8 → BCHW float32 in [0,1].
        img_chw = (
            torch.from_numpy(observation.image)
            .to(self._device, dtype=torch.float32)
            .permute(2, 0, 1)
            .div_(255.0)
            .unsqueeze(0)
        )
        batch: dict[str, Any] = {
            "observation.image": img_chw,
            "task": observation.instruction or self._instruction,
        }
        state = observation.extra.get("state") if observation.extra else None
        if state is not None:
            batch["observation.state"] = (
                torch.from_numpy(np.asarray(state, dtype=np.float32))
                .to(self._device)
                .unsqueeze(0)
            )
        return batch

    @property
    def name(self) -> str:
        return f"lerobot:{self._repo_id}"

    @property
    def checkpoint_id(self) -> str:
        return f"hf:{self._repo_id}@{str(self._resolved_revision)[:16]}"

    @property
    def family(self) -> str:
        return "lerobot"


def _action_from_vector(vec: NDArrayF32) -> ActionLike:
    """Unpack a LeRobot action vector into either Action (7-DoF EEF) or JointAction.

    The 7-DoF EEF convention matches SimplerEnv + Bridge + LIBERO and is
    the canonical OXE shape — that path keeps the manipulation pipeline
    unchanged. Any other length becomes a JointAction so the legged /
    bimanual cases route through MujocoPlaygroundAdapter or a sibling.
    """
    if vec.shape[0] == 7:
        return Action(
            world_vector=vec[:3].copy(),
            rot_axangle=vec[3:6].copy(),
            gripper=vec[6:7].copy(),
            terminate=False,
        )
    return JointAction(vector=vec.copy(), terminate=False)
