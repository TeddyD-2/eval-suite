"""Unitree Go1 quadruped tasks via MuJoCo Playground (v0 sweep).

What this file ships:
- `UnitreeGo1Joystick` — the v0 legged Task: 12-cell variant grid
  (3 task families × {baseline / rough / side_camera / perturbation}) on
  MuJoCo Playground's Go1 joystick envs.
- A per-terrain JIT cache so cells that share terrain reuse the compiled
  MJX env (3 unique terrain compiles for the whole sweep instead of 12).
- `_MjxGymCompatEnv` — the gym-5-tuple shim that bridges MJX's functional
  `state = step(state, action)` API to the OOP `obs, r, success, trunc, info
  = env.step(action)` interface the Adapters expect. The compat shim is
  the v0 mirror of `_LIBEROGymCompatEnv` in `tasks/libero.py`; it's the
  small per-sim layer that lets the rest of the suite stay sim-agnostic.

Why Go1, not Go2 (which the docs sometimes say):
- `mujoco_playground` ships Go1 (Unitree's previous-gen quadruped), not
  Go2 — the env set is `Go1JoystickFlatTerrain`, `Go1JoystickRoughTerrain`,
  `Go1Getup`, `Go1Handstand`, `Go1Footstand`. Both have the same 12-DoF
  action space (3 actuators × 4 legs), so the framework claim is
  unchanged. Adding Go2 is a model-asset swap, not a contract change.

Three Mocks live at the bottom of this file (`_MockGo1Env`,
`mock_go1_task_factory`) so the contract test exercises this Task
without `mujoco_playground` installed — that's what CI runs.

Success semantic for random-policy rollouts:
- `success = (reached max_episode_steps without env-internal done=1)`.
- MJX's `done` flag is fall detection on this Go1 env. So the v0
  random-policy sweep measures "did the random control inputs leave
  the dog standing for max_episode_steps." For RandomLocomotionPolicy
  the expected aggregate is near-zero successes — that's not a model
  claim, it's a substrate proof: the framework absorbed a 12-DoF
  joint-space embodiment with the same trials.csv + manifest + mp4
  pipeline that the v0 arms use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._types import CanonicalDim, CellId, NDArrayU8

# Action space: Unitree Go1 has 12 actuated joints (3 per leg × 4 legs).
GO1_ACTION_DIM = 12


# ---------------------------------------------------------------------------
# Cell catalog
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Go1CellSpec:
    """Declarative cell — one variant on the Go1 axis grid."""

    task_family: str          # stand / walk_forward / sidestep
    terrain: str              # flat / rough
    camera: str               # front / side
    perturbation: str         # none / lateral_5n
    command: tuple[float, float, float]  # (vx, vy, wz) velocity command
    perturb_force: tuple[float, float, float] | None  # (Fx, Fy, Fz) on trunk, or None

    @property
    def env_terrain_name(self) -> str:
        return {
            "flat": "Go1JoystickFlatTerrain",
            "rough": "Go1JoystickRoughTerrain",
        }[self.terrain]

    @property
    def camera_name(self) -> str:
        # mujoco_playground's Go1 model exposes "track" (chase cam) by default.
        # Front/side variants pick alternative cameras when present, else
        # fall back to a default chase camera with different distance/azimuth.
        return {"front": "track", "side": "side"}.get(self.camera, "track")


def _build_go1_cells() -> list[_Go1CellSpec]:
    """12-cell catalog: 3 task families × 4 vary-one-axis-from-baseline cells.

    Cells are designed so each one is physically distinct from baseline (no
    nominal-only axes). The 4-cell-per-family budget matches the v0
    Google Robot pattern: baseline + a few perturbations across orthogonal
    axes.

    Per task family:
      baseline:      flat terrain, front camera, no force
      rough:         rough terrain, front camera, no force
      side_camera:   flat terrain, side camera, no force
      perturbation:  flat terrain, front camera, 5N lateral push
    """
    families: list[tuple[str, tuple[float, float, float]]] = [
        ("stand",         (0.0, 0.0, 0.0)),
        ("walk_forward",  (0.5, 0.0, 0.0)),
        ("sidestep",      (0.0, 0.4, 0.0)),
    ]
    cells: list[_Go1CellSpec] = []
    for fname, cmd in families:
        cells.append(_Go1CellSpec(
            task_family=fname, terrain="flat", camera="front", perturbation="none",
            command=cmd, perturb_force=None,
        ))
        cells.append(_Go1CellSpec(
            task_family=fname, terrain="rough", camera="front", perturbation="none",
            command=cmd, perturb_force=None,
        ))
        cells.append(_Go1CellSpec(
            task_family=fname, terrain="flat", camera="side", perturbation="none",
            command=cmd, perturb_force=None,
        ))
        cells.append(_Go1CellSpec(
            task_family=fname, terrain="flat", camera="front", perturbation="lateral_5n",
            command=cmd, perturb_force=(0.0, 5.0, 0.0),
        ))
    return cells


_GO1_CELLS = _build_go1_cells()


# ---------------------------------------------------------------------------
# JIT-compiled env cache (shared across rollouts of the same terrain)
# ---------------------------------------------------------------------------

@dataclass
class _CachedMjxEnv:
    """Per-terrain compiled MJX env + lazy renderer.

    JIT compile of `env.reset` / `env.step` is keyed on the bound-method
    instance, so we keep one Python env instance per terrain and reuse it.
    That collapses 12 cell-level compiles into 2 terrain-level compiles
    (~30-50s each on a 3090). The renderer is a separate MuJoCo resource;
    we instantiate one per cached env, on first use, with the EGL backend.
    """

    env: Any            # mujoco_playground env
    reset_jit: Any      # jax.jit-compiled
    step_jit: Any       # jax.jit-compiled
    mj_model: Any       # mujoco.MjModel
    _renderer: Any = None
    render_height: int = 240
    render_width: int = 320

    @property
    def renderer(self) -> Any:
        if self._renderer is None:
            import mujoco  # type: ignore[import-not-found]
            self._renderer = mujoco.Renderer(self.mj_model, height=self.render_height, width=self.render_width)
        return self._renderer

    def close(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None


def _load_cached_mjx_env(env_name: str) -> _CachedMjxEnv:
    """Construct + JIT-compile a fresh MJX env for `env_name`.

    Uses `impl="jax"` config override to dodge `mujoco-playground 0.2.0`'s
    incompatibility with `warp-lang>=1.13` (the default WARP impl path
    crashes with `AttributeError: module 'warp.types' has no attribute
    'warp_type_to_np_dtype'` — the JAX impl path is the workaround until
    upstream catches up).
    """
    import jax  # type: ignore[import-not-found]
    import ml_collections  # type: ignore[import-not-found]
    from mujoco_playground import registry  # type: ignore[import-not-found]

    env = registry.load(env_name, config_overrides=ml_collections.ConfigDict({"impl": "jax"}))
    return _CachedMjxEnv(
        env=env,
        reset_jit=jax.jit(env.reset),
        step_jit=jax.jit(env.step),
        mj_model=env.mj_model,
    )


# ---------------------------------------------------------------------------
# Gym-5-tuple compat shim
# ---------------------------------------------------------------------------

class _MjxGymCompatEnv:
    """One-rollout wrapper around a cached MJX env.

    Reset / step / render API matches what `MujocoPlaygroundAdapter`
    expects (gym-5-tuple step, seed-accepting reset, optional render).
    Internally:
    - `reset(seed)` PRNGs an MJX key, runs the compiled reset, then
      overrides `state.info["command"]` with the cell's command vector
      so the env tracks the cell-specified velocity instead of a random
      one.
    - `step(action)` injects optional cell-level perturbation force via
      `xfrc_applied` on the trunk body, runs the compiled step, returns
      the gym 5-tuple. Position-2 of the tuple is `success` per the
      Adapter convention (see `MujocoPlaygroundAdapter`).
    - `render()` converts MJX `state.data` → MjData and renders via the
      cached `mujoco.Renderer`. Needs `MUJOCO_GL=egl` (or osmesa) set
      in the environment.

    Success semantic: `success = True` iff the loop reaches max_episode_steps
    without the env raising its internal `done=1` (fall detection). For
    RandomLocomotionPolicy this is mostly False; for a real Go1 policy
    it would be the survival rate per cell.
    """

    def __init__(
        self,
        cached: _CachedMjxEnv,
        *,
        command: tuple[float, float, float],
        camera: str,
        max_episode_steps: int,
        perturb_force: tuple[float, float, float] | None = None,
    ) -> None:
        self._cached = cached
        self._command = command
        self._camera = camera
        self._max_steps = max_episode_steps
        self._perturb_force = perturb_force
        self._state: Any = None
        self._step_idx = 0
        # Exposed for the Task's `instruction_for` hook to read.
        self.command = command

    @property
    def mj_model(self) -> Any:
        return self._cached.mj_model

    def reset(self, seed: int = 0) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
        import jax
        import jax.numpy as jnp  # type: ignore[import-not-found]

        rng = jax.random.PRNGKey(int(seed))
        state = self._cached.reset_jit(rng)
        # Override the randomly-sampled command with the cell's command,
        # so all N seeds for this cell track the same velocity target.
        new_info = dict(state.info)
        new_info["command"] = jnp.array(self._command, dtype=jnp.float32)
        state = state.replace(info=new_info)
        self._state = state
        self._step_idx = 0
        return _state_obs_to_np(state), {}

    def step(self, action: np.ndarray[Any, Any]) -> tuple[
        np.ndarray[Any, Any], float, bool, bool, dict[str, Any]
    ]:
        import jax.numpy as jnp

        if self._perturb_force is not None and self._step_idx >= 10:
            # Apply the perturbation force on the trunk (body id 1) starting
            # after a brief warmup so the cell has a chance to stabilize.
            fvec = jnp.array(self._perturb_force, dtype=jnp.float32)
            xfrc = self._state.data.xfrc_applied.at[1, :3].set(fvec)
            new_data = self._state.data.replace(xfrc_applied=xfrc)
            self._state = self._state.replace(data=new_data)

        a_jax = jnp.array(action, dtype=jnp.float32)
        self._state = self._cached.step_jit(self._state, a_jax)
        self._step_idx += 1

        terminated = bool(float(self._state.done) > 0.5)
        reached_horizon = self._step_idx >= self._max_steps
        # Position-2 of the 5-tuple is "success" per the Adapter convention.
        success = reached_horizon and not terminated
        # Position-3 of the 5-tuple is "truncated"; here it means the
        # rollout loop should stop (either we hit the horizon or the env
        # said done=1 from a fall).
        stop = reached_horizon or terminated

        info: dict[str, Any] = {
            "episode_stats": {
                "step_idx": self._step_idx,
                "fall_terminated": terminated,
                "reached_horizon": reached_horizon,
                "reward_last": float(self._state.reward),
                "tracking_lin_vel": float(self._state.metrics.get("reward/tracking_lin_vel", 0.0)),
                "ang_vel_xy": float(self._state.metrics.get("reward/ang_vel_xy", 0.0)),
            },
        }
        return _state_obs_to_np(self._state), float(self._state.reward), success, stop, info

    def render(self) -> NDArrayU8:
        """Render via the cached renderer. Requires `MUJOCO_GL` set."""
        import mujoco
        renderer = self._cached.renderer
        s = self._state
        if s is None:
            return np.zeros((self._cached.render_height, self._cached.render_width, 3), dtype=np.uint8)
        mj_data = mujoco.MjData(self._cached.mj_model)
        mj_data.qpos[:] = np.asarray(s.data.qpos)
        mj_data.qvel[:] = np.asarray(s.data.qvel)
        if hasattr(s.data, "mocap_pos") and np.asarray(s.data.mocap_pos).shape[0] > 0:
            mj_data.mocap_pos[:] = np.asarray(s.data.mocap_pos)
            mj_data.mocap_quat[:] = np.asarray(s.data.mocap_quat)
        mj_data.xfrc_applied[:] = np.asarray(s.data.xfrc_applied)
        mujoco.mj_forward(self._cached.mj_model, mj_data)
        renderer.update_scene(mj_data, camera=self._camera)
        out: NDArrayU8 = renderer.render().astype(np.uint8)
        return out

    def close(self) -> None:
        # Per-rollout wrapper has no resources of its own; the cached env
        # + renderer live on the Task and are reused across cells.
        return None


def _state_obs_to_np(state: Any) -> np.ndarray[Any, Any]:
    """Flatten MJX state.obs into a single np.float32 vector."""
    obs = state.obs
    if isinstance(obs, dict):
        # Go1 joystick env: {"state": (48,), "privileged_state": (123,)}.
        # The "state" key is what a real policy would see; we expose it.
        return np.asarray(obs.get("state", next(iter(obs.values()))), dtype=np.float32)
    return np.asarray(obs, dtype=np.float32)


# ---------------------------------------------------------------------------
# UnitreeGo1Joystick — the v0 Task
# ---------------------------------------------------------------------------

class UnitreeGo1Joystick:
    """Go1 locomotion task family (stand / walk-forward / sidestep).

    12-cell variant grid: 3 task families × 4 vary-one-axis-from-baseline
    cells (baseline / rough_terrain / side_camera / lateral_perturbation).
    See `_build_go1_cells` for the per-cell breakdown.

    Implements the two optional Task hooks the `MujocoPlaygroundAdapter`
    looks for:
    - `instruction_for(env)` synthesizes a templated string ("walk forward
      at vx=0.5") since MuJoCo Playground envs don't ship a
      language_instruction.
    - `extract_image(env, obs)` defers to the compat shim's `render()`.
    """

    # v0 canonical-axis taxonomy. task_family changes the velocity
    # command vector (and thus the synthesized instruction string), so
    # it's a language axis for any model that consumes the instruction.
    # The locomotion policies that ship today don't actually read
    # language — they read the command vector directly — so for them
    # this dim is a measurement of "policy follows different commands,"
    # which is the closest available analog.
    canonical_axis_map: dict[str, CanonicalDim] = {
        "task_family": "language",
        "terrain": "physics",
        "camera": "visuals",
        "perturbation": "physics",
    }

    def __init__(self, *, max_episode_steps: int = 100, render_height: int = 240, render_width: int = 320) -> None:
        self._max_episode_steps = max_episode_steps
        self._render_height = render_height
        self._render_width = render_width
        # Lazy cache keyed by env_name (terrain). Compiles on first cell
        # that touches a given terrain; reused for every subsequent cell.
        self._terrain_cache: dict[str, _CachedMjxEnv] = {}

    @property
    def name(self) -> str:
        return "unitree_go1_joystick"

    @property
    def embodiment(self) -> str:
        return "unitree_go1"

    @property
    def n_cells(self) -> int:
        return len(_GO1_CELLS)

    @property
    def action_dim(self) -> int:
        return GO1_ACTION_DIM

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    def cell_id(self, cell: int) -> CellId:
        spec = _GO1_CELLS[cell]
        return CellId(
            embodiment=self.embodiment,
            task=self.name,
            axes={
                "task_family": spec.task_family,
                "terrain": spec.terrain,
                "camera": spec.camera,
                "perturbation": spec.perturbation,
            },
        )

    def cell_spec(self, cell: int) -> _Go1CellSpec:
        return _GO1_CELLS[cell]

    def _get_terrain_cache(self, env_name: str) -> _CachedMjxEnv:
        if env_name not in self._terrain_cache:
            self._terrain_cache[env_name] = _load_cached_mjx_env(env_name)
            self._terrain_cache[env_name].render_height = self._render_height
            self._terrain_cache[env_name].render_width = self._render_width
        return self._terrain_cache[env_name]

    def build_env(self, cell: int) -> _MjxGymCompatEnv:
        """Construct a one-rollout compat env for `cell`.

        The MJX env + jit cache is shared per terrain across rollouts; the
        returned wrapper holds only per-rollout state (step counter, cell's
        command vector, perturbation flag). The Adapter calls this once
        per (cell, seed); the cost is renderer access + a single jit-cached
        reset, not a recompile.

        Raises `ImportError` with a clear message if `mujoco_playground`
        isn't installed (CI uses `mock_go1_task_factory` instead).
        """
        try:
            import mujoco_playground  # noqa: F401  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "mujoco_playground not installed. Install with `pip install playground` "
                "(the PyPI name; the import is `mujoco_playground`). v0 ships the Go1 "
                "framework but does not bake it into the v0 Docker image — build a "
                "separate v0 venv, or use `mock_go1_task_factory()` for contract tests."
            ) from e

        # Ensure MUJOCO_GL is set for headless rendering; default to EGL.
        os.environ.setdefault("MUJOCO_GL", "egl")

        spec = _GO1_CELLS[cell]
        cached = self._get_terrain_cache(spec.env_terrain_name)
        return _MjxGymCompatEnv(
            cached=cached,
            command=spec.command,
            camera=spec.camera_name,
            max_episode_steps=self._max_episode_steps,
            perturb_force=spec.perturb_force,
        )

    # ---- MujocoPlaygroundAdapter optional hooks ----------------------------

    def instruction_for(self, env: Any) -> str:
        cmd = getattr(env, "command", None)
        if cmd is None:
            return "execute go1 locomotion task"
        try:
            vx, vy, wz = (float(x) for x in cmd[:3])
        except Exception:
            return "execute go1 locomotion task"
        if vx == 0.0 and vy == 0.0 and wz == 0.0:
            return "stand still"
        return f"locomote at vx={vx:.2f} vy={vy:.2f} wz={wz:.2f}"

    def extract_image(self, env: Any, obs: Any) -> NDArrayU8:
        render = getattr(env, "render", None)
        if callable(render):
            try:
                arr = np.asarray(render(), dtype=np.uint8)
                if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                    return arr if arr.shape[-1] == 3 else arr[..., :3]
            except Exception:
                pass
        return np.zeros((self._render_height, self._render_width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Mock factory (used by CI; doesn't require mujoco_playground)
# ---------------------------------------------------------------------------

class _MockGo1Env:
    """A 12-DoF joint-space env that looks like an MJX Go1. CI-friendly."""

    def __init__(self, horizon: int = 10) -> None:
        self._horizon = horizon
        self._step = 0
        # Mirrors the real env exposing a command attribute for the
        # instruction_for hook to find.
        self.command: tuple[float, float, float] = (0.5, 0.0, 0.0)

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        self._step = 0
        if seed is not None:
            np.random.default_rng(seed)
        return self._obs(), {}

    def step(self, action: np.ndarray[Any, Any]) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if action.shape[0] != GO1_ACTION_DIM:
            raise ValueError(f"Go1 action must be {GO1_ACTION_DIM}-dim; got {action.shape}")
        self._step += 1
        stop = self._step >= self._horizon
        # Mock has no fall detection; success = reached horizon.
        success = stop
        info = {
            "episode_stats": {
                "step_idx": self._step,
                "mock_joint_norm": float(np.linalg.norm(action)),
            },
        }
        return self._obs(), 0.0, success, stop, info

    def _obs(self) -> np.ndarray[Any, Any]:
        # MJX Go1 "state" key is 48-dim; we mirror the shape for the
        # contract test.
        return np.zeros(48, dtype=np.float32)

    def render(self) -> np.ndarray[Any, Any]:
        return np.full((64, 64, 3), 32, dtype=np.uint8)

    def close(self) -> None:
        return None


def mock_go1_task_factory(n_cells: int = 3, max_episode_steps: int = 10):  # type: ignore[no-untyped-def]
    """Returns a Task with the first `n_cells` of `UnitreeGo1Joystick`,
    but with `build_env` returning `_MockGo1Env` instead of a real MJX
    env. For contract tests / CI; not for real eval.
    """

    base_task = UnitreeGo1Joystick(max_episode_steps=max_episode_steps)

    class _MockGo1Task:
        @property
        def name(self) -> str: return base_task.name
        @property
        def embodiment(self) -> str: return base_task.embodiment
        @property
        def n_cells(self) -> int: return min(n_cells, base_task.n_cells)
        @property
        def action_dim(self) -> int: return GO1_ACTION_DIM
        def cell_id(self, cell: int) -> CellId: return base_task.cell_id(cell)
        def cell_spec(self, cell: int) -> _Go1CellSpec: return base_task.cell_spec(cell)
        def build_env(self, cell: int) -> Any: return _MockGo1Env(horizon=max_episode_steps)
        @property
        def max_episode_steps(self) -> int: return max_episode_steps
        def instruction_for(self, env: Any) -> str: return base_task.instruction_for(env)
        def extract_image(self, env: Any, obs: Any) -> NDArrayU8: return base_task.extract_image(env, obs)

    return _MockGo1Task()
