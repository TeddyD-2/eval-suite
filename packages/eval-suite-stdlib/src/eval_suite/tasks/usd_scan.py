"""NamaqualandScanTask — v0 real-world scan ingestion demo.

**In plain words.** This task drops a photogrammetric scan of a real
boulder into a Go1 simulator scene and runs the locomotion sweep
against the result. It's the proof that the suite can take a
real-world capture (here, a free CC0 scan from Poly Haven) and
evaluate a policy against it through unchanged contract code —
which is the precondition for the splat / RGB-D ingest pipelines
that turn user-supplied real-world scans into benchmark tasks.


Wraps the converted Poly Haven Namaqualand Boulder 05 scan (see
`assets/namaqualand_scan/` and `assets/namaqualand_scan/convert.py`) as
a single-cell Task driven through the existing v0
`MujocoPlaygroundAdapter`. The Adapter is sim-agnostic at the env
interface; this Task uses **plain MuJoCo (CPU)**, NOT MJX. The contract
is satisfied through the same gym-5-tuple shim pattern used by
`_MjxGymCompatEnv` in `tasks/unitree_go1.py`.

Composition strategy at `build_env`:
  1. Load the Menagerie Go1 MJCF that ships with `mujoco_playground` at
     `external_deps/mujoco_menagerie/unitree_go1/go1.xml`. Resolved at
     sweep time, not bake time, so the path survives mujoco_playground
     version bumps.
  2. Load the boulder scene MJCF produced by `convert.py`
     (`MJCF/scene.xml` — boulder visual mesh + decimated convex hull
     collision geom, no robot).
  3. Use `mujoco.MjSpec.attach(boulder_spec, prefix="scan_",
     frame=<go1.worldbody.frame>)` to compose them into one model.
  4. Compile to `MjModel`. The composed model has Go1's 12 actuators,
     19 qpos (free joint 7 + 12 leg joints), and the boulder as a
     static body in front of Go1.

Success semantic for v0: `survived max_episode_steps without falling`
where fall = Go1 trunk z-height below `FALL_HEIGHT_THRESHOLD` (0.15m).
Mirrors v0's UnitreeGo1Joystick (`tasks/unitree_go1.py:268-272`) but
implemented in plain MuJoCo rather than via MJX's `done` signal.

Single-cell Task (`n_cells=1`, `axes={}`). v0's
`canonical_axis_map` is the empty dict — v0 doesn't vary any axes;
the contribution is the asset ingestion pipeline, not a new variant
grid. Analysis treats empty maps as "coverage badge only" (consistent
with v0's zero-coverage handling per EXTENSION.md §2).

CI tests use `mock_namaqualand_task_factory(...)` which bypasses
MuJoCo entirely — same idiom as `mock_go1_task_factory` at
`tasks/unitree_go1.py:516-541`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .._types import CanonicalDim, CellId, NDArrayU8

GO1_ACTION_DIM = 12
TASK_NAME = "namaqualand_scan_v09"
TASK_EMBODIMENT = "unitree_go1"
DEFAULT_MAX_EPISODE_STEPS = 200
DEFAULT_BOULDER_POS = (1.0, 0.0, 0.0)
FALL_HEIGHT_THRESHOLD = 0.15
RENDER_HEIGHT = 240
RENDER_WIDTH = 320
TRUNK_BODY_NAME = "trunk"
SCENE_CAMERA_NAME = "scene_overview"
SCENE_CAMERA_POS = (-1.5, -2.0, 1.0)
SCENE_CAMERA_LOOKAT = (0.5, 0.0, 0.3)


def _lookat_quat(
    pos: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray[Any, Any]:
    """Compute the world-frame (w, x, y, z) quaternion for a MuJoCo camera
    at `pos` pointing at `target`.

    MuJoCo camera convention: +x is right, +y is up, -z is forward.
    """
    import mujoco  # type: ignore[import-not-found]

    p = np.asarray(pos, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    u = np.asarray(up, dtype=np.float64)
    forward = t - p
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, u)
    right /= np.linalg.norm(right)
    real_up = np.cross(right, forward)
    rot = np.column_stack([right, real_up, -forward])
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    return quat


def _resolve_go1_xml() -> Path:
    """Resolve `mujoco_playground`'s bundled Menagerie Go1 MJCF path.

    Called at sweep time (NOT bake time) so the include is robust to
    `mujoco_playground` version bumps. The Menagerie copy is preferred
    over `_src/locomotion/go1/xmls/go1_mjx.xml` because the Menagerie
    MJCF is plain MuJoCo (no MJX-specific opt-ins).
    """
    import mujoco_playground  # type: ignore[import-not-found]

    mp_root = Path(mujoco_playground.__file__).parent
    go1_xml = mp_root / "external_deps/mujoco_menagerie/unitree_go1/go1.xml"
    if not go1_xml.is_file():
        raise FileNotFoundError(
            f"Menagerie Go1 MJCF not found at {go1_xml}. The expected path is bundled "
            "with mujoco_playground's external_deps; reinstall mujoco_playground if missing."
        )
    return go1_xml


class _ScanSceneCompatEnv:
    """Plain-mujoco gym-5-tuple wrapper for the Go1+boulder composed scene.

    Sibling of `_MjxGymCompatEnv` in `tasks/unitree_go1.py`, but for
    plain `mujoco.MjModel`/`mujoco.MjData` instead of MJX. The Adapter
    is sim-agnostic, so this is contract-clean — it only needs to
    expose `reset(seed)`, `step(action)`, `render()`, `close()`.

    Fall detection: Go1's trunk body z-coordinate below
    `FALL_HEIGHT_THRESHOLD`. Step returns `(obs, reward, success,
    truncated, info)` per the Adapter's gym-5-tuple convention.
    """

    def __init__(
        self,
        *,
        scene_path: Path,
        go1_xml_path: Path,
        max_episode_steps: int,
        boulder_pos: tuple[float, float, float] = DEFAULT_BOULDER_POS,
    ) -> None:
        import mujoco

        self._max_steps = max_episode_steps
        go1_spec = mujoco.MjSpec.from_file(str(go1_xml_path))
        boulder_spec = mujoco.MjSpec.from_file(str(scene_path))
        anchor = go1_spec.worldbody.add_frame(name="boulder_anchor", pos=list(boulder_pos))
        go1_spec.attach(boulder_spec, prefix="scan_", frame=anchor)
        floor = go1_spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = np.array([10.0, 10.0, 0.05])
        floor.rgba = np.array([0.55, 0.55, 0.55, 1.0])
        light = go1_spec.worldbody.add_light()
        light.pos = np.array([0.0, 0.0, 5.0])
        light.dir = np.array([0.0, 0.0, -1.0])
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        camera = go1_spec.worldbody.add_camera()
        camera.name = SCENE_CAMERA_NAME
        camera.pos = np.array(SCENE_CAMERA_POS)
        camera.quat = _lookat_quat(SCENE_CAMERA_POS, SCENE_CAMERA_LOOKAT)
        self._model = go1_spec.compile()
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)
        self._initial_qpos = self._data.qpos.copy()
        trunk_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, TRUNK_BODY_NAME)
        if trunk_id < 0:
            raise RuntimeError(
                f"trunk body '{TRUNK_BODY_NAME}' not found in composed model — Go1 MJCF schema changed?"
            )
        self._trunk_id = int(trunk_id)
        self._renderer: Any | None = None
        self._step_idx = 0
        self.command: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def mj_model(self) -> Any:
        return self._model

    def reset(self, seed: int = 0) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
        import mujoco

        np.random.default_rng(int(seed))
        self._data.qpos[:] = self._initial_qpos
        self._data.qvel[:] = 0.0
        self._data.ctrl[:] = 0.0
        mujoco.mj_forward(self._model, self._data)
        self._step_idx = 0
        return self._obs(), {}

    def step(
        self, action: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], float, bool, bool, dict[str, Any]]:
        import mujoco

        n_ctrl = int(self._model.nu)
        ctrl = np.asarray(action, dtype=np.float32).reshape(-1)
        if ctrl.shape[0] < n_ctrl:
            raise ValueError(f"action dim {ctrl.shape[0]} < model.nu {n_ctrl}")
        self._data.ctrl[:n_ctrl] = ctrl[:n_ctrl]
        mujoco.mj_step(self._model, self._data)
        self._step_idx += 1

        trunk_z = float(self._data.xpos[self._trunk_id, 2])
        terminated = trunk_z < FALL_HEIGHT_THRESHOLD
        reached_horizon = self._step_idx >= self._max_steps
        success = reached_horizon and not terminated
        stop = reached_horizon or terminated

        info: dict[str, Any] = {
            "episode_stats": {
                "step_idx": self._step_idx,
                "trunk_z": trunk_z,
                "fall_terminated": terminated,
                "reached_horizon": reached_horizon,
            }
        }
        return self._obs(), 0.0, success, stop, info

    def render(self) -> NDArrayU8:
        import mujoco

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
        self._renderer.update_scene(self._data, camera=SCENE_CAMERA_NAME)
        return np.asarray(self._renderer.render(), dtype=np.uint8)

    def close(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None

    def _obs(self) -> np.ndarray[Any, Any]:
        out: np.ndarray[Any, Any] = np.concatenate(
            [self._data.qpos, self._data.qvel]
        ).astype(np.float32)
        return out


class NamaqualandScanTask:
    """Single-cell Task wrapping the Namaqualand Boulder 05 scan composed with Go1.

    The deliverable is the *ingestion pipeline*, not a model claim;
    `RandomLocomotionPolicy` is the v0 substrate baseline, expected
    to be out-of-distribution against this unfamiliar scene. The
    manifest records this framing in `Manifest.notes`.
    """

    canonical_axis_map: dict[str, CanonicalDim] = {}

    def __init__(
        self,
        *,
        scene_path: Path | str,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        boulder_pos: tuple[float, float, float] = DEFAULT_BOULDER_POS,
    ) -> None:
        self._scene_path = Path(scene_path)
        self._max_episode_steps = max_episode_steps
        self._boulder_pos = boulder_pos

    @property
    def name(self) -> str:
        return TASK_NAME

    @property
    def embodiment(self) -> str:
        return TASK_EMBODIMENT

    @property
    def n_cells(self) -> int:
        return 1

    @property
    def action_dim(self) -> int:
        return GO1_ACTION_DIM

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    def cell_id(self, cell: int) -> CellId:
        if cell != 0:
            raise IndexError(cell)
        return CellId(embodiment=TASK_EMBODIMENT, task=TASK_NAME, axes={})

    def build_env(self, cell: int) -> _ScanSceneCompatEnv:
        if cell != 0:
            raise IndexError(cell)
        try:
            import mujoco_playground  # noqa: F401  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "mujoco_playground not installed. Install with `pip install playground` "
                "(the PyPI name; the import is `mujoco_playground`). v0 reuses the v0 "
                "venv (`.venv-mjx`); CI uses `mock_namaqualand_task_factory()`."
            ) from e
        if not self._scene_path.is_file():
            raise FileNotFoundError(
                f"Boulder scene MJCF missing at {self._scene_path}. Did you run "
                "`assets/namaqualand_scan/convert.py` against `source.usdc`?"
            )
        os.environ.setdefault("MUJOCO_GL", "egl")
        go1_xml = _resolve_go1_xml()
        return _ScanSceneCompatEnv(
            scene_path=self._scene_path,
            go1_xml_path=go1_xml,
            max_episode_steps=self._max_episode_steps,
            boulder_pos=self._boulder_pos,
        )

    def instruction_for(self, env: Any) -> str:
        return "stand on the scanned boulder without falling"

    def extract_image(self, env: Any, obs: Any) -> NDArrayU8:
        render = getattr(env, "render", None)
        if callable(render):
            try:
                arr = np.asarray(render(), dtype=np.uint8)
                if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                    return arr if arr.shape[-1] == 3 else arr[..., :3]
            except Exception:
                pass
        return np.zeros((RENDER_HEIGHT, RENDER_WIDTH, 3), dtype=np.uint8)


class _MockScanEnv:
    """12-DoF joint-space env that mimics `_ScanSceneCompatEnv` for CI."""

    def __init__(self, horizon: int = 10) -> None:
        self._horizon = horizon
        self._step = 0
        self.command: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
        self._step = 0
        if seed is not None:
            np.random.default_rng(seed)
        return self._obs(), {}

    def step(
        self, action: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], float, bool, bool, dict[str, Any]]:
        if action.shape[0] != GO1_ACTION_DIM:
            raise ValueError(f"Go1 action must be {GO1_ACTION_DIM}-dim; got {action.shape}")
        self._step += 1
        stop = self._step >= self._horizon
        success = stop
        info = {
            "episode_stats": {
                "step_idx": self._step,
                "trunk_z": 0.4,
                "fall_terminated": False,
                "reached_horizon": stop,
            }
        }
        return self._obs(), 0.0, success, stop, info

    def _obs(self) -> np.ndarray[Any, Any]:
        return np.zeros(37, dtype=np.float32)

    def render(self) -> np.ndarray[Any, Any]:
        return np.full((64, 64, 3), 96, dtype=np.uint8)

    def close(self) -> None:
        return None


def mock_namaqualand_task_factory(max_episode_steps: int = 10) -> Any:
    """CI-friendly Task that bypasses MuJoCo. Same idiom as `mock_go1_task_factory`."""

    class _MockTask:
        canonical_axis_map: dict[str, CanonicalDim] = {}

        @property
        def name(self) -> str:
            return TASK_NAME

        @property
        def embodiment(self) -> str:
            return TASK_EMBODIMENT

        @property
        def n_cells(self) -> int:
            return 1

        @property
        def action_dim(self) -> int:
            return GO1_ACTION_DIM

        @property
        def max_episode_steps(self) -> int:
            return max_episode_steps

        def cell_id(self, cell: int) -> CellId:
            if cell != 0:
                raise IndexError(cell)
            return CellId(embodiment=TASK_EMBODIMENT, task=TASK_NAME, axes={})

        def build_env(self, cell: int) -> _MockScanEnv:
            if cell != 0:
                raise IndexError(cell)
            return _MockScanEnv(horizon=max_episode_steps)

        def instruction_for(self, env: Any) -> str:
            return "stand on the scanned boulder without falling"

        def extract_image(self, env: Any, obs: Any) -> NDArrayU8:
            return np.full((RENDER_HEIGHT, RENDER_WIDTH, 3), 96, dtype=np.uint8)

    return _MockTask()
