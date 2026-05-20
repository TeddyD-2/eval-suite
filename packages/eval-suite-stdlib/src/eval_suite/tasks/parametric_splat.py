"""ParametricSplatTask — v1 splat-derived trial-generation prototype.

A generic `Task` driving a splat-converted MJCF scene with declarative
axes (lighting × camera × ...) and a declarative success predicate.
Each (axis-tuple, predicate) combination is one cell; `n_cells` is the
product of axis lengths. The predicate's `to_dict()` flows into the
manifest's `success_criterion` field (schema 0.3.0), so changing the
goal on the same scene produces a distinct `run_id`.

Per EXTENSION.md §7, the v1 demo: TNT Truck splat with 3 lighting × 3
camera = 9 cells and `RobotReachedRegion("behind_truck", 0.5)` as the
success predicate. The 9 cells then evaluate "did Go1 reach behind the
truck under variant lighting/camera" — exercising the annotation
substrate end-to-end on real-world-captured geometry.

The real env (`_ParametricSplatEnv`) requires MuJoCo and a converted
MJCF on disk; CI uses `_MockSplatEnv` via `mock_tnt_truck_splat_task_factory`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Any

import numpy as np

from .._types import CanonicalDim, CellId, NDArrayU8
from ..ingest.splat.annotation import SceneMetadata
from ._parametric_variants import (
    STANDARD_CAMERAS_3,
    STANDARD_LIGHTING_3,
    CameraVariant,
    LightingVariant,
)
from ._success_predicates import (
    EnvState,
    RobotReachedRegion,
    SuccessPredicate,
)
from .usd_scan import _lookat_quat, _resolve_go1_xml

__all__ = [
    "ParametricSplatTaskConfig",
    "ParametricSplatTask",
    "tnt_truck_splat_task_factory",
    "mock_tnt_truck_splat_task_factory",
]


TNT_TRUCK_TASK_NAME = "tnt_truck_splat_v01"
DEFAULT_EMBODIMENT = "unitree_go1"
DEFAULT_MAX_EPISODE_STEPS = 200
GO1_ACTION_DIM = 12
FALL_HEIGHT_THRESHOLD = 0.15
TRUNK_BODY_NAME = "trunk"
RENDER_HEIGHT = 240
RENDER_WIDTH = 320


@dataclass(frozen=True)
class ParametricSplatTaskConfig:
    """All knobs for one ParametricSplatTask instance.

    `axes` is an ordered dict-like; cell decode uses Python's insertion
    order, so don't pass an unordered dict in if order matters for the
    cell index. The mixed-radix decode keeps cell_id determinism: cell `i`
    always maps to the same (axis: variant) tuple for a given config.

    `canonical_axis_map` declares how each axis projects onto the four
    `CanonicalDim` enum members — the substrate's auditable taxonomy
    (per EXTENSION.md §3).
    """

    name: str
    scene_dir: Path
    embodiment: str
    max_episode_steps: int
    axes: dict[str, list[Any]]
    success_predicate: SuccessPredicate
    canonical_axis_map: dict[str, CanonicalDim]
    spawn_point_name: str = "go1_start"
    instruction_template: str = "reach the {region} in the scanned scene"


class ParametricSplatTask:
    """Task Protocol impl for parametric splat-derived scenes."""

    def __init__(self, config: ParametricSplatTaskConfig) -> None:
        if not config.axes:
            raise ValueError("ParametricSplatTaskConfig.axes must be non-empty")
        for axis_name, levels in config.axes.items():
            if not levels:
                raise ValueError(
                    f"ParametricSplatTaskConfig.axes[{axis_name!r}] is empty; "
                    "every axis needs at least one variant level"
                )
        self._config = config

    # --- Task Protocol surface ----------------------------------------------

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def embodiment(self) -> str:
        return self._config.embodiment

    @property
    def n_cells(self) -> int:
        return reduce(mul, (len(v) for v in self._config.axes.values()), 1)

    @property
    def max_episode_steps(self) -> int:
        return self._config.max_episode_steps

    @property
    def action_dim(self) -> int:
        # Go1 is the v1 demo embodiment. Custom embodiments would override.
        return GO1_ACTION_DIM

    @property
    def canonical_axis_map(self) -> dict[str, CanonicalDim]:
        return dict(self._config.canonical_axis_map)

    @property
    def success_criterion(self) -> dict[str, Any]:
        """Read by `sweep.run_sweep` and bound into Manifest.success_criterion.

        Returns the same dict on every call (the predicate's params are
        frozen at config-construction time). This is what makes the
        scene-and-predicate pair content-address into the manifest's
        run_id — change the predicate, change the run_id.
        """
        return dict(self._config.success_predicate.to_dict())

    def cell_id(self, cell: int) -> CellId:
        if cell < 0 or cell >= self.n_cells:
            raise IndexError(
                f"cell {cell} out of range [0, {self.n_cells})"
            )
        axes_str: dict[str, str] = {}
        i = cell
        # Mixed-radix decode: leftmost axis is the slowest-varying.
        # (We iterate over reversed-pair list so the LAST axis varies fastest.)
        for axis_name, levels in reversed(list(self._config.axes.items())):
            n = len(levels)
            variant = levels[i % n]
            axes_str[axis_name] = _variant_name_or_str(variant)
            i //= n
        return CellId(
            embodiment=self._config.embodiment,
            task=self._config.name,
            axes=axes_str,
        )

    def _cell_variants(self, cell: int) -> dict[str, Any]:
        """Same decode as cell_id but returns the actual variant OBJECTS
        instead of name strings. Used by build_env."""
        if cell < 0 or cell >= self.n_cells:
            raise IndexError(cell)
        out: dict[str, Any] = {}
        i = cell
        for axis_name, levels in reversed(list(self._config.axes.items())):
            n = len(levels)
            out[axis_name] = levels[i % n]
            i //= n
        return out

    def build_env(self, cell: int) -> Any:
        import importlib.util

        variants = self._cell_variants(cell)
        if importlib.util.find_spec("mujoco_playground") is None:
            raise ImportError(
                "mujoco_playground not installed. Install with "
                "`pip install playground`. CI uses "
                "mock_tnt_truck_splat_task_factory() instead."
            )
        scene_mjcf = self._config.scene_dir / "MJCF" / "scene.xml"
        if not scene_mjcf.is_file():
            raise FileNotFoundError(
                f"Composed scene MJCF missing at {scene_mjcf}. Run "
                "`python -m eval_suite.ingest.splat ingest ...` against the splat first."
            )
        scene_metadata = SceneMetadata.load(self._config.scene_dir / "scene_metadata.json")
        os.environ.setdefault("MUJOCO_GL", "egl")
        return _ParametricSplatEnv(
            scene_mjcf=scene_mjcf,
            go1_xml_path=_resolve_go1_xml(),
            scene_metadata=scene_metadata,
            lighting=_extract_variant(variants, LightingVariant),
            camera=_extract_variant(variants, CameraVariant),
            success_predicate=self._config.success_predicate,
            spawn_point_name=self._config.spawn_point_name,
            max_episode_steps=self._config.max_episode_steps,
        )

    def instruction_for(self, env: Any) -> str:
        # If the predicate is region-based, fill {region} with the name.
        crit = self._config.success_predicate.to_dict()
        region = crit.get("params", {}).get("region_name", "goal")
        return self._config.instruction_template.format(region=region)

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


# ---------------------------------------------------------------------------
# Real env (MuJoCo)
# ---------------------------------------------------------------------------


class _ParametricSplatEnv:
    """Gym-5-tuple env composing Go1 + a splat-converted scene with per-cell
    lighting + camera variants applied pre-compile.

    Pattern is structurally similar to `_ScanSceneCompatEnv` in
    `usd_scan.py:111-230` but the lighting/camera are NOT fixed at
    construction — they're parameters of the cell. Reunification with
    `_ScanSceneCompatEnv` is a v1.5 task; v1 keeps them separate so the
    Namaqualand 5/5 contract tests stay green without parametric
    refactoring risk.
    """

    def __init__(
        self,
        *,
        scene_mjcf: Path,
        go1_xml_path: Path,
        scene_metadata: SceneMetadata,
        lighting: LightingVariant,
        camera: CameraVariant,
        success_predicate: SuccessPredicate,
        spawn_point_name: str,
        max_episode_steps: int,
    ) -> None:
        import mujoco  # type: ignore[import-not-found]

        self._max_steps = max_episode_steps
        self._metadata = scene_metadata
        self._predicate = success_predicate

        spawn = scene_metadata.spawn(spawn_point_name)

        go1_spec = mujoco.MjSpec.from_file(str(go1_xml_path))
        scene_spec = mujoco.MjSpec.from_file(str(scene_mjcf))

        # Attach the scene to the world via a frame at the spawn position
        # offset so Go1 is placed at the spawn point.
        anchor = go1_spec.worldbody.add_frame(
            name="splat_scene_anchor",
            pos=[-spawn.pos[0], -spawn.pos[1], 0.0],  # scene moves opposite of robot spawn
        )
        go1_spec.attach(scene_spec, prefix="splat_", frame=anchor)

        floor = go1_spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = np.array([20.0, 20.0, 0.05])
        floor.rgba = np.array([0.55, 0.55, 0.55, 1.0])

        # Lighting variant.
        light = go1_spec.worldbody.add_light()
        light.name = f"cell_light_{lighting.name}"
        light.pos = np.array(lighting.pos)
        light.dir = np.array(lighting.dir)
        light.diffuse = np.array(lighting.diffuse)
        light.ambient = np.array(lighting.ambient)
        if lighting.type == "directional":
            light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        elif lighting.type == "spot":
            light.type = mujoco.mjtLightType.mjLIGHT_SPOT
        # `ambient_plus_directional` is rendered as directional + non-zero
        # ambient component on the light; MuJoCo's per-light ambient is
        # the additive contribution.

        # Camera variant.
        cam = go1_spec.worldbody.add_camera()
        cam.name = f"cell_camera_{camera.name}"
        cam.pos = np.array(camera.pos)
        if camera.quat is not None:
            cam.quat = np.array(camera.quat)
        elif camera.lookat is not None:
            cam.quat = _lookat_quat(camera.pos, camera.lookat)
        # Robot-trunk-mounted cameras would require nesting the camera
        # inside the trunk body via MjSpec.find_body — v1.5 task; v1
        # treats both mounts as world-mounted with a static pose.

        self._model = go1_spec.compile()
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)
        self._initial_qpos = self._data.qpos.copy()

        trunk_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, TRUNK_BODY_NAME)
        if trunk_id < 0:
            raise RuntimeError(
                f"trunk body '{TRUNK_BODY_NAME}' not found in composed model"
            )
        self._trunk_id = int(trunk_id)
        self._camera_name = cam.name
        self._renderer: Any | None = None
        self._step_idx = 0
        self.command: tuple[float, float, float] = (0.0, 0.0, 0.0)

        # Predicate per-episode state.
        self._predicate.reset(self._metadata)

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
        self._predicate.reset(self._metadata)
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

        xpos = self._data.xpos[self._trunk_id]
        trunk_pos: tuple[float, float, float] = (
            float(xpos[0]), float(xpos[1]), float(xpos[2])
        )
        env_state = EnvState(
            step_idx=self._step_idx,
            max_steps=self._max_steps,
            trunk_pos=trunk_pos,
        )
        outcome = self._predicate.step(env_state)
        reached_horizon = self._step_idx >= self._max_steps
        if outcome.done:
            stop = True
            success = outcome.success
        elif reached_horizon:
            stop = True
            success = self._predicate.at_horizon()
        else:
            stop = False
            success = False

        info: dict[str, Any] = {
            "episode_stats": {
                "step_idx": self._step_idx,
                "trunk_pos": list(trunk_pos),
                "predicate_done": outcome.done,
                "reached_horizon": reached_horizon,
            }
        }
        return self._obs(), 0.0, success, stop, info

    def render(self) -> NDArrayU8:
        import mujoco

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
        self._renderer.update_scene(self._data, camera=self._camera_name)
        return np.asarray(self._renderer.render(), dtype=np.uint8)

    def close(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None

    def _obs(self) -> np.ndarray[Any, Any]:
        out: np.ndarray[Any, Any] = np.concatenate([self._data.qpos, self._data.qvel]).astype(np.float32)
        return out


# ---------------------------------------------------------------------------
# Mock env (CI, no MuJoCo)
# ---------------------------------------------------------------------------


class _MockSplatEnv:
    """12-DoF joint-space env that mimics _ParametricSplatEnv for CI.

    Does NOT exercise the predicate evaluation against real geometry;
    instead it deterministically synthesizes a trunk trajectory that
    eventually enters the predicate's region (for `RobotReachedRegion`)
    or stays out (for `MaintainedClearance` / `Survived`), so the
    sweep loop sees plausible success rates.
    """

    def __init__(
        self,
        *,
        horizon: int,
        scene_metadata: SceneMetadata | None,
        success_predicate: SuccessPredicate,
        start_pos: tuple[float, float, float] = (0.0, 0.0, 0.3),
    ) -> None:
        self._horizon = horizon
        self._step_count = 0
        self._scene_metadata = scene_metadata
        self._predicate = success_predicate
        self._start = start_pos
        self._trunk_pos = start_pos
        self.command: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._predicate.reset(scene_metadata)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
        self._step_count = 0
        self._trunk_pos = self._start
        if seed is not None:
            np.random.default_rng(seed)
        self._predicate.reset(self._scene_metadata)
        return self._obs(), {}

    def step(
        self, action: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], float, bool, bool, dict[str, Any]]:
        if action.shape[0] != GO1_ACTION_DIM:
            raise ValueError(f"Go1 action must be {GO1_ACTION_DIM}-dim; got {action.shape}")
        self._step_count += 1
        # Synthesize a fake trunk trajectory that drifts toward the
        # predicate's region (if region-based), or stays put (Survived).
        target_pos = _predicate_target_position(self._predicate, fallback=self._start)
        frac = min(1.0, self._step_count / max(1, self._horizon))
        self._trunk_pos = (
            self._start[0] + frac * (target_pos[0] - self._start[0]),
            self._start[1] + frac * (target_pos[1] - self._start[1]),
            self._start[2] + frac * (target_pos[2] - self._start[2]),
        )

        env_state = EnvState(
            step_idx=self._step_count,
            max_steps=self._horizon,
            trunk_pos=self._trunk_pos,
        )
        outcome = self._predicate.step(env_state)
        reached_horizon = self._step_count >= self._horizon
        if outcome.done:
            stop = True
            success = outcome.success
        elif reached_horizon:
            stop = True
            success = self._predicate.at_horizon()
        else:
            stop = False
            success = False

        info = {
            "episode_stats": {
                "step_idx": self._step_count,
                "trunk_pos": list(self._trunk_pos),
                "predicate_done": outcome.done,
                "reached_horizon": reached_horizon,
            }
        }
        return self._obs(), 0.0, success, stop, info

    def _obs(self) -> np.ndarray[Any, Any]:
        out: np.ndarray[Any, Any] = np.zeros(37, dtype=np.float32)
        return out

    def render(self) -> np.ndarray[Any, Any]:
        out: np.ndarray[Any, Any] = np.full((64, 64, 3), 96, dtype=np.uint8)
        return out

    def close(self) -> None:
        return None


def _predicate_target_position(
    predicate: SuccessPredicate, fallback: tuple[float, float, float]
) -> tuple[float, float, float]:
    """For mock testing: returns the center of the predicate's named region
    if it has one, else the fallback position. Lets the mock env drift
    the synthetic trunk toward the predicate's target and produce
    interesting success values.

    Reads `predicate.target_position()` (optional public method, see
    `_success_predicates.SuccessPredicate`). Predicates without one
    (third-party plugins targeting an older protocol version) silently
    fall back to the start position.
    """
    target = getattr(predicate, "target_position", None)
    if callable(target):
        pos = target()
        if pos is not None:
            return (float(pos[0]), float(pos[1]), float(pos[2]))
    return fallback


# ---------------------------------------------------------------------------
# Variant helpers + factories
# ---------------------------------------------------------------------------


def _extract_variant(variants: dict[str, Any], cls: type) -> Any:
    """Find the first variant in `variants` whose type matches `cls`.
    Raises if none found — every cell decode must produce both a
    LightingVariant and a CameraVariant for the real env to compile."""
    for v in variants.values():
        if isinstance(v, cls):
            return v
    raise ValueError(
        f"No variant of type {cls.__name__} found in cell decode; "
        f"got {[type(v).__name__ for v in variants.values()]}"
    )


def _variant_name_or_str(variant: Any) -> str:
    """For cell_id axis labels: use the variant's `name` attr if present,
    else fall back to str(variant)."""
    name = getattr(variant, "name", None)
    if isinstance(name, str) and name:
        return name
    return str(variant)


def tnt_truck_splat_task_factory(
    *,
    scene_dir: Path | None = None,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    success_predicate: SuccessPredicate | None = None,
) -> ParametricSplatTask:
    """v1 demo factory: Go1 navigates the TNT Truck splat scene to a
    region behind the truck. 3 lighting × 3 camera = 9 cells.

    Default predicate is `RobotReachedRegion("behind_truck", 0.5)` — the
    9 cells then evaluate "did Go1 reach behind the truck under variant
    lighting/camera conditions." Override `success_predicate` to swap in
    `MaintainedClearance("truck", 0.3)` etc.
    """
    if scene_dir is None:
        scene_dir = (
            Path(__file__).resolve().parents[4]
            / "assets"
            / "tnt_truck_splat"
        )
    predicate = success_predicate or RobotReachedRegion(
        region_name="behind_truck", tolerance=0.5
    )
    config = ParametricSplatTaskConfig(
        name=TNT_TRUCK_TASK_NAME,
        scene_dir=Path(scene_dir),
        embodiment=DEFAULT_EMBODIMENT,
        max_episode_steps=max_episode_steps,
        axes={
            "lighting": list(STANDARD_LIGHTING_3),
            "camera": list(STANDARD_CAMERAS_3),
        },
        success_predicate=predicate,
        canonical_axis_map={"lighting": "visuals", "camera": "visuals"},
    )
    return ParametricSplatTask(config)


def mock_tnt_truck_splat_task_factory(
    *,
    max_episode_steps: int = 10,
    success_predicate: SuccessPredicate | None = None,
) -> ParametricSplatTask:
    """CI-friendly mock that bypasses MuJoCo. Same 9-cell grid as
    `tnt_truck_splat_task_factory`; `build_env` returns `_MockSplatEnv`."""
    predicate = success_predicate or RobotReachedRegion(
        region_name="behind_truck", tolerance=0.5
    )

    # Synthetic in-memory SceneMetadata for mock cells (no on-disk file needed).
    from ..ingest.splat.annotation import NamedRegion, SceneMetadata, SceneTransform, SpawnPoint

    mock_metadata = SceneMetadata(
        schema_version="0.1.0",
        scene_transform=SceneTransform(up_axis="z", meters_per_unit=1.0, world_origin=(0.0, 0.0, 0.0)),
        named_regions=(
            NamedRegion(name="behind_truck", shape="box", pos=(3.0, 0.0, 0.3), size=(0.5, 0.5, 0.3)),
        ),
        spawn_points=(
            SpawnPoint(name="go1_start", pos=(0.0, 0.0, 0.3), quat=(1.0, 0.0, 0.0, 0.0)),
        ),
    )

    config = ParametricSplatTaskConfig(
        name=TNT_TRUCK_TASK_NAME,
        scene_dir=Path("/nonexistent-mock-scene-dir"),
        embodiment=DEFAULT_EMBODIMENT,
        max_episode_steps=max_episode_steps,
        axes={
            "lighting": list(STANDARD_LIGHTING_3),
            "camera": list(STANDARD_CAMERAS_3),
        },
        success_predicate=predicate,
        canonical_axis_map={"lighting": "visuals", "camera": "visuals"},
    )

    class _MockParametricSplatTask(ParametricSplatTask):
        def build_env(self, cell: int) -> _MockSplatEnv:
            # Variants decoded for parity, but the mock env doesn't actually
            # use them — it synthesizes a fake trajectory toward the predicate's
            # region. This keeps the mock CI-friendly without MuJoCo.
            _ = self._cell_variants(cell)
            return _MockSplatEnv(
                horizon=self._config.max_episode_steps,
                scene_metadata=mock_metadata,
                success_predicate=self._config.success_predicate,
            )

        def extract_image(self, env: Any, obs: Any) -> NDArrayU8:
            return np.full((RENDER_HEIGHT, RENDER_WIDTH, 3), 96, dtype=np.uint8)

    return _MockParametricSplatTask(config)
