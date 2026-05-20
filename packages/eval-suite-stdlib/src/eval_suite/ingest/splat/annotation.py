"""Declarative scene annotation schemas for splat-derived environments.

**In plain words.** Once you have a scanned scene, you need to tell
the simulator some semantic facts about it: where the robot
spawns, which named region counts as "the goal," how the mesh
needs to be rotated to line up with sim gravity, which bodies (a
movable chair, a refrigerator door) should be carved out so the
robot can interact with them. This file defines the small JSON
format that captures all of that.


Two on-disk files:

  - `scene_metadata.json` — ALWAYS-WORKS layer. Carries the
    `scene_transform` (mesh-frame → MuJoCo-frame alignment),
    `NamedRegion`s (invisible sites the success predicate library can
    reference by name), and `SpawnPoint`s (robot initial poses). These
    are pure coordinate specs; loading and using them never invokes
    boolean mesh ops or any other fragile pipeline.

  - `scene_extractions.json` — FRAGILE layer. Declares `ExtractedBody`
    entries: regions of the static scene to carve out and re-emit as
    articulable MJCF bodies. Requires `manifold3d` boolean mesh ops on
    splat-derived (non-watertight) meshes, which is the highest-risk
    step in the v1 pipeline. Two files (not one) so the always-works
    semantics ship cleanly even when extraction is deferred or fails on
    a particular scene.

Both files are content-hashed into the manifest's `run_id` via
`AssetProvenance.assets[]` entries with roles `"scene_metadata"` and
`"scene_extractions"`. Modifying either file by one byte changes the
run_id of any sweep that uses it — the determinism contract for a
factory engineer who edits their goal zone or keep-out region.

Schema version 0.1.0 for both files. Future evolutions (articulated
joint kinematics, materials, lighting probes) bump these without
breaking v1 manifests because the schemas are hashed via
AssetProvenance, not via a typed Manifest field.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from eval_suite.hashing import canonical_json

__all__ = [
    "SCENE_METADATA_SCHEMA_VERSION",
    "SCENE_EXTRACTIONS_SCHEMA_VERSION",
    "SceneTransform",
    "NamedRegion",
    "SpawnPoint",
    "SceneMetadata",
    "BodyPhysics",
    "ExtractedBody",
    "SceneExtractions",
    "AnnotationSchemaError",
]

SCENE_METADATA_SCHEMA_VERSION = "0.1.0"
SCENE_EXTRACTIONS_SCHEMA_VERSION = "0.1.0"

_VALID_UP_AXIS: tuple[str, ...] = ("x", "y", "z")
_VALID_REGION_SHAPE: tuple[str, ...] = ("box", "sphere", "cylinder")
_VALID_JOINT_TYPE: tuple[str, ...] = ("free", "hinge", "slide", "fixed")
_VALID_COLLISION: tuple[str, ...] = ("convex_hull", "vhacd")


class AnnotationSchemaError(ValueError):
    """Raised by load methods when the on-disk JSON violates the schema."""


def _as_tuple3(seq: Sequence[float], *, name: str) -> tuple[float, float, float]:
    if len(seq) != 3:
        raise AnnotationSchemaError(f"{name}: expected 3 floats, got {len(seq)}")
    return (float(seq[0]), float(seq[1]), float(seq[2]))


def _as_tuple4(seq: Sequence[float], *, name: str) -> tuple[float, float, float, float]:
    if len(seq) != 4:
        raise AnnotationSchemaError(f"{name}: expected 4 floats, got {len(seq)}")
    return (float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3]))


def _as_size_tuple(seq: Sequence[float], *, shape: str) -> tuple[float, ...]:
    """`size` arity depends on `shape`: box=3, sphere=1, cylinder=2 (radius, half_height)."""
    expected = {"box": 3, "sphere": 1, "cylinder": 2}[shape]
    if len(seq) != expected:
        raise AnnotationSchemaError(
            f"size: shape='{shape}' expects {expected} floats, got {len(seq)}"
        )
    return tuple(float(x) for x in seq)


@dataclass(frozen=True)
class SceneTransform:
    """Maps mesh-frame coordinates (arbitrary, from SuGaR / Nerfstudio output)
    onto the MuJoCo world frame (Z-up, meters). Mandatory — without an
    explicit alignment, a factory user's robot spawns 47 meters above the
    floor and they conclude the substrate is broken.

    `up_axis` is the mesh-frame axis that points up. The ingest pipeline
    rotates the mesh so this axis becomes MuJoCo's +Z. `meters_per_unit`
    scales mesh-frame distances to meters. `world_origin` is the
    mesh-frame point that becomes MuJoCo's (0, 0, 0).
    """

    up_axis: Literal["x", "y", "z"]
    meters_per_unit: float
    world_origin: tuple[float, float, float]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SceneTransform:
        up = str(d.get("up_axis", ""))
        if up not in _VALID_UP_AXIS:
            raise AnnotationSchemaError(
                f"scene_transform.up_axis must be one of {_VALID_UP_AXIS}, got {up!r}"
            )
        mpu = float(d.get("meters_per_unit", 0.0))
        if mpu <= 0.0:
            raise AnnotationSchemaError(
                f"scene_transform.meters_per_unit must be > 0, got {mpu}"
            )
        origin = _as_tuple3(d["world_origin"], name="scene_transform.world_origin")
        return cls(
            up_axis=cast(Literal["x", "y", "z"], up),
            meters_per_unit=mpu,
            world_origin=origin,
        )


@dataclass(frozen=True)
class NamedRegion:
    """An invisible primitive site in the composed MJCF that success
    predicates reference by name (e.g., `RobotReachedRegion("loading_dock")`).
    `pos` and `size` are in the POST-transform MuJoCo world frame, NOT
    mesh-frame; the convert pipeline does NOT re-apply scene_transform to
    these — they're assumed already in MuJoCo coordinates."""

    name: str
    shape: Literal["box", "sphere", "cylinder"]
    pos: tuple[float, float, float]
    size: tuple[float, ...]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NamedRegion:
        name = str(d.get("name", "")).strip()
        if not name:
            raise AnnotationSchemaError("NamedRegion.name must be non-empty")
        shape = str(d.get("shape", ""))
        if shape not in _VALID_REGION_SHAPE:
            raise AnnotationSchemaError(
                f"NamedRegion.shape must be one of {_VALID_REGION_SHAPE}, got {shape!r}"
            )
        pos = _as_tuple3(d["pos"], name=f"NamedRegion[{name}].pos")
        size = _as_size_tuple(d["size"], shape=shape)
        return cls(
            name=name,
            shape=cast(Literal["box", "sphere", "cylinder"], shape),
            pos=pos,
            size=size,
        )


@dataclass(frozen=True)
class SpawnPoint:
    """A named robot initial pose. The ParametricSplatTask config picks
    one SpawnPoint by name (default `"go1_start"`)."""

    name: str
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SpawnPoint:
        name = str(d.get("name", "")).strip()
        if not name:
            raise AnnotationSchemaError("SpawnPoint.name must be non-empty")
        pos = _as_tuple3(d["pos"], name=f"SpawnPoint[{name}].pos")
        quat_raw = d.get("quat", [1.0, 0.0, 0.0, 0.0])
        quat = _as_tuple4(quat_raw, name=f"SpawnPoint[{name}].quat")
        return cls(name=name, pos=pos, quat=quat)


@dataclass(frozen=True)
class SceneMetadata:
    """Top-level `scene_metadata.json` shape."""

    schema_version: str
    scene_transform: SceneTransform
    named_regions: tuple[NamedRegion, ...] = ()
    spawn_points: tuple[SpawnPoint, ...] = ()

    def region(self, name: str) -> NamedRegion:
        for r in self.named_regions:
            if r.name == name:
                return r
        raise KeyError(f"NamedRegion {name!r} not in scene_metadata")

    def spawn(self, name: str) -> SpawnPoint:
        for s in self.spawn_points:
            if s.name == name:
                return s
        raise KeyError(f"SpawnPoint {name!r} not in scene_metadata")

    def to_canonical_json(self) -> str:
        return canonical_json(_dataclass_to_jsonable(self))

    def save(self, path: Path) -> None:
        Path(path).write_text(self.to_canonical_json())

    @classmethod
    def load(cls, path: Path) -> SceneMetadata:
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SceneMetadata:
        schema = str(d.get("schema_version", ""))
        if schema != SCENE_METADATA_SCHEMA_VERSION:
            # We don't refuse — future minor bumps add fields. Just record.
            # If a major-incompatible version appears, downstream consumers
            # will refuse via their own checks.
            pass
        transform = SceneTransform.from_dict(d["scene_transform"])
        regions = tuple(
            NamedRegion.from_dict(r) for r in d.get("named_regions", [])
        )
        spawns = tuple(SpawnPoint.from_dict(s) for s in d.get("spawn_points", []))
        return cls(
            schema_version=schema or SCENE_METADATA_SCHEMA_VERSION,
            scene_transform=transform,
            named_regions=regions,
            spawn_points=spawns,
        )


@dataclass(frozen=True)
class BodyPhysics:
    """Physical attributes for an extracted body."""

    mass: float
    joint_type: Literal["free", "hinge", "slide", "fixed"]
    joint_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    joint_range: tuple[float, float] | None = None  # used by hinge/slide only

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BodyPhysics:
        mass = float(d.get("mass", 0.0))
        if mass < 0.0:
            raise AnnotationSchemaError(f"BodyPhysics.mass must be >= 0, got {mass}")
        jt = str(d.get("joint_type", ""))
        if jt not in _VALID_JOINT_TYPE:
            raise AnnotationSchemaError(
                f"BodyPhysics.joint_type must be one of {_VALID_JOINT_TYPE}, got {jt!r}"
            )
        axis_raw = d.get("joint_axis", [0.0, 0.0, 1.0])
        axis = _as_tuple3(axis_raw, name="BodyPhysics.joint_axis")
        jr_raw = d.get("joint_range")
        joint_range: tuple[float, float] | None
        if jr_raw is None:
            joint_range = None
        else:
            if len(jr_raw) != 2:
                raise AnnotationSchemaError(
                    f"BodyPhysics.joint_range must be 2 floats, got {len(jr_raw)}"
                )
            joint_range = (float(jr_raw[0]), float(jr_raw[1]))
        return cls(
            mass=mass,
            joint_type=cast(Literal["free", "hinge", "slide", "fixed"], jt),
            joint_axis=axis,
            joint_range=joint_range,
        )


@dataclass(frozen=True)
class ExtractedBody:
    """A region to carve out of the static scene mesh and re-emit as an
    articulable MJCF body."""

    name: str
    extraction_bounds: NamedRegion
    physics: BodyPhysics
    collision: Literal["convex_hull", "vhacd"] = "convex_hull"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExtractedBody:
        name = str(d.get("name", "")).strip()
        if not name:
            raise AnnotationSchemaError("ExtractedBody.name must be non-empty")
        bounds = NamedRegion.from_dict(d["extraction_bounds"])
        physics = BodyPhysics.from_dict(d["physics"])
        collision = str(d.get("collision", "convex_hull"))
        if collision not in _VALID_COLLISION:
            raise AnnotationSchemaError(
                f"ExtractedBody.collision must be one of {_VALID_COLLISION}, got {collision!r}"
            )
        return cls(
            name=name,
            extraction_bounds=bounds,
            physics=physics,
            collision=cast(Literal["convex_hull", "vhacd"], collision),
        )


@dataclass(frozen=True)
class SceneExtractions:
    """Top-level `scene_extractions.json` shape."""

    schema_version: str
    extracted_bodies: tuple[ExtractedBody, ...] = ()

    def body(self, name: str) -> ExtractedBody:
        for b in self.extracted_bodies:
            if b.name == name:
                return b
        raise KeyError(f"ExtractedBody {name!r} not in scene_extractions")

    def to_canonical_json(self) -> str:
        return canonical_json(_dataclass_to_jsonable(self))

    def save(self, path: Path) -> None:
        Path(path).write_text(self.to_canonical_json())

    @classmethod
    def load(cls, path: Path) -> SceneExtractions:
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SceneExtractions:
        bodies = tuple(
            ExtractedBody.from_dict(b) for b in d.get("extracted_bodies", [])
        )
        return cls(
            schema_version=str(
                d.get("schema_version", SCENE_EXTRACTIONS_SCHEMA_VERSION)
            ),
            extracted_bodies=bodies,
        )


def _dataclass_to_jsonable(obj: Any) -> Any:
    """Recursive `asdict` that turns tuples into lists for JSON. `asdict`
    already handles this; we wrap for symmetry with the `from_dict`
    methods above and to document intent."""
    return asdict(obj)
