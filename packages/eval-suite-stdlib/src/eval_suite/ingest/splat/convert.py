"""Splat → static-mesh → composed-MJCF conversion pipeline.

Two functions:

  - `splat_to_static_mesh(splat_ply, output_dir, target_faces=3000)` —
    subprocess to SuGaR (preferred) or `ns-export poisson` (Nerfstudio
    fallback). Decimates the resulting mesh and writes a convex hull.
    Records subprocess commit SHAs, args, and wall-clock into
    `convert_log.json`.

  - `compose_with_annotations(static_mesh_result, scene_metadata,
    scene_extractions=None) -> ComposedMjcfResult` — applies the
    `scene_transform` (mesh-frame → MuJoCo-frame), carves declared
    `ExtractedBody` regions out of the static mesh via
    `extract.cut_region_from_mesh`, and emits the final composed MJCF
    with named-region sites baked in.

Heavy subprocess invocations (SuGaR, Nerfstudio) are NOT exercised by
the CI test suite — they require GPU and large training-data fixtures.
The structural Python around them is unit-tested. The
`test_cut_region_on_real_tnt_truck_mesh` release gate is the manual
validation that this all works end-to-end on a real splat.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .._mesh_utils import (
    decimate_and_hull,
    sha256_file,
    sorted_serialize,
    tool_versions,
)
from .annotation import (
    SceneExtractions,
    SceneMetadata,
)
from .extract import (
    ExtractionReport,
    cut_region_from_mesh,
    repair_mesh,
)

__all__ = [
    "SplatToolUnavailableError",
    "StaticMeshResult",
    "ComposedMjcfResult",
    "splat_to_static_mesh",
    "compose_with_annotations",
    "apply_scene_transform_to_mesh",
]


class SplatToolUnavailableError(RuntimeError):
    """Neither SuGaR nor `ns-export poisson` is available on PATH."""


@dataclass(frozen=True)
class StaticMeshResult:
    """Output of `splat_to_static_mesh`."""

    source_ply: Path
    source_sha256: str
    visual_obj: Path
    visual_obj_sha256: str
    collision_hull_obj: Path
    collision_hull_obj_sha256: str
    mesh_stats: dict[str, int] = field(default_factory=dict)
    tool: Literal["sugar", "nerfstudio_poisson"] | None = None
    tool_args: tuple[str, ...] = ()
    wall_clock_s: float = 0.0


@dataclass(frozen=True)
class ComposedMjcfResult:
    """Output of `compose_with_annotations`."""

    composed_mjcf: Path
    composed_mjcf_sha256: str
    extracted_bodies: tuple[tuple[str, Path, str], ...] = ()  # (name, obj_path, sha256)
    extraction_reports: tuple[ExtractionReport, ...] = ()


def _which(cmd: str) -> str | None:
    return shutil.which(cmd)


def _ts_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _git_rev(repo: Path | None) -> str:
    if repo is None or not Path(repo).is_dir():
        return "unknown"
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def splat_to_static_mesh(
    splat_ply: Path,
    output_dir: Path,
    *,
    target_faces: int = 3000,
    sugar_repo: Path | None = None,
    prefer_tool: Literal["sugar", "nerfstudio_poisson"] | None = None,
    log_path: Path | None = None,
) -> StaticMeshResult:
    """Run the splat → static mesh pipeline.

    Tries SuGaR first (preferred surface quality on splat input). Falls
    back to `ns-export poisson` if SuGaR is not on PATH or fails. Raises
    `SplatToolUnavailableError` if neither is available.

    Writes `visuals/scene.obj` and `collision_hull/scene_hull.obj`
    under `output_dir`. Returns the per-file SHA256s + mesh stats.
    """
    splat_ply = Path(splat_ply).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not splat_ply.is_file():
        raise FileNotFoundError(f"splat .ply not found: {splat_ply}")

    visual_obj = output_dir / "visuals" / "scene.obj"
    hull_obj = output_dir / "collision_hull" / "scene_hull.obj"
    visual_obj.parent.mkdir(parents=True, exist_ok=True)
    hull_obj.parent.mkdir(parents=True, exist_ok=True)

    sugar_available = _which("sugar") is not None or (
        sugar_repo and (Path(sugar_repo) / "scripts" / "run_sugar.py").is_file()
    )
    ns_available = _which("ns-export") is not None
    use_sugar = prefer_tool == "sugar" or (prefer_tool is None and sugar_available)
    if not use_sugar and not ns_available:
        if not sugar_available:
            raise SplatToolUnavailableError(
                "Neither SuGaR (sugar) nor Nerfstudio (ns-export) is on PATH. "
                "Install one of: "
                "https://github.com/Anttwo/SuGaR or "
                "pip install nerfstudio."
            )
        use_sugar = True

    start = time.monotonic()
    tool_args: tuple[str, ...]
    if use_sugar:
        tool_name: Literal["sugar", "nerfstudio_poisson"] = "sugar"
        tool_args = (
            "sugar",
            "extract-mesh",
            "--input", str(splat_ply),
            "--output", str(visual_obj),
        )
    else:
        tool_name = "nerfstudio_poisson"
        tool_args = (
            "ns-export",
            "poisson",
            "--load-config", str(splat_ply.with_suffix(".yml")),
            "--output-dir", str(visual_obj.parent),
        )

    env = {**os.environ, "LC_ALL": "C"}
    completed = subprocess.run(
        list(tool_args), env=env, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{tool_name} subprocess failed (exit {completed.returncode}). "
            f"stderr tail: {completed.stderr[-1024:]}"
        )

    if not visual_obj.is_file():
        # Some tools write under a different name; auto-detect by walking.
        candidates = list(visual_obj.parent.rglob("*.obj"))
        if not candidates:
            raise RuntimeError(
                f"{tool_name} did not produce a .obj under {visual_obj.parent}"
            )
        # Prefer the largest candidate.
        candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
        candidates[0].rename(visual_obj)

    mesh_stats = decimate_and_hull(visual_obj, hull_obj, target_faces=target_faces)
    wall = time.monotonic() - start

    src_sha = sha256_file(splat_ply)
    vis_sha = sha256_file(visual_obj)
    hull_sha = sha256_file(hull_obj)

    result = StaticMeshResult(
        source_ply=splat_ply,
        source_sha256=src_sha,
        visual_obj=visual_obj,
        visual_obj_sha256=vis_sha,
        collision_hull_obj=hull_obj,
        collision_hull_obj_sha256=hull_sha,
        mesh_stats=mesh_stats,
        tool=tool_name,
        tool_args=tool_args,
        wall_clock_s=round(wall, 3),
    )

    if log_path is not None:
        _write_convert_log(
            log_path,
            phase="static_mesh",
            result=result,
            sugar_repo=sugar_repo,
        )

    return result


def apply_scene_transform_to_mesh(mesh: Any, transform: Any) -> Any:
    """Apply a SceneTransform's up-axis remap + scale + translation to a
    trimesh.Trimesh. Returns a new mesh (does not mutate input).
    """
    import numpy as np
    import trimesh

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"apply_scene_transform_to_mesh expects trimesh.Trimesh, got {type(mesh)!r}")

    out = mesh.copy()
    # 1. Up-axis remap to Z-up.
    if transform.up_axis == "x":
        rot = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
    elif transform.up_axis == "y":
        rot = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
    else:  # "z"
        rot = np.eye(3, dtype=float)
    transform4 = np.eye(4, dtype=float)
    transform4[:3, :3] = rot
    out.apply_transform(transform4)
    # 2. Scale to meters.
    if transform.meters_per_unit != 1.0:
        scale4 = np.eye(4, dtype=float)
        for i in range(3):
            scale4[i, i] = transform.meters_per_unit
        out.apply_transform(scale4)
    # 3. World origin translation.
    out.apply_translation(-np.asarray(transform.world_origin, dtype=float) * transform.meters_per_unit)
    return out


def compose_with_annotations(
    static_mesh: StaticMeshResult,
    scene_metadata: SceneMetadata,
    scene_extractions: SceneExtractions | None,
    output_dir: Path,
    *,
    log_path: Path | None = None,
    body_name_for_static: str = "splat_scene",
) -> ComposedMjcfResult:
    """Apply the scene_transform, carve extracted bodies, and emit the
    composed MJCF.

    Output layout under `output_dir`:
      - `MJCF/scene.xml` — composed MJCF (Go1 is attached later, at sweep time)
      - `visuals/{body_name}.obj` — transformed static visual mesh
      - `visuals/extracted_<name>.obj` — per-extracted-body visual mesh
      - `collision_hull/{body_name}_hull.obj` — transformed static collision hull
      - `collision_hull/extracted_<name>_hull.obj` — per-extracted-body hull

    If `scene_extractions` is None, the static mesh is emitted whole
    (no boolean ops), but named-region sites + spawn-point sites are
    still baked into the MJCF.
    """
    import trimesh

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = output_dir / "visuals"
    hulls_dir = output_dir / "collision_hull"
    mjcf_dir = output_dir / "MJCF"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    hulls_dir.mkdir(parents=True, exist_ok=True)
    mjcf_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load + transform the static mesh.
    static = trimesh.load(static_mesh.visual_obj, force="mesh")
    static = apply_scene_transform_to_mesh(static, scene_metadata.scene_transform)
    repaired_static, _repair_report = repair_mesh(static)

    extracted_bodies: list[tuple[str, Path, str]] = []
    extraction_reports: list[ExtractionReport] = []

    residual = repaired_static
    if scene_extractions is not None:
        for body in scene_extractions.extracted_bodies:
            residual, extracted_mesh, report = cut_region_from_mesh(
                residual, body.extraction_bounds
            )
            extraction_reports.append(report)
            body_vis = visuals_dir / f"extracted_{body.name}.obj"
            body_hull = hulls_dir / f"extracted_{body.name}_hull.obj"
            extracted_mesh.export(body_vis)
            extracted_mesh.convex_hull.export(body_hull)
            extracted_bodies.append(
                (body.name, body_vis, sha256_file(body_vis))
            )

    # 2. Write residual visual + hull.
    static_visual_out = visuals_dir / f"{body_name_for_static}.obj"
    static_hull_out = hulls_dir / f"{body_name_for_static}_hull.obj"
    residual.export(static_visual_out)
    residual.convex_hull.export(static_hull_out)

    # 3. Emit composed MJCF.
    composed_mjcf = mjcf_dir / "scene.xml"
    _emit_composed_mjcf(
        composed_mjcf,
        body_name_for_static=body_name_for_static,
        static_visual_rel=os.path.relpath(static_visual_out, mjcf_dir),
        static_hull_rel=os.path.relpath(static_hull_out, mjcf_dir),
        extracted_bodies=[
            (name, os.path.relpath(p, mjcf_dir),
             os.path.relpath(hulls_dir / f"extracted_{name}_hull.obj", mjcf_dir))
            for (name, p, _sha) in extracted_bodies
        ],
        scene_metadata=scene_metadata,
        scene_extractions=scene_extractions,
    )

    result = ComposedMjcfResult(
        composed_mjcf=composed_mjcf,
        composed_mjcf_sha256=sha256_file(composed_mjcf),
        extracted_bodies=tuple(extracted_bodies),
        extraction_reports=tuple(extraction_reports),
    )

    if log_path is not None:
        _write_convert_log(
            log_path,
            phase="compose",
            result=result,
            extra={
                "scene_transform": asdict(scene_metadata.scene_transform),
                "n_named_regions": len(scene_metadata.named_regions),
                "n_spawn_points": len(scene_metadata.spawn_points),
                "n_extractions": (
                    len(scene_extractions.extracted_bodies) if scene_extractions else 0
                ),
            },
        )

    return result


def _emit_composed_mjcf(
    out: Path,
    *,
    body_name_for_static: str,
    static_visual_rel: str,
    static_hull_rel: str,
    extracted_bodies: list[tuple[str, str, str]],  # (name, vis_rel, hull_rel)
    scene_metadata: SceneMetadata,
    scene_extractions: SceneExtractions | None,
) -> None:
    """Emit the MJCF that the ParametricSplatTask attaches Go1 to."""
    root = ET.Element("mujoco", {"model": f"{body_name_for_static}_composed"})
    ET.SubElement(root, "compiler", {"angle": "radian", "coordinate": "local"})
    ET.SubElement(root, "option", {"integrator": "RK4", "timestep": "0.01"})

    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh",
                  {"name": f"{body_name_for_static}_visual", "file": static_visual_rel})
    ET.SubElement(asset, "mesh",
                  {"name": f"{body_name_for_static}_collision", "file": static_hull_rel})
    for (name, vis_rel, hull_rel) in extracted_bodies:
        ET.SubElement(asset, "mesh", {"name": f"extracted_{name}_visual", "file": vis_rel})
        ET.SubElement(asset, "mesh",
                      {"name": f"extracted_{name}_collision", "file": hull_rel})

    wb = ET.SubElement(root, "worldbody")
    # Light is added per-cell by the ParametricSplatTask; this is a default
    # baseline so the MJCF loads standalone for inspection.
    ET.SubElement(wb, "light",
                  {"pos": "0 0 5", "dir": "0 0 -1", "directional": "true",
                   "name": "default_light"})

    # Static residual body.
    static_body = ET.SubElement(wb, "body", {"name": body_name_for_static, "pos": "0 0 0"})
    ET.SubElement(static_body, "geom",
                  {"name": f"{body_name_for_static}_vis",
                   "mesh": f"{body_name_for_static}_visual", "type": "mesh", "group": "1",
                   "contype": "0", "conaffinity": "0"})
    ET.SubElement(static_body, "geom",
                  {"name": f"{body_name_for_static}_col",
                   "mesh": f"{body_name_for_static}_collision", "type": "mesh", "group": "0"})

    # Extracted bodies (separate, with their own joint / mass).
    if scene_extractions is not None:
        for body in scene_extractions.extracted_bodies:
            xb = ET.SubElement(
                wb,
                "body",
                {"name": body.name, "pos": _pos_str(body.extraction_bounds.pos)},
            )
            if body.physics.joint_type == "free":
                ET.SubElement(xb, "freejoint", {"name": f"{body.name}_freejoint"})
            elif body.physics.joint_type in ("hinge", "slide"):
                jattr = {
                    "name": f"{body.name}_joint",
                    "type": body.physics.joint_type,
                    "axis": " ".join(f"{x}" for x in body.physics.joint_axis),
                }
                if body.physics.joint_range is not None:
                    jattr["range"] = " ".join(f"{x}" for x in body.physics.joint_range)
                ET.SubElement(xb, "joint", jattr)
            # "fixed" → no joint; body is welded into the world frame at pos.
            ET.SubElement(
                xb,
                "inertial",
                {
                    "pos": "0 0 0",
                    "mass": f"{body.physics.mass}",
                    "diaginertia": "0.1 0.1 0.1",
                },
            )
            ET.SubElement(xb, "geom",
                          {"name": f"{body.name}_vis",
                           "mesh": f"extracted_{body.name}_visual",
                           "type": "mesh", "group": "1",
                           "contype": "0", "conaffinity": "0"})
            ET.SubElement(xb, "geom",
                          {"name": f"{body.name}_col",
                           "mesh": f"extracted_{body.name}_collision",
                           "type": "mesh", "group": "0"})

    # Named-region sites: invisible primitives the predicate library looks up by name.
    for region in scene_metadata.named_regions:
        ET.SubElement(
            wb,
            "site",
            {
                "name": f"region_{region.name}",
                "type": region.shape,
                "pos": _pos_str(region.pos),
                "size": " ".join(f"{x}" for x in region.size),
                "group": "4",  # group 4 = invisible by default
                "rgba": "0 1 0 0.2",
            },
        )

    # Spawn-point sites: same idea, used by the task to position Go1.
    for spawn in scene_metadata.spawn_points:
        ET.SubElement(
            wb,
            "site",
            {
                "name": f"spawn_{spawn.name}",
                "type": "sphere",
                "pos": _pos_str(spawn.pos),
                "size": "0.05",
                "group": "4",
                "rgba": "1 0 0 0.4",
            },
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('<?xml version="1.0" encoding="utf-8"?>\n' + sorted_serialize(root) + "\n")


def _pos_str(p: tuple[float, float, float]) -> str:
    return " ".join(f"{x}" for x in p)


def _write_convert_log(
    log_path: Path,
    *,
    phase: str,
    result: Any,
    sugar_repo: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append a phase entry to `convert_log.json`. Idempotent: first call
    creates the file with a fresh `phases` list; subsequent calls append.
    """
    log_path = Path(log_path)
    existing: dict[str, Any] = {}
    if log_path.is_file():
        try:
            existing = json.loads(log_path.read_text())
        except Exception:
            existing = {}
    phases: list[dict[str, Any]] = list(existing.get("phases", []))

    phase_entry: dict[str, Any] = {
        "phase": phase,
        "iso": _ts_now(),
        "tool_versions": tool_versions(),
        "sugar_commit": _git_rev(sugar_repo) if sugar_repo else "unknown",
        "result": _result_to_dict(result),
    }
    if extra:
        phase_entry["extra"] = extra
    phases.append(phase_entry)
    log_path.write_text(json.dumps({"phases": phases}, indent=2, sort_keys=True, default=str))


def _result_to_dict(result: Any) -> dict[str, Any]:
    """Convert a StaticMeshResult or ComposedMjcfResult to a dict suitable
    for JSON. Path objects become strings; tuples become lists."""
    if hasattr(result, "__dataclass_fields__"):
        d = asdict(result)
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
        return d
    return {"value": str(result)}
