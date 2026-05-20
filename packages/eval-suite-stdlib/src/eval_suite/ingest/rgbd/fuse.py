"""RGB-D TSDF fusion via Open3D.

Input layout (under `frames_dir`):

    color/0000.png      (HxWx3 uint8 RGB)
    color/0001.png
    ...
    depth/0000.png      (HxW uint16 depth in mm; or float32 m)
    depth/0001.png
    ...
    poses.txt           # one 4x4 row-major T_world_cam per frame (16 floats)
    intrinsics.json     # {"fx": ..., "fy": ..., "cx": ..., "cy": ..., "width": ..., "height": ..., "depth_scale": 1000.0}

Output: a `StaticMeshResult` matching what `splat_to_static_mesh`
produces, so the downstream `compose_with_annotations` consumes both
sources without modification. All input frames' SHA256s + fusion
parameters are recorded in `convert_log.json`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._mesh_utils import sha256_file
from ..splat.convert import StaticMeshResult


class Open3DUnavailableError(RuntimeError):
    """Open3D isn't installed — install with `pip install 'eval-suite-stdlib[rgbd]'`."""


@dataclass(frozen=True)
class FusionParams:
    voxel_length: float = 0.01            # meters
    sdf_trunc: float = 0.04               # meters
    depth_max: float = 3.0                # meters
    depth_scale_default: float = 1000.0   # mm → m for uint16 depth


def fuse_rgbd_to_mesh(
    *,
    frames_dir: Path,
    output_dir: Path,
    target_faces: int = 3000,
    fusion: FusionParams | None = None,
    log_path: Path | None = None,
) -> StaticMeshResult:
    """Fuse a directory of RGB-D frames into a triangle mesh, decimate,
    and emit a `StaticMeshResult` matching the splat pipeline's shape.
    """
    try:
        import open3d as o3d  # type: ignore[import-not-found]
    except ImportError as e:
        raise Open3DUnavailableError(
            "Open3D not installed. `pip install 'eval-suite-stdlib[rgbd]'`."
        ) from e

    import numpy as np

    frames_dir = Path(frames_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fusion = fusion or FusionParams()

    intrinsics_path = frames_dir / "intrinsics.json"
    poses_path = frames_dir / "poses.txt"
    color_dir = frames_dir / "color"
    depth_dir = frames_dir / "depth"
    for required in (intrinsics_path, poses_path, color_dir, depth_dir):
        if not required.exists():
            raise FileNotFoundError(f"required input missing: {required}")

    intrinsics = json.loads(intrinsics_path.read_text())
    depth_scale = float(intrinsics.get("depth_scale", fusion.depth_scale_default))
    intr = o3d.camera.PinholeCameraIntrinsic(
        width=int(intrinsics["width"]),
        height=int(intrinsics["height"]),
        fx=float(intrinsics["fx"]),
        fy=float(intrinsics["fy"]),
        cx=float(intrinsics["cx"]),
        cy=float(intrinsics["cy"]),
    )

    poses = _load_poses(poses_path)
    color_files = sorted(color_dir.glob("*.png"))
    depth_files = sorted(depth_dir.glob("*.png"))
    if not (len(color_files) == len(depth_files) == len(poses)):
        raise ValueError(
            f"frame-count mismatch: color={len(color_files)} depth={len(depth_files)} "
            f"poses={len(poses)}"
        )

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=fusion.voxel_length,
        sdf_trunc=fusion.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    t0 = time.monotonic()
    input_shas: list[dict[str, str]] = []
    for color_p, depth_p, T_wc in zip(color_files, depth_files, poses, strict=True):
        color = o3d.io.read_image(str(color_p))
        depth = o3d.io.read_image(str(depth_p))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth,
            depth_scale=depth_scale,
            depth_trunc=fusion.depth_max,
            convert_rgb_to_intensity=False,
        )
        T_cw = np.linalg.inv(T_wc)
        volume.integrate(rgbd, intr, T_cw)
        input_shas.append({
            "color": sha256_file(color_p),
            "depth": sha256_file(depth_p),
        })

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    # Decimate to the target face count.
    if target_faces > 0 and len(mesh.triangles) > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(target_faces))

    # Write visual + collision-hull OBJs in the same paths the splat pipeline uses.
    visual_obj = output_dir / "visuals" / "scene.obj"
    collision_obj = output_dir / "collision_hull" / "scene_hull.obj"
    visual_obj.parent.mkdir(parents=True, exist_ok=True)
    collision_obj.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(visual_obj), mesh, write_vertex_colors=True)
    # Convex hull as collision proxy (mirrors the splat pipeline's coacd fallback).
    hull, _ = mesh.compute_convex_hull()
    o3d.io.write_triangle_mesh(str(collision_obj), hull)

    elapsed = time.monotonic() - t0
    mesh_stats = {
        "n_vertices": len(mesh.vertices),
        "n_triangles": len(mesh.triangles),
        "n_hull_vertices": len(hull.vertices),
        "n_hull_triangles": len(hull.triangles),
    }

    static = StaticMeshResult(
        source_ply=Path(""),  # not a splat source
        source_sha256="",      # n/a; provenance is per-frame, not per-source
        visual_obj=visual_obj,
        visual_obj_sha256=sha256_file(visual_obj),
        collision_hull_obj=collision_obj,
        collision_hull_obj_sha256=sha256_file(collision_obj),
        mesh_stats=mesh_stats,
        tool=None,  # not a splat tool — mark None
        tool_args=(),
        wall_clock_s=elapsed,
    )

    if log_path is not None:
        _emit_log(log_path, intrinsics, fusion, input_shas, mesh_stats, elapsed)

    return static


def _load_poses(path: Path) -> list[Any]:
    """Parse a poses.txt where each non-empty, non-comment line is 16
    floats forming a row-major 4x4 T_world_cam."""
    import numpy as np

    poses = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        vals = [float(x) for x in line.split()]
        if len(vals) != 16:
            raise ValueError(
                f"pose line must have 16 floats (4x4 row-major), got {len(vals)} in {path}"
            )
        poses.append(np.asarray(vals, dtype=np.float64).reshape(4, 4))
    return poses


def _emit_log(
    log_path: Path,
    intrinsics: dict[str, Any],
    fusion: FusionParams,
    input_shas: list[dict[str, str]],
    mesh_stats: dict[str, int],
    wall_clock_s: float,
) -> None:
    payload = {
        "phase": "rgbd_fusion",
        "tool": "open3d.pipelines.integration.ScalableTSDFVolume",
        "intrinsics": intrinsics,
        "fusion_params": {
            "voxel_length": fusion.voxel_length,
            "sdf_trunc": fusion.sdf_trunc,
            "depth_max": fusion.depth_max,
        },
        "input_frames": input_shas,
        "mesh_stats": mesh_stats,
        "wall_clock_s": wall_clock_s,
    }
    existing = []
    if log_path.exists():
        try:
            existing_raw = json.loads(log_path.read_text())
            if isinstance(existing_raw, list):
                existing = existing_raw
        except json.JSONDecodeError:
            pass
    existing.append(payload)
    log_path.write_text(json.dumps(existing, indent=2, sort_keys=True))
