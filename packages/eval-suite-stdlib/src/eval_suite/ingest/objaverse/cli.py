"""`python -m eval_suite.ingest objaverse <uid> --output-dir <dir> ...` CLI.

**In plain words.** The user-facing command that runs the Objaverse
fetch + convert. One asset ID in, one MJCF scene out.


Fetches one Objaverse-XL asset, converts the .glb to OBJ via trimesh,
and emits the MJCF artifact shape via the existing splat-pipeline
`compose_with_annotations` (so `ParametricSplatTask` consumes the
output unchanged).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .._mesh_utils import sha256_file
from ..splat.convert import StaticMeshResult


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m eval_suite.ingest objaverse",
        description="Objaverse-XL asset → MJCF.",
    )
    p.add_argument("uid", help="Objaverse asset uid.")
    p.add_argument("--scene-metadata", type=Path, required=True)
    p.add_argument("--scene-extractions", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--body-name", type=str, default="objaverse_asset")
    p.add_argument(
        "--license-allowlist",
        default="CC-BY,CC-BY-4.0,CC0,CC0-1.0",
        help="Comma-separated allowed license strings (default: CC-BY / CC0).",
    )
    p.add_argument("--cache-dir", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    from ..splat.annotation import SceneExtractions, SceneMetadata
    from ..splat.convert import compose_with_annotations
    from .fetch import (
        AssetLicenseRefusedError,
        ObjaverseUnavailableError,
        fetch_asset,
    )

    allowlist = tuple(s.strip() for s in args.license_allowlist.split(",") if s.strip())
    try:
        fetched = fetch_asset(uid=args.uid, license_allowlist=allowlist, cache_dir=args.cache_dir)
    except ObjaverseUnavailableError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except AssetLicenseRefusedError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3

    scene_metadata = SceneMetadata.load(args.scene_metadata)
    scene_extractions: SceneExtractions | None = None
    if args.scene_extractions is not None:
        scene_extractions = SceneExtractions.load(args.scene_extractions)

    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "convert_log.json"

    # Convert the fetched mesh (.glb / .obj / .ply) to an OBJ pair using trimesh.
    static = _convert_to_static_mesh(
        fetched.local_path, output_dir=output_dir, body_name=args.body_name,
    )

    # Record provenance in the same convert_log.json the splat path uses.
    _emit_log(
        log_path,
        uid=fetched.uid,
        local_path=fetched.local_path,
        license=fetched.license,
        metadata=fetched.metadata,
        mesh_stats=static.mesh_stats,
    )

    composed = compose_with_annotations(
        static,
        scene_metadata=scene_metadata,
        scene_extractions=scene_extractions,
        output_dir=output_dir,
        log_path=log_path,
        body_name_for_static=args.body_name,
    )

    summary = {
        "source": "objaverse",
        "uid": fetched.uid,
        "license": fetched.license,
        "static_mesh": str(static.visual_obj),
        "collision_hull": str(static.collision_hull_obj),
        "composed_mjcf": str(composed.composed_mjcf),
        "convert_log": str(log_path),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _convert_to_static_mesh(
    source: Path, *, output_dir: Path, body_name: str,
) -> StaticMeshResult:
    """Read source (GLB/OBJ/PLY) with trimesh, write visual + hull OBJs."""
    try:
        import trimesh
    except ImportError as e:
        raise RuntimeError(
            "trimesh required for the Objaverse path. Already a [splat] dep."
        ) from e

    visual_obj = output_dir / "visuals" / "scene.obj"
    collision_obj = output_dir / "collision_hull" / "scene_hull.obj"
    visual_obj.parent.mkdir(parents=True, exist_ok=True)
    collision_obj.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    loaded: Any = trimesh.load(source, force="mesh")
    if hasattr(loaded, "geometry") and loaded.geometry:
        # Multi-mesh scene: concatenate.
        loaded = trimesh.util.concatenate(list(loaded.geometry.values()))
    mesh: Any = loaded
    mesh.export(visual_obj)
    hull: Any = mesh.convex_hull
    hull.export(collision_obj)
    elapsed = time.monotonic() - t0
    mesh_stats = {
        "n_vertices": int(len(mesh.vertices)),
        "n_triangles": int(len(mesh.faces)),
        "n_hull_vertices": int(len(hull.vertices)),
        "n_hull_triangles": int(len(hull.faces)),
    }
    return StaticMeshResult(
        source_ply=source,
        source_sha256=sha256_file(source),
        visual_obj=visual_obj,
        visual_obj_sha256=sha256_file(visual_obj),
        collision_hull_obj=collision_obj,
        collision_hull_obj_sha256=sha256_file(collision_obj),
        mesh_stats=mesh_stats,
        tool=None,
        tool_args=(),
        wall_clock_s=elapsed,
    )


def _emit_log(
    log_path: Path,
    *,
    uid: str,
    local_path: Path,
    license: str,
    metadata: dict[str, str],
    mesh_stats: dict[str, int],
) -> None:
    payload = {
        "phase": "objaverse_fetch",
        "uid": uid,
        "source_path": str(local_path),
        "source_sha256": sha256_file(local_path),
        "license": license,
        "metadata": metadata,
        "mesh_stats": mesh_stats,
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


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
