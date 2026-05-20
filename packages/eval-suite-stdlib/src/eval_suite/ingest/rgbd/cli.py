"""`python -m eval_suite.ingest rgbd <frames_dir> --output-dir <dir> ...` CLI.

**In plain words.** The user-facing command that runs the RGB-D
fusion. Hands the frames folder to `fuse.py` and writes the
resulting MJCF artifact alongside a `convert_log.json` so the run
is reproducible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m eval_suite.ingest rgbd",
        description=(
            "Fuse a directory of RGB-D frames into a triangle mesh, "
            "compose against scene_metadata, and emit a ParametricSplatTask-"
            "ready MJCF."
        ),
    )
    p.add_argument("frames_dir", type=Path, help="Directory of color/*, depth/*, poses.txt, intrinsics.json.")
    p.add_argument("--scene-metadata", type=Path, required=True)
    p.add_argument("--scene-extractions", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--target-faces", type=int, default=3000)
    p.add_argument("--body-name", type=str, default="rgbd_scene")
    p.add_argument("--voxel-length", type=float, default=0.01,
                   help="TSDF voxel size in meters (default 0.01).")
    p.add_argument("--sdf-trunc", type=float, default=0.04,
                   help="TSDF SDF truncation distance in meters (default 0.04).")
    p.add_argument("--depth-max", type=float, default=3.0,
                   help="Max depth in meters; further-away pixels are dropped (default 3.0).")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Lazy-import the heavy paths only after argparse has resolved.
    from ..splat.annotation import SceneExtractions, SceneMetadata
    from ..splat.convert import compose_with_annotations
    from .fuse import FusionParams, Open3DUnavailableError, fuse_rgbd_to_mesh

    scene_metadata = SceneMetadata.load(args.scene_metadata)
    scene_extractions: SceneExtractions | None = None
    if args.scene_extractions is not None:
        scene_extractions = SceneExtractions.load(args.scene_extractions)

    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "convert_log.json"

    try:
        static = fuse_rgbd_to_mesh(
            frames_dir=args.frames_dir,
            output_dir=output_dir,
            target_faces=args.target_faces,
            fusion=FusionParams(
                voxel_length=args.voxel_length,
                sdf_trunc=args.sdf_trunc,
                depth_max=args.depth_max,
            ),
            log_path=log_path,
        )
    except Open3DUnavailableError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    composed = compose_with_annotations(
        static,
        scene_metadata=scene_metadata,
        scene_extractions=scene_extractions,
        output_dir=output_dir,
        log_path=log_path,
        body_name_for_static=args.body_name,
    )

    summary = {
        "source": "rgbd",
        "static_mesh": str(static.visual_obj),
        "collision_hull": str(static.collision_hull_obj),
        "composed_mjcf": str(composed.composed_mjcf),
        "n_extracted_bodies": len(composed.extracted_bodies),
        "convert_log": str(log_path),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
