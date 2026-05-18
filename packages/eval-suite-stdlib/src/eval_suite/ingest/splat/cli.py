"""One-command splat ingest CLI.

Usage:
  python -m eval_suite.ingest.splat ingest \\
    <splat.ply> \\
    --scene-metadata <scene_metadata.json> \\
    [--scene-extractions <scene_extractions.json>] \\
    --output-dir <dir> \\
    [--target-faces 3000] \\
    [--body-name splat_scene] \\
    [--sugar-repo /path/to/sugar]

Runs:
  1. `splat_to_static_mesh(splat, output_dir, target_faces)` →
     `visuals/scene.obj`, `collision_hull/scene_hull.obj`.
  2. `compose_with_annotations(static_mesh, scene_metadata, scene_extractions, output_dir)` →
     `MJCF/scene.xml` + per-extracted-body OBJs.

Writes a single `convert_log.json` aggregating both phases. Subprocess
locale is forced to `LC_ALL=C` to dodge non-ASCII output issues from
SuGaR / Nerfstudio on pods without `en_US.UTF-8` installed.

This CLI is the factory-engineer-facing entry point — invoked once per
scene capture, produces the converted artifacts the
ParametricSplatTask consumes. Heavy CUDA dependencies (SuGaR,
Nerfstudio) must be installed separately; see ../splat/README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .annotation import SceneExtractions, SceneMetadata
from .convert import (
    SplatToolUnavailableError,
    compose_with_annotations,
    splat_to_static_mesh,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval_suite.ingest.splat",
        description="Splat → MJCF ingest pipeline for the eval-suite substrate.",
    )
    subs = parser.add_subparsers(dest="cmd", required=True)

    ingest = subs.add_parser("ingest", help="Convert a splat .ply to MJCF.")
    ingest.add_argument("splat_ply", type=Path, help="Source Gaussian-splat .ply.")
    ingest.add_argument(
        "--scene-metadata", type=Path, required=True,
        help="Path to scene_metadata.json (transform + regions + spawn points).",
    )
    ingest.add_argument(
        "--scene-extractions", type=Path, default=None,
        help="Optional path to scene_extractions.json (bodies to carve out).",
    )
    ingest.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory; populated with MJCF/, visuals/, collision_hull/.",
    )
    ingest.add_argument(
        "--target-faces", type=int, default=3000,
        help="Target face count after decimation (default: 3000).",
    )
    ingest.add_argument(
        "--body-name", type=str, default="splat_scene",
        help="Name of the static body in the composed MJCF (default: splat_scene).",
    )
    ingest.add_argument(
        "--sugar-repo", type=Path, default=None,
        help="Path to a SuGaR clone (recorded in convert_log.json's sugar_commit).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.cmd != "ingest":  # pragma: no cover
        parser.error(f"unknown command {args.cmd!r}")
        return 2

    scene_metadata = SceneMetadata.load(args.scene_metadata)
    scene_extractions: SceneExtractions | None = None
    if args.scene_extractions is not None:
        scene_extractions = SceneExtractions.load(args.scene_extractions)

    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "convert_log.json"

    try:
        static = splat_to_static_mesh(
            args.splat_ply,
            output_dir,
            target_faces=args.target_faces,
            sugar_repo=args.sugar_repo,
            log_path=log_path,
        )
    except SplatToolUnavailableError as e:
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
