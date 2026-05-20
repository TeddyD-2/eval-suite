"""Top-level ingest CLI dispatcher.

The ingest path has grown from a single splat → MJCF pipeline to a
small family of asset-source-specific paths, each of which produces
the same artifact shape (an MJCF + scene_metadata + asset_provenance
sidecar). This module is the single user-facing entry point:

    python -m eval_suite.ingest splat <ply> --scene-metadata <json> --output-dir <dir>
    python -m eval_suite.ingest rgbd  <frames_dir> --output-dir <dir>
    python -m eval_suite.ingest objaverse <asset_id> --output-dir <dir>

Each subcommand defers to its own subpackage. Heavy deps (open3d,
objaverse, SuGaR) are documented inside the relevant subpackage; the
dispatcher itself imports nothing heavy so `--help` works without any
extras.
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m eval_suite.ingest",
        description=(
            "Asset → MJCF ingest dispatcher. Each subcommand turns a "
            "specific source (Gaussian splat, RGB-D capture, public asset "
            "library) into the same composed-MJCF + scene_metadata "
            "artifact ParametricSplatTask consumes."
        ),
    )
    sub = p.add_subparsers(dest="source", required=True)

    sub.add_parser(
        "splat",
        help="Gaussian-splat .ply → MJCF (existing v1 path).",
        add_help=False,  # forward --help to the subpackage's argparser
    )
    sub.add_parser(
        "rgbd",
        help="RGB-D capture directory → TSDF mesh → MJCF.",
        add_help=False,
    )
    sub.add_parser(
        "objaverse",
        help="Objaverse-XL asset id → procedural scene MJCF.",
        add_help=False,
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        _build_parser().print_help()
        return 0
    source, rest = args[0], args[1:]

    if source == "splat":
        from .splat.cli import main as splat_main
        # The splat CLI already takes an `ingest` subcommand; re-route
        # so `python -m eval_suite.ingest splat <ply> ...` works the
        # same as `python -m eval_suite.ingest.splat ingest <ply> ...`.
        return splat_main(["ingest", *rest])
    if source == "rgbd":
        from .rgbd.cli import main as rgbd_main
        return rgbd_main(rest)
    if source == "objaverse":
        from .objaverse.cli import main as obj_main
        return obj_main(rest)
    print(f"unknown ingest source: {source!r}", file=sys.stderr)
    _build_parser().print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
