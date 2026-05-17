#!/usr/bin/env python3
"""v0 driver — run a sweep on `NamaqualandScanTask` and write the
`asset_provenance.json` sidecar.

Deliberately a standalone script (not a CLI subcommand) so the v0
build doesn't modify `eval_suite/sweep.py` or `eval_suite/cli.py`. The
sweep itself runs through the unchanged `run_sweep()` driver; this
script's only added value is computing the per-asset SHA256s and
writing the `asset_provenance.json` sidecar alongside the produced
manifest.

Run with the v0 MJX venv (mujoco_playground required for Go1):

    MUJOCO_GL=egl .venv-mjx/bin/python examples/namaqualand_sweep.py \\
        --output-dir results/v0.9_demo/ --trials 5 --seed 0

The script asserts the source USD and converted MJCF assets exist
before running the sweep; if they don't, point at the README in
`assets/namaqualand_scan/` for the conversion instructions.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from eval_suite.adapters import MujocoPlaygroundAdapter
from eval_suite.asset_provenance import AssetRef, write_for_run
from eval_suite.hashing import sha256_file
from eval_suite.policies import RandomLocomotionPolicy
from eval_suite.sweep import run_sweep
from eval_suite.tasks.usd_scan import GO1_ACTION_DIM, NamaqualandScanTask

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "namaqualand_scan"

_POLY_HAVEN_ASSET_URL = (
    "https://dl.polyhaven.org/file/ph-assets/Models/usd/8k/"
    "namaqualand_boulder_05/namaqualand_boulder_05_8k.usdc"
)
_POLY_HAVEN_PAGE_URL = "https://polyhaven.com/a/namaqualand_boulder_05"

# Module-level constant: the OOD-framing string written into Manifest.notes
# for every v0 sweep. Pulled out so the wording is recorded in one place,
# greppable, and stable across reruns (the manifest's run_id depends on it).
_OOD_NOTES = (
    "v0: real-world scan ingestion demo. Policy is out-of-distribution "
    "on this scene — the deliverable is the ingestion pipeline, not the "
    "success rate. See takehome/EXTENSION.md §2 for framing."
)

_CONVERT_CMD = "python assets/namaqualand_scan/convert.py"


def _git_head_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def _build_asset_refs() -> list[AssetRef]:
    """Build the per-asset provenance list. SHA256s are computed at sweep
    time, NOT read from convert_log.json — the load-bearing claim of the
    sidecar is "these specific bytes were on disk when the sweep ran."
    """
    source_usd = ASSETS_DIR / "source.usdc"
    visual_obj = ASSETS_DIR / "MJCF" / "visuals" / "Boulder.obj"
    hull_obj = ASSETS_DIR / "MJCF" / "collision_hull" / "boulder_hull.obj"
    scene_xml = ASSETS_DIR / "MJCF" / "scene.xml"

    for p in (source_usd, visual_obj, hull_obj, scene_xml):
        if not p.is_file():
            raise FileNotFoundError(
                f"required v0 asset missing: {p}. Run `python assets/namaqualand_scan/convert.py` "
                "in the usd2mjcf env (see assets/namaqualand_scan/README.md) before sweeping."
            )

    return [
        AssetRef(
            role="source_usd",
            path=str(source_usd.relative_to(REPO_ROOT)),
            sha256=sha256_file(source_usd),
            origin_url=_POLY_HAVEN_ASSET_URL,
            license="CC0-1.0",
            conversion_command="",
            notes=f"Poly Haven Namaqualand Boulder 05, photogrammetric scan, CC0. "
                  f"Page: {_POLY_HAVEN_PAGE_URL}",
        ),
        AssetRef(
            role="visual_mesh",
            path=str(visual_obj.relative_to(REPO_ROOT)),
            sha256=sha256_file(visual_obj),
            origin_url="",
            license="CC0-1.0",
            conversion_command=_CONVERT_CMD + " (Step A: usd2mjcf)",
            notes="Extracted visual OBJ; usd2mjcf TransformGraph processes the static USD as a single body.",
        ),
        AssetRef(
            role="collision_hull",
            path=str(hull_obj.relative_to(REPO_ROOT)),
            sha256=sha256_file(hull_obj),
            origin_url="",
            license="CC0-1.0",
            conversion_command=_CONVERT_CMD + " (Step B: trimesh decimate face_count=3000 → convex_hull)",
            notes="Decimated convex hull used as collision geom. Replaces usd2mjcf's "
                  "coacd --generate_collision (which on this scan produced 269 convex pieces).",
        ),
        AssetRef(
            role="composed_mjcf",
            path=str(scene_xml.relative_to(REPO_ROOT)),
            sha256=sha256_file(scene_xml),
            origin_url="",
            license="CC0-1.0",
            conversion_command=_CONVERT_CMD + " (Step C: emit composed scene.xml)",
            notes="Boulder-only MJCF. Go1 is attached at sweep time via mujoco.MjSpec.attach "
                  "(see eval_suite/tasks/usd_scan.py:_ScanSceneCompatEnv).",
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v0 sweep driver (Namaqualand scan).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    args = parser.parse_args(argv)

    assets = _build_asset_refs()
    print(f"[sweep] {len(assets)} assets resolved:")
    for ref in assets:
        print(f"  {ref.role:20s}  {ref.sha256[:16]}…  {ref.path}")

    task = NamaqualandScanTask(
        scene_path=ASSETS_DIR / "MJCF" / "scene.xml",
        max_episode_steps=args.max_episode_steps,
    )
    policy = RandomLocomotionPolicy(action_dim=GO1_ACTION_DIM, seed=args.seed)
    adapter = MujocoPlaygroundAdapter(videos_dir=Path(args.output_dir) / "videos")

    seeds = list(range(args.seed, args.seed + args.trials))
    print(f"[sweep] task={task.name}, policy={policy.name}, "
          f"adapter={adapter.name}, seeds={seeds}")

    manifest = run_sweep(
        policy=policy,
        task=task,
        adapter=adapter,
        seeds=seeds,
        output_dir=args.output_dir,
        code_sha=_git_head_sha(),
        notes=_OOD_NOTES,
    )
    print(f"[sweep] manifest.run_id = {manifest.run_id}")

    sidecar = write_for_run(
        target_run_id=manifest.run_id,
        assets=assets,
        output_dir=Path(args.output_dir),
    )
    print(f"[sweep] asset_provenance.json written; target_run_id={sidecar.target_run_id}")
    if not sidecar.verify(repo_root=REPO_ROOT):
        print("[sweep] WARNING: asset_provenance.verify() returned False")
        return 1
    print("[sweep] OK — manifest + asset_provenance both verify.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
