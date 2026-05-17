"""Run the v0 Go1 (Unitree quadruped, MuJoCo Playground) sweep.

Standalone instead of a subcommand on `eval_suite.cli` so the v0 CLI
(SimplerEnv / TF / JAX) doesn't have to import jax-mujoco-playground in
its module-level set. Run via:

    MUJOCO_GL=egl .venv-mjx/bin/python scripts/run_go1_sweep.py \\
        --trials 10 --max-steps 100 \\
        --output-dir results/sweep_YYYYMMDD/random_go1_unitree_go1_joystick/

Use `--smoke` for a 2-cell × 2-seed × 30-step sanity pass.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="run_go1_sweep")
    p.add_argument("--trials", type=int, default=10, help="N seeds per cell")
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=100,
                   help="Per-rollout step budget. Random policy falls fast; 100 is a 2s sim cushion.")
    p.add_argument("--cells", default=None,
                   help="Comma-separated cell indices (default: all 12)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--videos-dir", default=None,
                   help="Optional. Defaults to <output-dir>/videos/.")
    p.add_argument("--smoke", action="store_true",
                   help="2 cells × 2 seeds × 30 steps; for end-to-end validation only.")
    p.add_argument("--notes", default="")
    args = p.parse_args(argv)

    # MUJOCO_GL has to be set before mujoco is imported anywhere that
    # might try to grab a GL context. Set it here at the entrypoint so the
    # whole process honors it.
    os.environ.setdefault("MUJOCO_GL", "egl")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Imports deferred until after MUJOCO_GL is set.
    from eval_suite.adapters import MujocoPlaygroundAdapter
    from eval_suite.manifest import CalibrationRef
    from eval_suite.policies import RandomLocomotionPolicy
    from eval_suite.sweep import run_sweep
    from eval_suite.tasks.unitree_go1 import GO1_ACTION_DIM, UnitreeGo1Joystick

    if args.smoke:
        args.trials = 2
        args.cells = "0,1"
        args.max_steps = 30

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = Path(args.videos_dir) if args.videos_dir else out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    task = UnitreeGo1Joystick(max_episode_steps=args.max_steps)
    policy = RandomLocomotionPolicy(action_dim=GO1_ACTION_DIM, seed=args.seed_base)
    adapter = MujocoPlaygroundAdapter(videos_dir=videos_dir, video_fps=30)

    seeds = list(range(args.seed_base, args.seed_base + args.trials))
    cells = [int(c) for c in args.cells.split(",")] if args.cells else None

    t0 = time.monotonic()
    manifest = run_sweep(
        policy=policy,
        task=task,
        adapter=adapter,
        seeds=seeds,
        cells=cells,
        output_dir=out_dir,
        # No container digest for the v0 venv path; the manifest still
        # captures code_sha + sim aux commits via _resolve_simulator_ref.
        container_digest="",
        calibration=CalibrationRef(tier="C"),
        notes=args.notes or (
            "v0 random-policy sweep against MuJoCo Playground Go1 joystick. "
            "Random control inputs; aggregate success near zero is expected. "
            "Deliverable: pipeline absorbs a 12-DoF joint-space embodiment with "
            "no contract changes (see takehome/EXTENSION.md)."
        ),
    )
    print(f"sweep done in {time.monotonic()-t0:.1f}s")
    print(f"run_id: {manifest.run_id[:16]}")
    print(f"manifest: {out_dir / 'manifest.json'}")
    print(f"trials.csv: {out_dir / 'trials.csv'}")
    print(f"videos: {videos_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
