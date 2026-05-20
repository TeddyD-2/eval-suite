"""Minimal end-to-end LeRobot interop demo.

Runs a small sweep with the `LeRobotPolicy` plugin against `MockTask`
(no GPU sim required) so you can see the wiring without provisioning
SAPIEN. For a real evaluation, swap `--task mock` for any of the
in-tree sim tasks (`google_robot_pick_coke_can`, `widowx_spoon_on_towel`,
`unitree_go1_joystick`) with the matching Adapter.

The point of this script: prove a HuggingFace-distributed LeRobot
checkpoint drops into the sweep without modifying the suite. The
manifest gets sealed, signed (if a key is provided), and is
indistinguishable in shape from a SimplerEnv-policy sweep.

Usage:
    pip install 'eval-suite-stdlib[lerobot]'
    python examples/lerobot_smolvla_pickcoke.py \\
        --repo-id lerobot/smolvla-base \\
        --device cpu \\
        --trials 3 \\
        --output-dir /tmp/lerobot_demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="lerobot/smolvla-base",
                   help="HuggingFace repo id of the LeRobot policy.")
    p.add_argument("--revision", default=None,
                   help="Optional commit SHA to pin; default resolves main.")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--task", default="mock",
                   help="Task plugin name; mock is the no-GPU default.")
    p.add_argument("--adapter", default="gym")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args(argv)

    from eval_suite.manifest import CalibrationRef
    from eval_suite.registry import get_adapter, get_policy, get_task
    from eval_suite.sweep import run_sweep

    task = get_task(args.task)()
    policy = get_policy("lerobot")(repo_id=args.repo_id, revision=args.revision, device=args.device)
    adapter = get_adapter(args.adapter)()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = run_sweep(
        policy=policy,
        task=task,
        adapter=adapter,
        seeds=list(range(args.trials)),
        output_dir=out,
        calibration=CalibrationRef(tier="C"),
        notes=f"LeRobot interop demo: {args.repo_id}",
    )
    print(f"run_id: {manifest.run_id[:16]}")
    print(f"model.family: {manifest.model.family}")
    print(f"model.checkpoint_id: {manifest.model.name}")
    print(f"manifest: {out / 'manifest.json'}")
    print(f"verify: {manifest.verify()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
