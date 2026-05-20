"""Thin-slice demo: a ROS 2 lifecycle node refuses to deploy a policy
whose eval-suite profile doesn't clear a deployer-set bar.

**In plain words.** Shows the admission-gate end of the suite in
action: load a fixture profile, run two gates against it (one
strict, one relaxed), and watch the strict one print refusal
reasons while the relaxed one passes. This is the deployment-time
trust contract every other piece of the suite plugs into.


This script doesn't actually start rclpy (the demo is the gate, not the
ROS 2 plumbing). It loads a fixture manifest, evaluates two gates
against it, and prints the refusal reasons in the strict case + the
pass log in the relaxed case. Tie this to your real deployment flow by
swapping the fixture for a real manifest path and calling the gate from
your lifecycle node's `on_activate`.
"""

from __future__ import annotations

import sys

from eval_suite.manifest import (
    CalibrationRef,
    CellResultPayload,
    HardwareRef,
    Manifest,
    ModelRef,
    SimulatorRef,
)
from eval_suite.ros2 import ProfileGate


def _fixture_manifest() -> Manifest:
    """Synthesize a SmolVLA-on-Google-Robot-style profile with a weak
    paraphrase axis (worst dim 0.27 — exactly the v0 RT-1 number from
    the README) so the strict gate has something concrete to refuse on.
    """
    cells = [
        # Strong visuals/physics cells
        CellResultPayload(
            cell_id=f"google_robot/pick_coke_can/{n}",
            axes=axes,
            n_trials=20, successes=int(s * 20),
            wilson_ci_low=s - 0.1, wilson_ci_high=s + 0.1,
        )
        for n, axes, s in [
            ("orientation=vertical", {"orientation": "vertical"}, 0.85),
            ("lighting=darker", {"lighting": "darker"}, 0.88),
            ("background=alt1", {"background": "alt1"}, 0.86),
            ("distractor=none", {"distractor": "none"}, 0.90),
            ("table_texture=wood", {"table_texture": "wood"}, 0.86),
            # The weak language axis (mirrors the v0 RT-1 paraphrase result)
            ("paraphrase=walk_forward", {"paraphrase": "walk_forward"}, 0.20),
            ("paraphrase=stand_up_grab", {"paraphrase": "stand_up_grab"}, 0.30),
        ]
    ]
    m = Manifest(
        schema_version="0.3.0",
        code_sha="fixturesha",
        container_digest="",
        model=ModelRef(name="smolvla-base", checkpoint_sha256="abc", family="lerobot"),
        simulator=SimulatorRef(name="simpler-env", commit="unknown"),
        task_name="pick_coke_can",
        embodiment="google_robot",
        trials_per_cell=20,
        cells=cells,
        hardware=HardwareRef(gpu="RTX 3090", cuda="12.4", driver="580"),
        seeds=list(range(20)),
        calibration=CalibrationRef(tier="C"),
        canonical_axis_map={
            "orientation": "physics",
            "lighting": "visuals",
            "background": "visuals",
            "distractor": "visuals",
            "table_texture": "visuals",
            "paraphrase": "language",
        },
    )
    from eval_suite.hashing import hash_dict
    m.run_id = hash_dict(m._hashable_payload())  # noqa: SLF001
    return m


def main(argv: list[str] | None = None) -> int:
    manifest = _fixture_manifest()
    print(f"manifest run_id: {manifest.run_id[:16]}")
    print(f"model.family:    {manifest.model.family}")
    print()

    strict = ProfileGate(
        worst_dim_min_score=0.6,
        required_canonical_dims=["language", "visuals", "physics"],
        min_calibration_tier="B",
        policy_family_allowlist=["lerobot", "simpler_env"],
    )
    relaxed = ProfileGate(
        worst_dim_min_score=0.15,  # accept the weak language axis
        policy_family_allowlist=["lerobot", "simpler_env"],
    )

    print("=== strict gate ===")
    result = strict.evaluate(manifest)
    print(f"passed: {result.passed}")
    for r in result.reasons:
        print(f"  - {r}")
    print()

    print("=== relaxed gate ===")
    result = relaxed.evaluate(manifest)
    print(f"passed: {result.passed}")
    for r in result.reasons:
        print(f"  - {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
