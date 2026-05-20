"""Sim-to-real calibration demo: outcome-level Pearson r + trajectory-level MMRV.

**In plain words.** Run this script to see what the calibration
layer actually produces: a real Pearson r number with a
confidence interval, computed by pairing the suite's per-cell sim
success rates against the published per-condition real-world
numbers from the SimplerEnv paper. The second half shows MMRV on a
synthesized trajectory pair as a stand-in for how a partner lab
would record a real one. Concrete demo of the v2 calibration pillar.


Two halves:

  1. **Outcome-level Pearson r.** Load a sealed manifest (one of the
     reference RT-1 sweeps under `manifests/`), pair its per-cell sim
     success rates with the SimplerEnv-paper per-condition real
     numbers seeded into `calibration/real_perf.json`, and report
     Pearson r + bootstrap CI.

  2. **Trajectory-level MMRV.** Synthesize a paired (sim, real) joint
     trajectory pair as a stand-in for what `OXEReplayPolicy` (Phase
     1) produces when you actually run it against a sim env, and
     record the resulting PairedTrial into a paired_trials.json
     sidecar.

The synthesized half is what makes this script runnable on a fresh
checkout. The intended production path: replace the synthesized
trajectories with (sim from a real sweep's `trajectory.npz`, real from
an OXE/LeRobot episode loaded through `OXEReplayPolicy`).

Honest framing per EXTENSION.md §4: this is what tier-A looks like —
when paired per-cell data exists for ≥10 cells, you can report a real
Pearson r, with the framework's bootstrap-CI honesty about how tight
the claim actually is.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from eval_suite._types import CellId
from eval_suite.calibration import (
    PairedTrial,
    paired_cell_data,
    paired_trials_sidecar_path,
    pearson_r_with_bootstrap_ci,
    record_paired_trial,
)
from eval_suite.manifest import Manifest


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "manifests/rt1_google_robot_pick_coke_can/manifest.json"
    registry_path = repo_root / "packages/eval-suite-core/src/eval_suite/calibration/real_perf.json"
    out_dir = repo_root / "results/calibration_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = Manifest.from_json(manifest_path.read_text())
    registry = json.loads(registry_path.read_text())

    print("=== Outcome-level Pearson r ===")
    print(f"manifest:    {manifest_path.relative_to(repo_root)}")
    print(f"run_id:      {manifest.run_id[:16]}")
    print(f"verify:      {manifest.verify()}")

    cells, sim_rates, real_rates = paired_cell_data(
        manifest_cells=manifest.cells,
        registry_entries=registry["entries"],
        task_key="google_robot_pick_coke_can",
        model_key="rt-1-converged",
    )
    print(f"\nPaired cells found: {len(cells)}")
    if not cells:
        print(
            "no per-cell registry entries matched this manifest's axes; "
            "calibration tier C until per-cell numbers seed."
        )
    for c, s, r in zip(cells, sim_rates, real_rates, strict=True):
        print(f"  {c.axes}  sim={s:.3f}  real={r:.3f}  delta={s-r:+.3f}")

    if len(cells) >= 3:
        result = pearson_r_with_bootstrap_ci(sim_rates, real_rates, n_bootstrap=5000, seed=42)
        print(
            f"\nPearson r (n={result.n_pairs}): {result.r:+.3f}  "
            f"95% bootstrap CI [{result.ci_low:+.3f}, {result.ci_high:+.3f}]"
        )
        if result.n_pairs < 10:
            print(
                "  → tier C/B (need ≥10 paired cells for tier A per EXTENSION.md §4)"
            )
        else:
            print("  → tier A reportable")
    else:
        print("\nfewer than 3 paired cells — Pearson r undefined")

    print("\n=== Trajectory-level MMRV (synthesized stand-in) ===")
    # Synthesize a 100-step joint trajectory (12-DoF, fake Go1-ish) where
    # the sim follows the real trajectory with a 10% velocity bias —
    # exactly the case where MMRV reports a small but non-zero error.
    t = np.linspace(0.0, 5.0, 100)[:, None]
    base = np.column_stack([
        np.sin(t * (1 + 0.1 * j)) for j in range(12)
    ])
    real_traj = base
    sim_traj = base * 1.10  # 10% velocity scaling
    cell = CellId(
        embodiment="google_robot",
        task="pick_coke_can",
        axes={"orientation": "vertical"},
    )
    pt = PairedTrial.from_trajectories(
        cell=cell,
        sim_run_id=manifest.run_id,
        sim_seed=0,
        sim_traj=sim_traj,
        real_traj=real_traj,
        real_episode_ref="synth:rt1-pickcoke-vertical#0",
        dt=0.05,
        notes="Synthesized stand-in for an OXE/LeRobot episode replay.",
    )
    print(f"MMRV (sim vs real trajectory): {pt.mmrv_score:.3f}")
    print(f"steps compared: {pt.n_steps_compared}")

    sidecar = paired_trials_sidecar_path(out_dir)
    # Sidecar is bound to a manifest run_id — refusing to mix means
    # earlier demo runs need to be cleared first.
    if sidecar.exists():
        sidecar.unlink()
    record_paired_trial(
        sidecar_path=sidecar,
        manifest_run_id=manifest.run_id,
        task_key="google_robot_pick_coke_can",
        model_key="rt-1-converged",
        paired_trial=pt,
    )
    print(f"sidecar written: {sidecar.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
