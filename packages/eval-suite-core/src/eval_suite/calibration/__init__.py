"""Sim-to-real calibration substrate.

The v1 substrate already shipped a tier-tagged registry of published
real-robot reference numbers (`real_perf.json` + `analysis.py
::calibration_overlay`). v2 Phase 3 adds the *statistics layer* the
tier-A and tier-A+ rules in EXTENSION.md §4 actually require:

  - Outcome-level Pearson r (sim cell vs real cell) with bootstrap CI
    (`pearson_r_with_bootstrap_ci`).
  - Trajectory-level MMRV (maximum mean relative velocity error)
    between paired sim/real trajectories (`mmrv`).
  - PairedTrial dataclass + signed paired_trials.json sidecar — the
    on-disk artifact a partner lab produces when they record a sim
    rollout alongside a real-robot rollout of the same condition.

Lives in `eval-suite-core` (not stdlib) because the statistics layer
is contract-shaped, not plugin-shaped: third-party plugins should
*call* these functions, not reimplement them. The substrate decision
to make outcome- and trajectory-level calibration first-class.
"""

from .paired_trials import (
    PairedTrial,
    paired_trials_sidecar_path,
    record_paired_trial,
)
from .statistics import (
    PearsonCI,
    mmrv,
    paired_cell_data,
    pearson_r_with_bootstrap_ci,
)

__all__ = [
    "PearsonCI",
    "pearson_r_with_bootstrap_ci",
    "mmrv",
    "paired_cell_data",
    "PairedTrial",
    "record_paired_trial",
    "paired_trials_sidecar_path",
]
