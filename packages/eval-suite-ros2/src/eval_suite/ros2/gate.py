"""ProfileGate — the thin-slice deployment admission controller.

**In plain words.** The "should this policy be allowed to drive the
robot?" check. The deployer writes a short YAML stating their
minimum bar — worst dimension score, calibration tier, allowed
policy families, required Pearson r — and the gate reads the
attached eval-suite manifest's profile and refuses activation when
it doesn't clear the bar. Refusal reasons are logged so a fleet
engineer can see exactly which number was too low. This is what
ties the rest of the suite's honesty machinery to a real
deployment decision.


The credibility wedge between "I have a sim profile" and "I'm willing
to let this run on real hardware" is a *small, declarative* contract
the deployer sets and the eval-suite checks before transitioning the
ROS 2 lifecycle node to ACTIVE.

The contract is a YAML file:

    # gate.yaml
    worst_dim_min_score: 0.6
    required_canonical_dims: [visuals, physics]
    min_calibration_tier: B
    required_paired_pearson_r: 0.7      # Phase 3 — sidecar-based
    policy_family_allowlist: [lerobot, simpler_env]

Any of those keys can be omitted; the gate ANDs the ones that are
present. Refusal produces a list of human-readable reasons that go to
rosout when the lifecycle node refuses to transition to ACTIVE.

This is intentionally not "rich policy framework": it is the
smallest credible bar a fleet engineer would set today.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from eval_suite._types import canonical_dims
from eval_suite.manifest import Manifest

# Tier ordering for `min_calibration_tier` comparison.
_TIER_RANK = {"C": 0, "B": 1, "A": 2, "A+": 3}


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProfileGate:
    """Declarative deployment admission contract."""

    worst_dim_min_score: float | None = None
    required_canonical_dims: list[str] = field(default_factory=list)
    min_calibration_tier: str | None = None
    required_paired_pearson_r: float | None = None
    policy_family_allowlist: list[str] = field(default_factory=list)

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> ProfileGate:
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"gate YAML at {path} must be a mapping, got {type(raw)}")
        return cls(
            worst_dim_min_score=raw.get("worst_dim_min_score"),
            required_canonical_dims=list(raw.get("required_canonical_dims", [])),
            min_calibration_tier=raw.get("min_calibration_tier"),
            required_paired_pearson_r=raw.get("required_paired_pearson_r"),
            policy_family_allowlist=list(raw.get("policy_family_allowlist", [])),
        )

    def evaluate(self, manifest: Manifest) -> GateResult:
        """Apply every present constraint. Reasons are human-readable."""
        reasons: list[str] = []
        if self.policy_family_allowlist:
            if manifest.model.family not in self.policy_family_allowlist:
                reasons.append(
                    f"policy family {manifest.model.family!r} not in allowlist "
                    f"{self.policy_family_allowlist!r}"
                )
        if self.min_calibration_tier:
            have = manifest.calibration.tier
            need = self.min_calibration_tier
            if _TIER_RANK.get(have, -1) < _TIER_RANK.get(need, 99):
                reasons.append(
                    f"calibration tier {have!r} below required {need!r}"
                )
        # Outcome-driven checks: worst-dim and required dims need a profile.
        profile = _profile_from_manifest(manifest)
        if self.worst_dim_min_score is not None:
            worst = profile.get("worst_dim_score")
            if worst is None:
                reasons.append("worst_dim_score unavailable (no measured cells)")
            elif worst < self.worst_dim_min_score:
                reasons.append(
                    f"worst_dim_score={worst:.3f} below required "
                    f"{self.worst_dim_min_score:.3f}"
                )
        if self.required_canonical_dims:
            measured = set(profile.get("measured_dims", set()))
            missing = [d for d in self.required_canonical_dims if d not in measured]
            if missing:
                reasons.append(
                    f"required canonical dims not measured: {missing!r}"
                )
        if self.required_paired_pearson_r is not None:
            r = _paired_pearson_from_sidecar(manifest)
            if r is None:
                reasons.append(
                    "required paired Pearson r unavailable "
                    "(no paired_trials.json sidecar found)"
                )
            elif r < self.required_paired_pearson_r:
                reasons.append(
                    f"paired Pearson r={r:.3f} below required "
                    f"{self.required_paired_pearson_r:.3f}"
                )
        return GateResult(passed=not reasons, reasons=reasons)


def _profile_from_manifest(manifest: Manifest) -> dict[str, Any]:
    """Compute a thin profile snapshot for gate evaluation.

    The full profile renderer lives in `eval_suite.analysis`; we don't
    pull that in here because rendering a profile also walks the
    trials.csv on disk, which the deploy-time gate often doesn't have
    (the deployer ships only the manifest). So we pool over the cells
    payload — which is what's bound into the run_id anyway.
    """
    axis_map = manifest.canonical_axis_map or {}
    measured_dims: set[str] = set()
    per_dim_sums: dict[str, tuple[int, int]] = {}  # dim → (successes, trials)
    for cell in manifest.cells:
        for axis_name in cell.axes:
            dim = axis_map.get(axis_name)
            if not dim:
                continue
            s, n = per_dim_sums.get(dim, (0, 0))
            per_dim_sums[dim] = (s + cell.successes, n + cell.n_trials)
            measured_dims.add(dim)
    if not per_dim_sums:
        return {"worst_dim_score": None, "measured_dims": set(), "per_dim": {}}
    means = {d: s / n for d, (s, n) in per_dim_sums.items() if n > 0}
    worst_score = min(means.values()) if means else None
    return {
        "worst_dim_score": worst_score,
        "measured_dims": measured_dims,
        "per_dim": means,
        "canonical_dims": list(canonical_dims()),
    }


def _paired_pearson_from_sidecar(manifest: Manifest) -> float | None:
    """Read paired Pearson r from a paired_trials.json sidecar, if present.

    Phase 3 lands this sidecar — when it exists alongside the manifest,
    the gate can require an actual statistical floor. The gate itself
    doesn't *compute* r; it just reads the sidecar value.
    """
    sidecar = _expected_sidecar_path(manifest)
    if sidecar is None or not sidecar.exists():
        return None
    try:
        payload = json.loads(sidecar.read_text())
    except json.JSONDecodeError:
        return None
    r = payload.get("pearson_r")
    if isinstance(r, (int, float)):
        return float(r)
    return None


def _expected_sidecar_path(manifest: Manifest) -> Path | None:
    """Heuristic: a sidecar lives next to the manifest on disk. The
    Manifest doesn't carry its own path; in Phase 3 we'll thread this
    through explicitly. For now the gate looks at the CWD plus any
    path set via $EVAL_SUITE_PAIRED_TRIALS_SIDECAR."""
    import os
    explicit = os.environ.get("EVAL_SUITE_PAIRED_TRIALS_SIDECAR")
    if explicit:
        return Path(explicit)
    return None
