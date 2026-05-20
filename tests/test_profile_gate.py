"""ProfileGate contract tests.

**In plain words.** Pins down the rule that a policy with a weak
profile cannot be ACTIVE'd on a real robot. Every gate clause
(worst dim, calibration tier, required dimensions, paired Pearson
r, family allowlist) gets exercised. If this ever fails, the
deployment-admission contract is no longer trustworthy.

The gate is the thin slice: a deployer-set YAML bar the eval-suite
profile must clear before a ROS 2 lifecycle node will activate. Tests
exercise every clause + at least one combined refusal.
"""

from __future__ import annotations

from pathlib import Path

from eval_suite.manifest import (
    CalibrationRef,
    CellResultPayload,
    HardwareRef,
    Manifest,
    ModelRef,
    SimulatorRef,
)
from eval_suite.ros2 import ProfileGate


def _make_manifest(
    *,
    family: str = "lerobot",
    tier: str = "B",
    worst_axis_score: float = 0.8,
    measured_dims: list[str] | None = None,
) -> Manifest:
    """Synthesize a sealed manifest with the requested profile shape.

    We build cells whose Wilson-pooled mean per canonical dim equals
    `worst_axis_score`, so the gate's worst_dim check gets a clean number.
    """
    measured_dims = measured_dims if measured_dims is not None else ["visuals", "physics"]
    axis_map = {f"axis_{d}": d for d in measured_dims}
    cells = []
    for d in measured_dims:
        cells.append(CellResultPayload(
            cell_id=f"emb/task/axis_{d}=cellA",
            axes={f"axis_{d}": "cellA"},
            n_trials=20,
            successes=int(round(worst_axis_score * 20)),
            wilson_ci_low=worst_axis_score - 0.1,
            wilson_ci_high=worst_axis_score + 0.1,
        ))
    m = Manifest(
        schema_version="0.3.0",
        code_sha="testsha",
        container_digest="",
        model=ModelRef(name="testpolicy", checkpoint_sha256="abc", family=family),
        simulator=SimulatorRef(name="mock", commit="unknown"),
        task_name="testtask",
        embodiment="testemb",
        trials_per_cell=20,
        cells=cells,
        hardware=HardwareRef(gpu="cpu", cuda="none", driver="none"),
        seeds=list(range(20)),
        calibration=CalibrationRef(tier=tier),
        canonical_axis_map=axis_map,
    )
    # Force seal so run_id is populated.
    from eval_suite.hashing import hash_dict
    m.run_id = hash_dict(m._hashable_payload())  # noqa: SLF001
    return m


def test_empty_gate_passes_any_manifest() -> None:
    gate = ProfileGate()
    result = gate.evaluate(_make_manifest())
    assert result.passed
    assert result.reasons == []


def test_gate_worst_dim_below_threshold_refuses() -> None:
    gate = ProfileGate(worst_dim_min_score=0.9)
    result = gate.evaluate(_make_manifest(worst_axis_score=0.5))
    assert not result.passed
    assert any("worst_dim_score" in r for r in result.reasons)


def test_gate_worst_dim_above_threshold_passes() -> None:
    gate = ProfileGate(worst_dim_min_score=0.4)
    assert gate.evaluate(_make_manifest(worst_axis_score=0.8)).passed


def test_gate_calibration_tier_floor_refuses_below() -> None:
    gate = ProfileGate(min_calibration_tier="A")
    result = gate.evaluate(_make_manifest(tier="C"))
    assert not result.passed
    assert any("calibration tier" in r for r in result.reasons)


def test_gate_calibration_tier_floor_passes_at_or_above() -> None:
    gate = ProfileGate(min_calibration_tier="B")
    assert gate.evaluate(_make_manifest(tier="A")).passed
    assert gate.evaluate(_make_manifest(tier="B")).passed


def test_gate_policy_family_allowlist_refuses_unlisted() -> None:
    gate = ProfileGate(policy_family_allowlist=["simpler_env"])
    result = gate.evaluate(_make_manifest(family="lerobot"))
    assert not result.passed
    assert any("policy family" in r for r in result.reasons)


def test_gate_required_canonical_dims_refuses_missing() -> None:
    gate = ProfileGate(required_canonical_dims=["language"])
    result = gate.evaluate(_make_manifest(measured_dims=["visuals", "physics"]))
    assert not result.passed
    assert any("language" in r for r in result.reasons)


def test_gate_combined_clauses_collect_all_reasons() -> None:
    gate = ProfileGate(
        worst_dim_min_score=0.9,
        min_calibration_tier="A",
        policy_family_allowlist=["simpler_env"],
    )
    result = gate.evaluate(
        _make_manifest(family="lerobot", tier="C", worst_axis_score=0.3),
    )
    assert not result.passed
    # Three independent clauses fail — three reasons.
    assert len(result.reasons) == 3


def test_gate_from_yaml_file(tmp_path: Path) -> None:
    path = tmp_path / "gate.yaml"
    path.write_text(
        "worst_dim_min_score: 0.6\n"
        "min_calibration_tier: B\n"
        "policy_family_allowlist:\n"
        "  - lerobot\n"
        "  - simpler_env\n"
    )
    gate = ProfileGate.from_yaml_file(path)
    assert gate.worst_dim_min_score == 0.6
    assert gate.min_calibration_tier == "B"
    assert "lerobot" in gate.policy_family_allowlist


def test_gate_required_pearson_r_without_sidecar_refuses() -> None:
    gate = ProfileGate(required_paired_pearson_r=0.7)
    result = gate.evaluate(_make_manifest())
    assert not result.passed
    assert any("paired_trials.json" in r for r in result.reasons)
