"""v0 canonical-taxonomy contract.

Covers:
1. Closed-enum CanonicalDim — task `canonical_axis_map` declarations
   only use the 4 allowed dim names.
2. Schema 0.2.0 manifests round-trip the canonical_axis_map and bind it
   into the run_id (changing the map changes the hash).
3. Schema 0.1.0 manifests still verify byte-identically (the legacy
   path excludes canonical_axis_map from hashing).
4. canonical_profile_for_sweep pools Wilson CIs correctly across cells
   that touch the same canonical dim.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from eval_suite._types import CellId, CellResult, canonical_dims
from eval_suite.analysis import (
    CombinedSweep,
    canonical_profile_for_sweep,
    canonical_profile_from_manifest,
)
from eval_suite.manifest import (
    SCHEMA_VERSION,
    CalibrationRef,
    CellResultPayload,
    HardwareRef,
    Manifest,
    ModelRef,
    SimulatorRef,
)
from eval_suite.tasks.simpler_env import GoogleRobotPickCokeCan, WidowXSpoonOnTowel
from eval_suite.tasks.unitree_go1 import UnitreeGo1Joystick


def test_closed_enum_dim_names() -> None:
    """Every Task's canonical_axis_map only maps to {language, visuals, physics, embodiment}."""
    allowed = set(canonical_dims())
    for task in (GoogleRobotPickCokeCan(), WidowXSpoonOnTowel(), UnitreeGo1Joystick()):
        m: dict[str, str] = getattr(task, "canonical_axis_map", {})
        for axis, dim in m.items():
            assert dim in allowed, (
                f"{type(task).__name__}.canonical_axis_map[{axis!r}] = {dim!r} "
                f"is not a CanonicalDim; allowed: {sorted(allowed)}"
            )


def test_schema_0_2_0_hashes_canonical_map() -> None:
    """Changing the canonical_axis_map changes the run_id on 0.2.0."""
    base = Manifest(
        schema_version=SCHEMA_VERSION,
        code_sha="x",
        container_digest="",
        model=ModelRef(name="m", checkpoint_sha256="aa" * 32, family="x"),
        simulator=SimulatorRef(name="s", commit="abc"),
        task_name="t",
        embodiment="e",
        trials_per_cell=2,
        cells=[CellResultPayload(cell_id="e/t/k=v", axes={"k": "v"}, n_trials=2,
                                 successes=1, wilson_ci_low=0.1, wilson_ci_high=0.9)],
        hardware=HardwareRef(gpu="g", cuda="c", driver="d"),
        seeds=[0, 1],
        calibration=CalibrationRef(tier="C"),
        canonical_axis_map={"k": "physics"},
    )
    base.seal()
    other = Manifest(
        schema_version=SCHEMA_VERSION,
        code_sha="x",
        container_digest="",
        model=ModelRef(name="m", checkpoint_sha256="aa" * 32, family="x"),
        simulator=SimulatorRef(name="s", commit="abc"),
        task_name="t",
        embodiment="e",
        trials_per_cell=2,
        cells=[CellResultPayload(cell_id="e/t/k=v", axes={"k": "v"}, n_trials=2,
                                 successes=1, wilson_ci_low=0.1, wilson_ci_high=0.9)],
        hardware=HardwareRef(gpu="g", cuda="c", driver="d"),
        seeds=[0, 1],
        calibration=CalibrationRef(tier="C"),
        canonical_axis_map={"k": "language"},  # different map
    )
    other.seal()
    assert base.run_id != other.run_id
    assert base.verify()
    assert other.verify()


def test_schema_0_1_0_legacy_verify(tmp_path: Path) -> None:
    """Manifests stored under schema 0.1.0 still verify under the legacy
    rules (canonical_axis_map excluded from hashing)."""
    # Synthesize a 0.1.0 manifest, seal it (it should hash WITHOUT
    # canonical_axis_map), and confirm verify() passes.
    m = Manifest(
        schema_version="0.1.0",
        code_sha="legacy",
        container_digest="",
        model=ModelRef(name="m", checkpoint_sha256="bb" * 32, family="x"),
        simulator=SimulatorRef(name="s", commit="abc"),
        task_name="t",
        embodiment="e",
        trials_per_cell=2,
        cells=[CellResultPayload(cell_id="e/t/k=v", axes={"k": "v"}, n_trials=2,
                                 successes=1, wilson_ci_low=0.1, wilson_ci_high=0.9)],
        hardware=HardwareRef(gpu="g", cuda="c", driver="d"),
        seeds=[0, 1],
        calibration=CalibrationRef(tier="C"),
        # Even if a 0.1.0 manifest somehow carries a canonical_axis_map,
        # the hash ignores it — preserving back-compat with manifests
        # sealed under v0 code.
        canonical_axis_map={"k": "language"},
    )
    m.seal()
    assert m.verify()

    # Mutating canonical_axis_map under 0.1.0 schema should NOT break
    # verify (the map isn't in the hash).
    m.canonical_axis_map = {"k": "physics"}
    assert m.verify()


def _make_sweep(cells: list[CellResult], embodiment: str, task_name: str) -> CombinedSweep:
    """Tiny CombinedSweep harness for testing canonical_profile_for_sweep."""
    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        code_sha="t",
        container_digest="",
        model=ModelRef(name="test-model", checkpoint_sha256="cc" * 32, family="rt1"),
        simulator=SimulatorRef(name="sim", commit="abc"),
        task_name=task_name,
        embodiment=embodiment,
        trials_per_cell=cells[0].n_trials if cells else 0,
        cells=[],
        hardware=HardwareRef(gpu="g", cuda="c", driver="d"),
        seeds=[0],
        calibration=CalibrationRef(tier="C"),
    )
    return CombinedSweep(label="test", manifest=manifest, cell_results=cells)


def test_canonical_profile_pools_correctly() -> None:
    """Canonical-dim score pools successes + trials across all cells
    that touch any axis mapped to that dim."""
    # 2 cells on Google Robot: one lighting-variant cell (visuals dim)
    # and one paraphrase cell (language dim).
    cells = [
        CellResult(
            cell=CellId(embodiment="google_robot", task="pick_coke_can",
                        axes={"orientation": "upright", "lighting": "darker", "background": "base",
                              "distractor": "base", "table_texture": "base"}),
            n_trials=20, successes=4,
            wilson_ci_low=0.07, wilson_ci_high=0.42,
        ),
        CellResult(
            cell=CellId(embodiment="google_robot", task="pick_coke_can",
                        axes={"orientation": "upright", "lighting": "base", "background": "base",
                              "distractor": "base", "table_texture": "base",
                              "paraphrase": "synonym"}),
            n_trials=20, successes=14,
            wilson_ci_low=0.49, wilson_ci_high=0.86,
        ),
    ]
    sweep = _make_sweep(cells, "google_robot", "pick_coke_can")
    profile = canonical_profile_for_sweep(sweep)

    # Visuals dim: only the first cell has a *non-base* visuals axis,
    # but the second cell also has visuals axes (all base). Both
    # contribute to visuals pooling. Combined: 4+14 = 18 / 40 = 0.45.
    assert profile.per_dim["visuals"].n_cells == 2
    assert profile.per_dim["visuals"].n_trials == 40
    assert abs(profile.per_dim["visuals"].mean - 0.45) < 1e-6

    # Language dim: only the second (paraphrase) cell contributes.
    # 14 / 20 = 0.70.
    assert profile.per_dim["language"].n_cells == 1
    assert profile.per_dim["language"].n_trials == 20
    assert abs(profile.per_dim["language"].mean - 0.70) < 1e-6

    # Physics dim: orientation is mapped to physics; both cells have orientation.
    # Combined: 18 / 40 = 0.45.
    assert profile.per_dim["physics"].n_cells == 2

    # Embodiment dim: no Google Robot axis maps to embodiment. Coverage = 0.
    assert profile.per_dim["embodiment"].n_cells == 0
    assert profile.per_dim["embodiment"].n_trials == 0

    # Worst dim ranking excludes the zero-coverage embodiment dim.
    assert profile.worst_dim_name in {"visuals", "physics"}  # both = 0.45 in this toy
    assert profile.worst_dim_score is not None
    assert profile.worst_dim_score == pytest.approx(0.45)


def test_canonical_profile_from_manifest_matches_sweep_version() -> None:
    """v0: canonical_profile_from_manifest should produce the same
    per-dim numbers as canonical_profile_for_sweep does on the equivalent
    synthesized sweep. Manifest carries enough data (cells, axes,
    successes, n_trials) — no trials.csv needed."""
    m = Manifest(
        schema_version=SCHEMA_VERSION,
        code_sha="x",
        container_digest="",
        model=ModelRef(name="m", checkpoint_sha256="aa" * 32, family="rt1"),
        simulator=SimulatorRef(name="s", commit="abc"),
        task_name="pick_coke_can",
        embodiment="google_robot",
        trials_per_cell=20,
        cells=[
            CellResultPayload(
                cell_id="google_robot/pick_coke_can/lighting=darker,orientation=upright",
                axes={"lighting": "darker", "orientation": "upright",
                      "background": "base", "distractor": "base", "table_texture": "base"},
                n_trials=20, successes=4, wilson_ci_low=0.07, wilson_ci_high=0.42,
            ),
            CellResultPayload(
                cell_id="google_robot/pick_coke_can/paraphrase=synonym",
                axes={"orientation": "upright", "lighting": "base", "background": "base",
                      "distractor": "base", "table_texture": "base", "paraphrase": "synonym"},
                n_trials=20, successes=14, wilson_ci_low=0.49, wilson_ci_high=0.86,
            ),
        ],
        hardware=HardwareRef(gpu="g", cuda="c", driver="d"),
        seeds=[0, 1],
        calibration=CalibrationRef(tier="C"),
    )

    profile = canonical_profile_from_manifest(m)
    # visuals: both cells contribute (they have visuals-mapped axes). Pooled = 18/40 = 0.45
    assert profile.per_dim["visuals"].n_cells == 2
    assert profile.per_dim["visuals"].n_trials == 40
    assert abs(profile.per_dim["visuals"].mean - 0.45) < 1e-6
    # language: only paraphrase cell. 14/20 = 0.7
    assert profile.per_dim["language"].n_cells == 1
    assert abs(profile.per_dim["language"].mean - 0.7) < 1e-6
    # embodiment: not measured on this Task. n_cells = 0.
    assert profile.per_dim["embodiment"].n_cells == 0


def test_canonical_profile_excludes_zero_coverage_dim_from_worst() -> None:
    """A dim with n_cells == 0 must not be picked as worst."""
    # WidowX in v0: only "condition" axis, mapped to visuals.
    cells = [
        CellResult(
            cell=CellId(embodiment="widowx", task="spoon_on_towel",
                        axes={"condition": "clean"}),
            n_trials=20, successes=2,
            wilson_ci_low=0.02, wilson_ci_high=0.30,
        ),
    ]
    sweep = _make_sweep(cells, "widowx", "spoon_on_towel")
    profile = canonical_profile_for_sweep(sweep)
    # Only visuals has coverage on WidowX in v0; that's the worst (and only) dim.
    assert profile.worst_dim_name == "visuals"
    assert profile.per_dim["language"].n_cells == 0
    assert profile.per_dim["physics"].n_cells == 0
    assert profile.per_dim["embodiment"].n_cells == 0
