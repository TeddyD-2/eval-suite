"""Load sweep manifests/CSVs and compute the profile metrics the
notebook renders. Pure-Python; no plotting deps (matplotlib stays in
the notebook).

Public surface:
  - load_sweep_dir(path) → SweepBundle
  - profile_for_bundle(bundle) → ProfileReport (per-axis means, worst-axis)
  - REAL_PERF — published real-robot reference numbers (for calibration overlay)
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._types import CanonicalDim, CellId, CellResult, canonical_dims
from .manifest import Manifest
from .statistics import per_axis_means, wilson_ci, worst_axis

# Default location for the calibration registry. Override by setting
# EVAL_SUITE_CALIBRATION_REGISTRY to an absolute path (used by tests
# and by v2's customer-deployment-data pipeline).
# The registry ships as package data inside eval-suite-core so it follows
# the install — no separate `calibration/` directory at the repo root to
# remember to copy into the Docker image.
_DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent / "calibration" / "real_perf.json"


def _registry_path() -> Path:
    override = os.environ.get("EVAL_SUITE_CALIBRATION_REGISTRY")
    return Path(override) if override else _DEFAULT_REGISTRY_PATH


def _load_real_perf() -> dict[str, dict[str, float]]:
    """Load published real-robot performance numbers from the calibration
    registry JSON. Returns the same nested {task_key: {model_key: value}}
    shape that the old hardcoded REAL_PERF dict had, so the rest of the
    module's API is unchanged.

    Tolerates a missing registry (returns {}) so contract tests that don't
    care about calibration overlays don't have to seed a registry file.
    """
    path = _registry_path()
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out: dict[str, dict[str, float]] = {}
    for entry in raw.get("entries", []):
        task_key = entry["task_key"]
        model_key = entry["model_key"]
        out.setdefault(task_key, {})[model_key] = float(entry["value"])
    return out


def _load_real_perf_full() -> list[dict[str, Any]]:
    """Returns the raw entries list with full provenance fields. The
    calibration overlay caller uses _load_real_perf() for the lookup;
    the portal / notebook show the full provenance via this."""
    path = _registry_path()
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return list(raw.get("entries", []))


# Eagerly load so existing imports of `REAL_PERF` keep working
# (notebook header, third-party scripts). Override the env var BEFORE
# importing this module to point at a custom registry.
REAL_PERF: dict[str, dict[str, float]] = _load_real_perf()


@dataclass(frozen=True)
class CombinedSweep:
    """A single (model, task) sweep loaded from disk."""

    label: str  # e.g. "rt1_google_robot_pick_coke_can"
    manifest: Manifest
    cell_results: list[CellResult]

    @property
    def model_name(self) -> str:
        return self.manifest.model.name

    @property
    def model_family(self) -> str:
        return self.manifest.model.family

    @property
    def embodiment(self) -> str:
        return self.manifest.embodiment

    @property
    def task_name(self) -> str:
        return self.manifest.task_name

    @property
    def task_key(self) -> str:
        return f"{self.embodiment}_{self.task_name}"


@dataclass(frozen=True)
class SweepBundle:
    """All sweeps under a sweep_<timestamp>/ directory."""

    root: Path
    sweeps: list[CombinedSweep]

    def by_task(self, task_key: str) -> list[CombinedSweep]:
        return [s for s in self.sweeps if s.task_key == task_key]


def load_sweep_dir(path: str | Path) -> SweepBundle:
    """Load every <label>/{trials.csv, manifest.json} pair under `path`.

    Tolerates partial sweeps — if `trials.csv` exists but `manifest.json`
    is missing (sweep crashed mid-run), the sweep is skipped with a
    visible note in the bundle (caller can re-scan once it finishes).
    """
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(root)

    sweeps: list[CombinedSweep] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        trials_csv = entry / "trials.csv"
        manifest_json = entry / "manifest.json"
        if not (trials_csv.exists() and manifest_json.exists()):
            continue
        manifest = Manifest.from_json(manifest_json.read_text())
        cell_results = _load_cell_results_from_csv(trials_csv)
        sweeps.append(CombinedSweep(label=entry.name, manifest=manifest, cell_results=cell_results))
    return SweepBundle(root=root, sweeps=sweeps)


def _load_cell_results_from_csv(csv_path: Path) -> list[CellResult]:
    """Re-derive CellResults from the per-trial CSV. Lets us inspect
    partial sweeps before manifest.json lands."""
    by_cell_slug: dict[str, list[tuple[bool, dict[str, str]]]] = {}
    cell_meta: dict[str, dict[str, str]] = {}
    embodiment_per_cell: dict[str, str] = {}
    task_per_cell: dict[str, str] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            axes = json.loads(row["cell_axes"])
            embodiment = row["embodiment"]
            task = row["task_name"]
            slug = f"{embodiment}/{task}/" + ",".join(f"{k}={v}" for k, v in sorted(axes.items()))
            by_cell_slug.setdefault(slug, []).append((bool(int(row["success"])), axes))
            cell_meta[slug] = axes
            embodiment_per_cell[slug] = embodiment
            task_per_cell[slug] = task

    out: list[CellResult] = []
    for slug, trials in by_cell_slug.items():
        successes = [s for s, _ in trials]
        n = len(successes)
        s_count = sum(1 for x in successes if x)
        low, high = wilson_ci(s_count, n)
        out.append(CellResult(
            cell=CellId(embodiment=embodiment_per_cell[slug], task=task_per_cell[slug], axes=cell_meta[slug]),
            n_trials=n,
            successes=s_count,
            wilson_ci_low=low,
            wilson_ci_high=high,
            per_seed_success=successes,
        ))
    return out


@dataclass(frozen=True)
class ProfileReport:
    """The per-model summary the notebook renders."""

    sweep: CombinedSweep
    per_axis: dict[str, dict[str, float]]  # axis → level → mean success
    worst_axis_name: str
    worst_axis_score: float
    overall_success_rate: float

    @property
    def headline(self) -> str:
        return f"{self.sweep.model_name}: worst axis = {self.worst_axis_name} ({self.worst_axis_score:.2f})"


def profile_for_bundle(bundle: SweepBundle, *, task_key: str) -> list[ProfileReport]:
    out: list[ProfileReport] = []
    for sweep in bundle.by_task(task_key):
        if not sweep.cell_results:
            continue
        per_axis = per_axis_means(sweep.cell_results)
        # Filter axes that have only one level — they can't discriminate
        # and should not be eligible for "worst axis" (would always be
        # the same as overall mean). Match the wedge definition: we want
        # axes with variation.
        per_axis_multi = {a: lv for a, lv in per_axis.items() if len(lv) > 1}
        if per_axis_multi:
            axis_name, axis_score = worst_axis(per_axis_multi)
        else:
            # Single-cell task (WidowX platform validation): no per-axis story;
            # use the overall mean as the headline.
            axis_name = "overall"
            axis_score = sum(c.success_rate for c in sweep.cell_results) / len(sweep.cell_results)
        overall = sum(c.successes for c in sweep.cell_results) / sum(c.n_trials for c in sweep.cell_results)
        out.append(ProfileReport(
            sweep=sweep,
            per_axis=per_axis,
            worst_axis_name=axis_name,
            worst_axis_score=axis_score,
            overall_success_rate=overall,
        ))
    return out


@dataclass(frozen=True)
class CanonicalDimResult:
    """Per-canonical-dim summary for a single (model, task) sweep.

    `mean` is the pooled mean success rate over every cell whose axes
    contribute to this dim (i.e., the cell has at least one axis the
    Task's `canonical_axis_map` maps to this dim). `wilson_ci_low/high`
    are pooled-Wilson bounds over the underlying trials. `n_trials` and
    `n_cells` give the coverage so the notebook can render a coverage
    badge alongside the score.

    `dim` is the closed-enum CanonicalDim string. A dim with `n_cells == 0`
    means the Task declares no axes for it (e.g. WidowX has no
    language-mapped axis); the notebook renders such dims with a
    "not measured" indicator rather than a 0.0 bar.
    """

    dim: CanonicalDim
    mean: float
    wilson_ci_low: float
    wilson_ci_high: float
    n_trials: int
    n_cells: int


@dataclass(frozen=True)
class CanonicalProfile:
    """The cross-axis-comparable headline.

    One per (model, task) sweep. The four canonical dims always appear in
    the same order (`canonical_dims()`) so charts across models on the
    same task — and charts across tasks — share the same x-axis. Dims
    with `n_cells == 0` are placeholders; the worst-axis ranking excludes
    them.

    `worst_dim_name` is the canonical-dim with the lowest score among
    dims that have measured coverage. `worst_dim_score` is its mean.
    """

    sweep: CombinedSweep
    per_dim: dict[CanonicalDim, CanonicalDimResult]
    worst_dim_name: CanonicalDim | None
    worst_dim_score: float | None

    @property
    def headline(self) -> str:
        if self.worst_dim_name is None:
            return f"{self.sweep.model_name}: no canonical dims measured on this task"
        return (
            f"{self.sweep.model_name} on {self.sweep.embodiment}: "
            f"worst canonical dim = {self.worst_dim_name} "
            f"({self.worst_dim_score:.2f})"
        )


def _task_canonical_axis_map(sweep: CombinedSweep) -> dict[str, CanonicalDim]:
    """Recover the Task's `canonical_axis_map` from the sealed manifest.

    Schema 0.2.0+ binds the Task's `canonical_axis_map` into the hashed
    payload, so the published profile is provably linked to the mapping
    that produced it. We read directly from the manifest — no in-tree
    dict to maintain, third-party Tasks render correctly without a code
    change.

    For pre-0.2.0 manifests (which excluded the map from the hash and
    didn't always set it), we fall back to the in-tree reference dict so
    historical sweeps continue to render. Once those legacy manifests
    age out, the fallback can be deleted.
    """
    bound = sweep.manifest.canonical_axis_map
    if bound:
        # Cast: Manifest stores dict[str, str] (loosened from Literal at
        # serialization time); the runtime values are CanonicalDim
        # members. mypy can't see that without a cast.
        valid_dims = canonical_dims()
        return {axis: dim for axis, dim in bound.items() if dim in valid_dims}
    key = (sweep.embodiment, sweep.task_name)
    return _LEGACY_CANONICAL_AXIS_MAPS.get(key, {})


# Fallback for schema 0.1.0 manifests that predate canonical_axis_map
# being bound into the payload. Schema 0.2.0+ Tasks emit their own map;
# this dict is *only* read when the manifest's field is empty.
_LEGACY_CANONICAL_AXIS_MAPS: dict[tuple[str, str], dict[str, CanonicalDim]] = {
    ("google_robot", "pick_coke_can"): {
        "orientation": "physics",
        "lighting": "visuals",
        "background": "visuals",
        "distractor": "visuals",
        "table_texture": "visuals",
        "paraphrase": "language",
    },
    ("widowx", "spoon_on_towel"): {
        "condition": "visuals",
    },
    ("unitree_go1", "unitree_go1_joystick"): {
        "task_family": "language",
        "terrain": "physics",
        "camera": "visuals",
        "perturbation": "physics",
    },
}


def canonical_profile_from_manifest(manifest: Manifest) -> CanonicalProfile:
    """Compute the canonical-axis profile from a Manifest alone.

    The portal serves submitted manifests; it doesn't have the on-disk
    `trials.csv` that `canonical_profile_for_sweep` requires. But
    `Manifest.cells[*]` already carries per-cell `successes`, `n_trials`,
    and `axes` — every input the pooling logic needs. This helper
    synthesizes a transient `CombinedSweep` from the manifest's payload
    and delegates to `canonical_profile_for_sweep`, so the two paths
    share one implementation.

    For task pairs whose manifest doesn't bind a `canonical_axis_map`
    and that aren't in the legacy fallback, every dim returns
    `n_cells=0` and the rendered profile shows "not measured"
    everywhere — which is the honest answer.
    """
    synthetic_cells: list[CellResult] = []
    for cell_payload in manifest.cells:
        synthetic_cells.append(CellResult(
            cell=CellId(
                embodiment=manifest.embodiment,
                task=manifest.task_name,
                axes=dict(cell_payload.axes),
            ),
            n_trials=cell_payload.n_trials,
            successes=cell_payload.successes,
            wilson_ci_low=cell_payload.wilson_ci_low,
            wilson_ci_high=cell_payload.wilson_ci_high,
            per_seed_success=[],
        ))
    synthetic_sweep = CombinedSweep(
        label=f"{manifest.embodiment}_{manifest.task_name}",
        manifest=manifest,
        cell_results=synthetic_cells,
    )
    return canonical_profile_for_sweep(synthetic_sweep)


def canonical_profile_for_sweep(sweep: CombinedSweep) -> CanonicalProfile:
    """Compute the per-canonical-dim summary for one (model, task) sweep.

    For each canonical dim, pool successes + trials over every cell whose
    axes touch a Task-axis mapped to that dim. Compute the pooled mean
    and Wilson 95% CI. Dims with zero coverage on this Task get an
    explicit `n_cells == 0` marker (rendered as "not measured" in the
    notebook, excluded from worst-dim ranking).

    Cross-task aggregation is **not** done here; this function returns
    one profile per sweep. Cross-task aggregation is v1 (stratified
    bootstrap + deployment-relevance weighting).
    """
    axis_map = _task_canonical_axis_map(sweep)
    per_dim: dict[CanonicalDim, CanonicalDimResult] = {}
    for dim in canonical_dims():
        # Cells whose axes include at least one axis that maps to this dim.
        contributing: list[CellResult] = []
        for cr in sweep.cell_results:
            if any(axis_map.get(a) == dim for a in cr.cell.axes):
                contributing.append(cr)
        if not contributing:
            per_dim[dim] = CanonicalDimResult(
                dim=dim, mean=0.0, wilson_ci_low=0.0, wilson_ci_high=0.0,
                n_trials=0, n_cells=0,
            )
            continue
        s = sum(c.successes for c in contributing)
        n = sum(c.n_trials for c in contributing)
        if n == 0:
            per_dim[dim] = CanonicalDimResult(
                dim=dim, mean=0.0, wilson_ci_low=0.0, wilson_ci_high=0.0,
                n_trials=0, n_cells=len(contributing),
            )
            continue
        low, high = wilson_ci(s, n)
        per_dim[dim] = CanonicalDimResult(
            dim=dim, mean=s / n,
            wilson_ci_low=low, wilson_ci_high=high,
            n_trials=n, n_cells=len(contributing),
        )

    measured = [(d, r) for d, r in per_dim.items() if r.n_cells > 0 and r.n_trials > 0]
    if measured:
        worst = min(measured, key=lambda kv: kv[1].mean)
        worst_name: CanonicalDim | None = worst[0]
        worst_score: float | None = worst[1].mean
    else:
        worst_name = None
        worst_score = None

    return CanonicalProfile(
        sweep=sweep,
        per_dim=per_dim,
        worst_dim_name=worst_name,
        worst_dim_score=worst_score,
    )


def calibration_overlay(report: ProfileReport) -> dict[str, float | None]:
    """For tier-B/A reporting: look up published real perf for this
    (task, model). Returns {sim, real, delta} or {sim, real=None} if no
    real number is published.
    """
    task_lookup_key = report.sweep.task_key
    family = report.sweep.model_family
    # RT-1 ckpt name discriminator: RT-1-X has "rt_1_x" in the path;
    # RT-1-Converged is "rt_1_tf_trained_for_000400120". Both share family="rt1".
    if family == "rt1":
        model_lookup_key: str | None = "rt-1-x" if "rt_1_x" in report.sweep.model_name else "rt-1-converged"
    elif family == "octo":
        model_lookup_key = "octo-base"
    else:
        model_lookup_key = None
    sim = report.overall_success_rate
    real: float | None = None
    # Re-read at call time so the v2.0 pipeline updating the registry
    # mid-run gets picked up without needing to reimport this module.
    current = _load_real_perf()
    if task_lookup_key in current and model_lookup_key in current[task_lookup_key]:
        real = current[task_lookup_key][model_lookup_key]
    delta = (sim - real) if real is not None else None
    return {"sim": sim, "real": real, "delta": delta}
