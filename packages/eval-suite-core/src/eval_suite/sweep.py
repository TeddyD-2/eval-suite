"""Per-(model, task) sweep driver.

Structure:
  load model once → for cell in cells: for seed in seeds: rollout()
                  → emit per-trial CSV row immediately (resumable)
                  → at end, aggregate to per-cell Wilson CIs, seal manifest

Per the amortization benchmark in scripts/bench_amortization.py, both
RT-1 (TF) and Octo (JAX) recover ~50–70% of their first-trial wall
time after the first compile. The structure here keeps the model
process-resident across all cells and seeds in a given (model, task)
sweep.

CSV is append-mode so we can spot-check progress in tmux without
interrupting, and resume from the last completed trial if the sweep
crashes mid-run.
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import subprocess
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ._types import CellId, CellResult, RolloutResult
from .contracts import Adapter, Policy, Task
from .manifest import (
    SCHEMA_VERSION,
    CalibrationRef,
    CellResultPayload,
    HardwareRef,
    Manifest,
    ModelRef,
    SimulatorRef,
)
from .statistics import wilson_ci

log = logging.getLogger("eval_suite.sweep")


CSV_COLUMNS = [
    "timestamp_utc",
    "model_name",
    "model_checkpoint_id",
    "embodiment",
    "task_name",
    "cell_index",
    "cell_axes",  # JSON-encoded dict
    "seed",
    "success",
    "num_steps",
    "elapsed_wall_seconds",
    "episode_stats",  # JSON-encoded dict
    "video_path",  # relative path under the sweep's videos dir, or empty
]


def _git_sha(repo_dir: str | Path) -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def _detect_hardware() -> HardwareRef:
    gpu = "unknown"
    driver = "unknown"
    cuda = "unknown"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        first = out.splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        if len(parts) >= 2:
            gpu, driver = parts[0], parts[1]
    except Exception:
        pass
    try:
        nvcc = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.DEVNULL).decode()
        for line in nvcc.splitlines():
            if "release" in line:
                # e.g. "Cuda compilation tools, release 12.4, V12.4.131"
                cuda = line.split("release")[1].split(",")[0].strip()
    except Exception:
        pass
    return HardwareRef(gpu=gpu, cuda=cuda, driver=driver)


def _resolve_simulator_ref() -> SimulatorRef:
    aux: dict[str, str] = {}
    simpler_root = Path("/home/teddy/simpler-env")
    sim_commit = _git_sha(simpler_root) if simpler_root.exists() else "unknown"
    ms2_sub = simpler_root / "ManiSkill2_real2sim"
    if ms2_sub.exists():
        aux["maniskill2_real2sim"] = _git_sha(ms2_sub)
    return SimulatorRef(name="simpler-env", commit=sim_commit, auxiliary_commits=aux)


def _now_utc_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _ensure_csv(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        csv.writer(f).writerow(CSV_COLUMNS)


def _append_csv_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="") as f:
        csv.writer(f).writerow([row[c] for c in CSV_COLUMNS])


def _aggregate_to_cell_results(cell: CellId, rollouts: list[RolloutResult]) -> CellResult:
    successes = [r.success for r in rollouts]
    n = len(successes)
    s = sum(1 for x in successes if x)
    low, high = wilson_ci(s, n)
    return CellResult(
        cell=cell,
        n_trials=n,
        successes=s,
        wilson_ci_low=low,
        wilson_ci_high=high,
        per_seed_success=successes,
    )


def _cell_payload(cr: CellResult) -> CellResultPayload:
    return CellResultPayload(
        cell_id=cr.cell.slug,
        axes=dict(cr.cell.axes),
        n_trials=cr.n_trials,
        successes=cr.successes,
        wilson_ci_low=cr.wilson_ci_low,
        wilson_ci_high=cr.wilson_ci_high,
    )


def _model_family(name: str) -> str:
    n = name.lower()
    if "octo" in n:
        return "octo"
    if "rt1" in n or "rt-1" in n:
        return "rt1"
    return "unknown"


def run_sweep(
    *,
    policy: Policy,
    task: Task,
    adapter: Adapter,
    seeds: Iterable[int],
    output_dir: str | Path,
    code_sha: str | None = None,
    container_digest: str = "",
    calibration: CalibrationRef | None = None,
    notes: str = "",
    cells: Iterable[int] | None = None,
) -> Manifest:
    """Run a full sweep for one (policy, task). Returns the sealed manifest.

    `output_dir` receives:
      - `trials.csv` (append-mode; one row per rollout)
      - `manifest.json` (written at the end)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "trials.csv"
    manifest_path = out / "manifest.json"
    _ensure_csv(csv_path)

    seeds_list = list(seeds)
    cells_list = list(cells) if cells is not None else list(range(task.n_cells))

    code_sha = code_sha or _git_sha(Path(__file__).resolve().parents[1])
    hardware = _detect_hardware()
    simulator = _resolve_simulator_ref()

    # Optional pre-flight: if the Adapter declares a `can_drive`
    # method, ask it whether this Policy + Task combo is known to be
    # incompatible BEFORE wasting compute on trial 0. The method is
    # discovered via getattr so Adapters that don't implement it stay
    # contract-valid (the GymAdapter/MujocoPlaygroundAdapter's runtime
    # _flatten_action TypeError remains the safety net).
    can_drive = getattr(adapter, "can_drive", None)
    if callable(can_drive):
        report = can_drive(policy, task)
        if not getattr(report, "ok", True):
            raise RuntimeError(
                f"Adapter '{adapter.name}' refused this (policy, task): "
                f"{getattr(report, 'reason', 'no reason given')}. "
                f"This is the pre-flight check; see CompatibilityReport docstring "
                f"for what 'ok=False' means (conservative — refused on a known mismatch)."
            )

    sweep_start = time.monotonic()
    log.info(
        "sweep start: policy=%s task=%s embodiment=%s cells=%d seeds=%d output=%s",
        policy.name, task.name, task.embodiment, len(cells_list), len(seeds_list), out,
    )

    all_cell_results: list[CellResult] = []
    for cell_idx, cell_no in enumerate(cells_list):
        cell_id = task.cell_id(cell_no)
        cell_rollouts: list[RolloutResult] = []
        cell_start = time.monotonic()
        for seed_idx, seed in enumerate(seeds_list):
            trial_start = time.monotonic()
            result = adapter.rollout(policy=policy, task=task, cell=cell_no, seed=seed)
            cell_rollouts.append(result)
            row: dict[str, Any] = {
                "timestamp_utc": _now_utc_iso(),
                "model_name": policy.name,
                "model_checkpoint_id": policy.checkpoint_id,
                "embodiment": task.embodiment,
                "task_name": task.name,
                "cell_index": cell_no,
                "cell_axes": json.dumps(dict(cell_id.axes), sort_keys=True),
                "seed": seed,
                "success": int(result.success),
                "num_steps": result.num_steps,
                "elapsed_wall_seconds": round(result.elapsed_wall_seconds, 3),
                "episode_stats": json.dumps(dict(result.episode_stats), sort_keys=True, default=str),
                "video_path": result.video_path or "",
            }
            _append_csv_row(csv_path, row)
            log.info(
                "[%s/%s] cell=%d seed=%d success=%s steps=%d wall=%.1fs",
                seed_idx + 1, len(seeds_list), cell_no, seed,
                result.success, result.num_steps, time.monotonic() - trial_start,
            )
        cell_result = _aggregate_to_cell_results(cell_id, cell_rollouts)
        all_cell_results.append(cell_result)
        log.info(
            "[cell %d/%d done] %s — successes=%d/%d (%.2f, CI [%.2f, %.2f]) cell_wall=%.1fs",
            cell_idx + 1, len(cells_list), cell_id.slug,
            cell_result.successes, cell_result.n_trials, cell_result.success_rate,
            cell_result.wilson_ci_low, cell_result.wilson_ci_high,
            time.monotonic() - cell_start,
        )

    # Compute checkpoint ref. For Octo (HF), checkpoint_id starts with "hf:";
    # we extract the revision (no specific HF SHA pinning today).
    cp_id = policy.checkpoint_id
    if cp_id.startswith("sha256:"):
        ckpt_sha = cp_id[len("sha256:"):]
        hf_rev = None
    elif cp_id.startswith("hf:"):
        ckpt_sha = ""
        hf_rev = cp_id[len("hf:"):]
    else:
        ckpt_sha = cp_id
        hf_rev = None

    model_ref = ModelRef(
        name=policy.name,
        checkpoint_sha256=ckpt_sha,
        huggingface_revision=hf_rev,
        family=_model_family(policy.name),
    )

    # Capture the Task's canonical_axis_map so the manifest binds
    # the published profile to the mapping that produced it. Tasks that
    # don't declare a map (older or contract-test Tasks) record {}.
    canonical_map_raw = getattr(task, "canonical_axis_map", {}) or {}
    canonical_map: dict[str, str] = {str(k): str(v) for k, v in dict(canonical_map_raw).items()}

    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        code_sha=code_sha,
        container_digest=container_digest,
        model=model_ref,
        simulator=simulator,
        task_name=task.name,
        embodiment=task.embodiment,
        trials_per_cell=len(seeds_list),
        cells=[_cell_payload(cr) for cr in all_cell_results],
        hardware=hardware,
        seeds=list(seeds_list),
        calibration=calibration or CalibrationRef(tier="C"),
        notes=notes or f"python={platform.python_version()}",
        canonical_axis_map=canonical_map,
    )
    manifest.seal()
    manifest_path.write_text(manifest.to_json())

    # Write a sidecar `plugin_provenance.json` that records which
    # pip packages produced this run. This is NOT part of the manifest's
    # hash; it lives next to the manifest as an independent file. Plugin
    # author who bumps their version number wants the manifest's run_id
    # to stay stable across that bump (their *code* didn't change, just
    # the metadata); the sidecar carries the version-drift signal so a
    # reviewer can still trace which versions ran.
    from .contracts import CONTRACT_VERSION
    from .plugin_provenance import build_for_run

    provenance = build_for_run(
        task=task,
        policy=policy,
        adapter=adapter,
        target_run_id=manifest.run_id,
        contract_version=CONTRACT_VERSION,
    )
    provenance.save(out / "plugin_provenance.json")

    total_wall = time.monotonic() - sweep_start
    log.info(
        "sweep done: %d trials in %.1fs (%.1fs avg) → %s  run_id=%s",
        len(seeds_list) * len(cells_list), total_wall, total_wall / max(1, len(seeds_list) * len(cells_list)),
        manifest_path, manifest.run_id[:16],
    )
    return manifest
