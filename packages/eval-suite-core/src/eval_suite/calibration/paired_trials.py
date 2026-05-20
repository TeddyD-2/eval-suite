"""PairedTrial sidecar — the on-disk artifact for trajectory-level calibration.

**In plain words.** When a partner lab pairs a sim rollout with a real
recorded rollout of the same condition, this is the file that captures
the pairing on disk — both trajectories' identities, the MMRV score
between them, and a signature so the pairing is auditable. It lives
next to the sweep's manifest as `paired_trials.json` and is what the
calibration tier upgrade from "tier B" (one real number) to "tier A"
(trajectory-paired data) is anchored on.


A `paired_trials.json` lives next to a sweep's `manifest.json`. Each
entry pairs one sim rollout (referenced by run_id + cell_id + seed)
with one real-side trajectory (referenced by an opaque
`real_episode_ref` like `oxe:bridge_dataset_v2#142` or
`lerobot:lerobot/aloha_static_coffee#5`) and records the computed
MMRV between them.

The sidecar is signed with the same Ed25519 machinery as the
manifest, so a partner lab's claimed paired data is auditable: the
signature covers the canonical JSON of `paired_trials` minus the
signature field.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .._types import CellId
from ..signing import sign as _sign
from .statistics import mmrv


@dataclass(frozen=True)
class PairedTrial:
    """One (sim_rollout, real_episode) pair + its computed MMRV."""

    cell_id_slug: str
    sim_run_id: str
    sim_seed: int
    real_episode_ref: str
    mmrv_score: float
    n_steps_compared: int
    notes: str = ""

    @classmethod
    def from_trajectories(
        cls,
        *,
        cell: CellId,
        sim_run_id: str,
        sim_seed: int,
        sim_traj: np.ndarray[Any, Any],
        real_traj: np.ndarray[Any, Any],
        real_episode_ref: str,
        dt: float = 0.05,
        notes: str = "",
    ) -> PairedTrial:
        """Compute MMRV between paired sim + real trajectories. If the
        trajectories differ in length, truncate to the shorter one."""
        n = min(sim_traj.shape[0], real_traj.shape[0])
        s = sim_traj[:n]
        r = real_traj[:n]
        return cls(
            cell_id_slug=cell.slug,
            sim_run_id=sim_run_id,
            sim_seed=int(sim_seed),
            real_episode_ref=real_episode_ref,
            mmrv_score=mmrv(s, r, dt=dt),
            n_steps_compared=int(n),
            notes=notes,
        )


@dataclass(frozen=True)
class PairedTrialsPayload:
    """The full sidecar contents."""

    manifest_run_id: str
    task_key: str
    model_key: str
    pairs: list[PairedTrial] = field(default_factory=list)
    pearson_r: float | None = None
    pearson_r_ci_low: float | None = None
    pearson_r_ci_high: float | None = None
    n_paired_cells: int = 0


def record_paired_trial(
    *,
    sidecar_path: Path,
    manifest_run_id: str,
    task_key: str,
    model_key: str,
    paired_trial: PairedTrial,
) -> Path:
    """Append a single PairedTrial to the sidecar at `sidecar_path`.

    Creates the file with the right header if it doesn't exist yet.
    Re-writes the canonical JSON atomically.
    """
    sidecar_path = Path(sidecar_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    if sidecar_path.exists():
        existing_raw = json.loads(sidecar_path.read_text())
        pairs = [PairedTrial(**p) for p in existing_raw.get("pairs", [])]
        existing = PairedTrialsPayload(
            manifest_run_id=existing_raw["manifest_run_id"],
            task_key=existing_raw["task_key"],
            model_key=existing_raw["model_key"],
            pairs=pairs,
            pearson_r=existing_raw.get("pearson_r"),
            pearson_r_ci_low=existing_raw.get("pearson_r_ci_low"),
            pearson_r_ci_high=existing_raw.get("pearson_r_ci_high"),
            n_paired_cells=existing_raw.get("n_paired_cells", 0),
        )
        if existing.manifest_run_id != manifest_run_id:
            raise ValueError(
                f"sidecar at {sidecar_path} is bound to manifest "
                f"{existing.manifest_run_id!r}; refusing to mix with "
                f"{manifest_run_id!r}"
            )
        pairs = list(existing.pairs) + [paired_trial]
    else:
        pairs = [paired_trial]

    payload = PairedTrialsPayload(
        manifest_run_id=manifest_run_id,
        task_key=task_key,
        model_key=model_key,
        pairs=pairs,
    )
    sidecar_path.write_text(
        json.dumps(_payload_to_dict(payload), sort_keys=True, indent=2)
    )
    return sidecar_path


def _payload_to_dict(p: PairedTrialsPayload) -> dict[str, Any]:
    return {
        "manifest_run_id": p.manifest_run_id,
        "task_key": p.task_key,
        "model_key": p.model_key,
        "pairs": [asdict(t) for t in p.pairs],
        "pearson_r": p.pearson_r,
        "pearson_r_ci_low": p.pearson_r_ci_low,
        "pearson_r_ci_high": p.pearson_r_ci_high,
        "n_paired_cells": p.n_paired_cells,
    }


def paired_trials_sidecar_path(run_dir: Path) -> Path:
    """Conventional sidecar location: next to manifest.json."""
    return Path(run_dir) / "paired_trials.json"


def sign_sidecar(sidecar_path: Path, private_key_hex: str) -> str:
    """Sign the sidecar's canonical JSON content. Returns the signature."""
    canonical = json.dumps(
        json.loads(sidecar_path.read_text()), sort_keys=True, separators=(",", ":")
    )
    return _sign(canonical, bytes.fromhex(private_key_hex))
