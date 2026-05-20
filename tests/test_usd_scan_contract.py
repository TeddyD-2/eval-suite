"""End-to-end v0 contract: NamaqualandScanTask + AssetProvenance sidecar.

**In plain words.** Pins down that a real-world scan flows
end-to-end through the unchanged sweep pipeline and that the
asset provenance sidecar correctly binds to the run's
fingerprint. The "real captures plug in here" promise rests on
this test passing.

Asserts:
1. NamaqualandScanTask (via the mock factory) satisfies the Task Protocol.
2. The optional Adapter hooks (`instruction_for`, `extract_image`) are present.
3. `run_sweep()` drives the single-cell rollout through `MujocoPlaygroundAdapter`,
   produces a verifying manifest, and writes the v0 per-rollout artifacts.
4. `AssetProvenance` binds to `manifest.run_id`, re-hashes assets on disk
   in `verify()`, and detects tampering.
5. Optional Ed25519 signing of the sidecar roundtrips (sign → save → load → verify).

Tests run in `.venv-ci` (no `mujoco`, no `mujoco_playground`); the real
build_env path is exercised by `examples/namaqualand_sweep.py` against
the converted assets in `.venv-mjx`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from eval_suite._types import CellId, JointAction, Observation
from eval_suite.adapters import MujocoPlaygroundAdapter
from eval_suite.asset_provenance import AssetProvenance, AssetRef, write_for_run
from eval_suite.contracts import Task
from eval_suite.hashing import sha256_file
from eval_suite.policies import RandomLocomotionPolicy
from eval_suite.signing import generate_keypair
from eval_suite.sweep import run_sweep
from eval_suite.tasks.usd_scan import (
    GO1_ACTION_DIM,
    TASK_EMBODIMENT,
    TASK_NAME,
    mock_namaqualand_task_factory,
)


def test_namaqualand_task_satisfies_protocol() -> None:
    task = mock_namaqualand_task_factory(max_episode_steps=5)
    assert isinstance(task, Task)
    assert task.name == TASK_NAME
    assert task.embodiment == TASK_EMBODIMENT
    assert task.n_cells == 1
    assert task.max_episode_steps == 5
    cell = task.cell_id(0)
    assert isinstance(cell, CellId)
    assert cell.embodiment == "unitree_go1"
    assert cell.task == TASK_NAME
    assert cell.axes == {}


def test_namaqualand_task_optional_hooks_present() -> None:
    task = mock_namaqualand_task_factory()
    assert callable(getattr(task, "instruction_for", None))
    assert callable(getattr(task, "extract_image", None))
    env = task.build_env(0)
    assert "stand on the scanned boulder" in task.instruction_for(env)
    img = task.extract_image(env, obs=None)
    assert img.ndim == 3 and img.shape[-1] == 3 and img.dtype == np.uint8


def test_sweep_through_mujoco_adapter_produces_verifying_manifest(tmp_path: Path) -> None:
    """Drive `run_sweep()` end-to-end with the mock factory; assert
    manifest + CSV shape match v0 conventions for this Task."""
    task = mock_namaqualand_task_factory(max_episode_steps=5)
    policy = RandomLocomotionPolicy(action_dim=GO1_ACTION_DIM, seed=7)
    adapter = MujocoPlaygroundAdapter()

    manifest = run_sweep(
        policy=policy,
        task=task,
        adapter=adapter,
        seeds=[0, 1, 2],
        output_dir=tmp_path,
        code_sha="v0-contract-test",
        notes="v0 contract test",
    )

    assert manifest.verify()
    assert manifest.task_name == TASK_NAME
    assert manifest.embodiment == TASK_EMBODIMENT
    assert manifest.trials_per_cell == 3
    assert len(manifest.cells) == 1
    cell_payload = manifest.cells[0]
    assert cell_payload.n_trials == 3
    assert cell_payload.axes == {}

    # CSV schema: 3 rows (1 cell × 3 seeds), random-locomotion policy.
    rows = list(csv.DictReader((tmp_path / "trials.csv").open()))
    assert len(rows) == 3
    assert all(r["embodiment"] == TASK_EMBODIMENT for r in rows)
    assert all(r["task_name"] == TASK_NAME for r in rows)
    assert all(r["model_name"].startswith("random-locomotion") for r in rows)
    for r in rows:
        assert json.loads(r["cell_axes"]) == {}

    # Also confirm the policy emitted JointAction shape end-to-end.
    pol = RandomLocomotionPolicy(action_dim=GO1_ACTION_DIM, seed=0)
    pol.reset("smoke")
    out = pol.step(Observation(image=np.zeros((1, 1, 3), dtype=np.uint8), instruction="x"))
    assert isinstance(out, JointAction) and out.vector.shape == (GO1_ACTION_DIM,)


def test_asset_provenance_binds_and_detects_tamper(tmp_path: Path) -> None:
    """write_for_run binds a sidecar to a manifest.run_id; verify()
    re-hashes assets on disk and returns False after tampering."""
    asset_dir = tmp_path / "fake_assets"
    asset_dir.mkdir()
    src = asset_dir / "source.usdc"
    src.write_bytes(b"FAKE USD BYTES")
    refs = [AssetRef(
        role="source_usd",
        path=str(src.relative_to(tmp_path)),
        sha256=sha256_file(src),
        origin_url="https://example.com/scan.usdc",
        license="CC0-1.0",
    )]
    out_dir = tmp_path / "run_out"
    out_dir.mkdir()

    ap = write_for_run(target_run_id="abc123def456", assets=refs, output_dir=out_dir)
    assert (out_dir / "asset_provenance.json").exists()
    assert ap.target_run_id == "abc123def456"

    # Unsigned verify against the temp repo root succeeds.
    assert ap.verify(repo_root=tmp_path)

    # Tamper the asset: SHA256 mismatch → verify False.
    src.write_bytes(b"TAMPERED")
    assert not ap.verify(repo_root=tmp_path)

    # Restore: verify passes again.
    src.write_bytes(b"FAKE USD BYTES")
    assert ap.verify(repo_root=tmp_path)

    # Missing asset: verify False (returns; does not raise).
    src.unlink()
    assert not ap.verify(repo_root=tmp_path)

    # Roundtrip save/load preserves the binding and asset list.
    src.write_bytes(b"FAKE USD BYTES")
    loaded = AssetProvenance.load(out_dir / "asset_provenance.json")
    assert loaded.target_run_id == "abc123def456"
    assert len(loaded.assets) == 1 and loaded.assets[0].role == "source_usd"
    assert loaded.verify(repo_root=tmp_path)


def test_asset_provenance_optional_signing_roundtrip(tmp_path: Path) -> None:
    """Reuses signing.py's Ed25519 primitive — same surface as
    manifest.sign + plugin_provenance.sign. Sign + verify + tamper-detect."""
    asset = tmp_path / "scan.usdc"
    asset.write_bytes(b"signed-asset-bytes")
    refs = [AssetRef(role="source_usd", path=asset.name, sha256=sha256_file(asset),
                     origin_url="https://example.com", license="CC0-1.0")]

    ap = AssetProvenance(target_run_id="run-xyz", assets=refs)
    priv, pub = generate_keypair()
    ap.sign(priv, pub.hex(), "tester@example.com")
    assert ap.submitter_signature is not None
    assert ap.submitter_public_key == pub.hex()
    assert ap.submitter_identity == "tester@example.com"
    assert ap.verify(repo_root=tmp_path)

    # Reload from disk preserves the signature.
    out_path = tmp_path / "asset_provenance.json"
    ap.save(out_path)
    loaded = AssetProvenance.load(out_path)
    assert loaded.submitter_signature == ap.submitter_signature
    assert loaded.verify(repo_root=tmp_path)

    # Wrong public key → verify False (signature can't match).
    _, other_pub = generate_keypair()
    loaded.submitter_public_key = other_pub.hex()
    assert not loaded.verify(repo_root=tmp_path)
