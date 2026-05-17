"""v0 cross-device corroboration contract.

Two submitters running the same pinned inputs produce byte-identical
manifest run_ids. When both submit to the same portal, the portal
records both (keyed by `(run_id, submitter_pk)`) and the new
`GET /submissions?run_id=X` endpoint returns both as corroboration
evidence.

This is the v0-level cross-device verification claim:
- Manifest content hash + Ed25519 submitter signature = each submitter
  proves they held the key and the manifest hash they submitted is
  consistent with their declared inputs.
- Two distinct submitters producing the same hash from the same
  declared inputs = independent corroboration.
- The portal's append-only ledger captures both events so a reviewer
  can audit which submitters have corroborated which run.

What it does NOT prove (deferred to v1.0 Sigstore):
- That the portal operator hasn't selectively dropped submissions.
- Non-repudiation over time (transparency log).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eval_suite.manifest import (
    SCHEMA_VERSION,
    CalibrationRef,
    CellResultPayload,
    HardwareRef,
    Manifest,
    ModelRef,
    SimulatorRef,
)
from eval_suite.portal import SubmissionStore, create_app
from eval_suite.signing import generate_keypair, to_hex
from fastapi.testclient import TestClient


def _identical_manifest() -> Manifest:
    """Build a manifest with deterministic content so two invocations
    produce the same run_id."""
    return Manifest(
        schema_version=SCHEMA_VERSION,
        code_sha="aaaa",
        container_digest="",
        model=ModelRef(name="m", checkpoint_sha256="cc" * 32, family="test"),
        simulator=SimulatorRef(name="sim", commit="abcd"),
        task_name="t",
        embodiment="e",
        trials_per_cell=2,
        cells=[CellResultPayload(cell_id="e/t/k=v", axes={"k": "v"}, n_trials=2,
                                 successes=1, wilson_ci_low=0.1, wilson_ci_high=0.9)],
        hardware=HardwareRef(gpu="g", cuda="c", driver="d"),
        seeds=[0, 1],
        calibration=CalibrationRef(tier="C"),
    )


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    store = SubmissionStore(tmp_path / "submissions")
    monkeypatch.setenv("EVAL_SUITE_ALLOWED_KEYS", str(tmp_path / "no-such.json"))
    return TestClient(create_app(store=store))


def test_same_inputs_produce_same_run_id() -> None:
    """Foundational claim: two independent seals of the same inputs
    produce byte-identical run_ids. Without this, corroboration is
    impossible."""
    m1 = _identical_manifest()
    m1.seal()
    m2 = _identical_manifest()
    m2.seal()
    assert m1.run_id == m2.run_id, "identical inputs must hash to the same run_id"


def test_two_submitters_same_run_id_both_recorded(client: TestClient) -> None:
    """Two submitters sign and POST the same manifest. The portal
    records both; `/submissions?run_id=X` returns both."""
    priv_a, pub_a = generate_keypair()
    priv_b, pub_b = generate_keypair()
    m_a = _identical_manifest()
    m_a.seal()
    m_a.sign(priv_a, to_hex(pub_a), identity="alice@example.com")
    m_b = _identical_manifest()
    m_b.seal()
    m_b.sign(priv_b, to_hex(pub_b), identity="bob@example.com")

    # The signed manifests have different content hashes because the
    # signature fields are part of the payload — wait, no, signatures
    # are excluded from the hashable payload. So the run_id IS the same.
    assert m_a.run_id == m_b.run_id

    r1 = client.post("/submit", json={"manifest": m_a.to_json()})
    assert r1.status_code == 201, r1.text
    assert r1.json()["corroborating_submitters"] == 0  # alice was first

    r2 = client.post("/submit", json={"manifest": m_b.to_json()})
    assert r2.status_code == 201, r2.text
    assert r2.json()["corroborating_submitters"] == 1  # bob sees alice's earlier submission

    # GET /submissions?run_id=<the run_id> returns both
    r = client.get(f"/submissions?run_id={m_a.run_id}")
    assert r.status_code == 200
    subs = r.json()["submissions"]
    submitter_ids = {s["submitter_identity"] for s in subs}
    assert submitter_ids == {"alice@example.com", "bob@example.com"}, submitter_ids
    assert r.json()["corroborating_submitters"] == 2


def test_ledger_records_each_submission(client: TestClient) -> None:
    """Every accept event appends one line to /ledger."""
    priv, pub = generate_keypair()
    m = _identical_manifest()
    m.seal()
    m.sign(priv, to_hex(pub), identity="alice@example.com")
    client.post("/submit", json={"manifest": m.to_json()})

    r = client.get("/ledger")
    assert r.status_code == 200
    lines = [ln for ln in r.text.splitlines() if ln.strip()]
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["run_id"] == m.run_id
    assert entry["submitter_identity"] == "alice@example.com"
    assert entry["accepted"] is True
