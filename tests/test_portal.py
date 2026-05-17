"""Portal end-to-end contract — POST /submit, GET /submissions, GET /<id>."""

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


def _manifest_factory(name: str = "alice-model") -> Manifest:
    return Manifest(
        schema_version=SCHEMA_VERSION,
        code_sha="cccc",
        container_digest="",
        model=ModelRef(name=name, checkpoint_sha256="aa" * 32, family="example"),
        simulator=SimulatorRef(name="sim", commit="0123abcd"),
        task_name="t",
        embodiment="e",
        trials_per_cell=3,
        cells=[CellResultPayload(cell_id="e/t/k=v", axes={"k": "v"}, n_trials=3,
                                 successes=2, wilson_ci_low=0.25, wilson_ci_high=0.95)],
        hardware=HardwareRef(gpu="g", cuda="c", driver="d"),
        seeds=[0, 1, 2],
        calibration=CalibrationRef(tier="C"),
    )


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Make sure each test gets an isolated store + no allow-list (open mode).
    store = SubmissionStore(tmp_path / "submissions")
    monkeypatch.setenv("EVAL_SUITE_ALLOWED_KEYS", str(tmp_path / "no-such-file.json"))
    return TestClient(create_app(store=store))


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_submit_unsigned_in_open_mode(client: TestClient) -> None:
    """No allow-list configured → unsigned manifests accepted as long as
    content hash verifies."""
    m = _manifest_factory()
    m.seal()
    r = client.post("/submit", json={"manifest": m.to_json()})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["accepted"]
    assert body["run_id"] == m.run_id


def test_submit_signed(client: TestClient) -> None:
    priv, pub = generate_keypair()
    m = _manifest_factory()
    m.seal()
    m.sign(priv, to_hex(pub), identity="alice@example.com")
    r = client.post("/submit", json={"manifest": m.to_json()})
    assert r.status_code == 201
    assert r.json()["submitter_identity"] == "alice@example.com"


def test_submit_rejects_tampered(client: TestClient) -> None:
    m = _manifest_factory()
    m.seal()
    # Mutate the payload after seal so the hash no longer matches.
    payload = m.to_json()
    obj = json.loads(payload)
    obj["cells"][0]["successes"] = 99  # forge a higher success count
    bad = json.dumps(obj)
    r = client.post("/submit", json={"manifest": bad})
    assert r.status_code == 400
    assert "verify" in r.json()["detail"]


def test_allowlist_enforced(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Configure an allow-list with one specific key.
    priv_allowed, pub_allowed = generate_keypair()
    priv_other, pub_other = generate_keypair()
    keys_path = tmp_path / "allowed.json"
    keys_path.write_text(json.dumps({to_hex(pub_allowed): "alice-via-allowlist"}))
    monkeypatch.setenv("EVAL_SUITE_ALLOWED_KEYS", str(keys_path))

    # 1) Allowed key — accepted; identity overridden with allow-list value.
    m1 = _manifest_factory("from-alice")
    m1.seal()
    m1.sign(priv_allowed, to_hex(pub_allowed), identity="spoof-attempt")
    r1 = client.post("/submit", json={"manifest": m1.to_json()})
    assert r1.status_code == 201, r1.text
    assert r1.json()["submitter_identity"] == "alice-via-allowlist"  # spoof rejected

    # 2) Unsigned with allow-list enforced — rejected.
    m2 = _manifest_factory("unsigned")
    m2.seal()
    r2 = client.post("/submit", json={"manifest": m2.to_json()})
    assert r2.status_code == 403

    # 3) Different key, not on allow-list — rejected.
    m3 = _manifest_factory("from-eve")
    m3.seal()
    m3.sign(priv_other, to_hex(pub_other), identity="eve")
    r3 = client.post("/submit", json={"manifest": m3.to_json()})
    assert r3.status_code == 403


def test_list_and_get(client: TestClient) -> None:
    m = _manifest_factory("listme")
    m.seal()
    client.post("/submit", json={"manifest": m.to_json()})

    listed = client.get("/submissions").json()["submissions"]
    assert len(listed) == 1
    assert listed[0]["run_id"] == m.run_id

    detail = client.get(f"/submissions/{m.run_id}").json()
    assert detail["submission"]["accepted"] is True
    assert detail["manifest"]["model"]["name"] == "listme"

    miss = client.get("/submissions/nonexistent-run-id")
    assert miss.status_code == 404


def test_malformed_manifest_returns_400(client: TestClient) -> None:
    r = client.post("/submit", json={"manifest": "{not-valid-json}"})
    assert r.status_code == 400
    assert "malformed" in r.json()["detail"]

    r2 = client.post("/submit", json={})  # missing field
    assert r2.status_code == 400


def test_registry_endpoints(client: TestClient) -> None:
    """The v0 /registry/* endpoints return the installed plugin catalog."""
    r = client.get("/registry/tasks")
    assert r.status_code == 200
    names = {t["name"] for t in r.json()["tasks"]}
    assert "mock" in names
    assert "google_robot_pick_coke_can" in names

    r = client.get("/registry/policies")
    assert r.status_code == 200
    assert any(p["name"] == "mock" for p in r.json()["policies"])

    r = client.get("/registry/adapters")
    assert r.status_code == 200
    assert any(a["name"] == "gym" for a in r.json()["adapters"])

    r = client.get("/registry/failed")
    assert r.status_code == 200
    # Clean install: no failed plugins
    assert r.json()["failed"] == []


def test_index_renders(client: TestClient) -> None:
    """GET /ui/ returns the v0 multi-page landing HTML.

    `/` (root) now 307-redirects to `/ui/`. We hit /ui/ directly to
    avoid TestClient's follow-redirects-by-default subtlety.
    """
    r = client.get("/ui/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    body = r.text
    assert "eval-suite portal" in body
    assert "Installed plugins" in body
    assert "tasks" in body and "policies" in body and "adapters" in body
    assert "CONTRACT_VERSION" in body
    # Trust-model note lives on /ui/about; surface it on the home page
    # via the nav link.
    assert "/ui/about" in body


def test_root_redirects_to_ui(client: TestClient) -> None:
    """`/` issues a 307 to `/ui/`."""
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/ui/"


def test_static_css_serves(client: TestClient) -> None:
    """The static stylesheet is mounted and reachable."""
    r = client.get("/static/style.css")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/css")
    assert "Courier New" in r.text


def test_submissions_list_renders_filters(client: TestClient) -> None:
    """POST two manifests with distinct models; GET /ui/submissions?model=alice
    should return only the matching row."""
    m1 = _manifest_factory("alice-model")
    m1.seal()
    client.post("/submit", json={"manifest": m1.to_json()})
    m2 = _manifest_factory("bob-model")
    m2.seal()
    client.post("/submit", json={"manifest": m2.to_json()})

    r_all = client.get("/ui/submissions")
    assert r_all.status_code == 200
    assert "alice-model" in r_all.text and "bob-model" in r_all.text

    r_filt = client.get("/ui/submissions?model=alice")
    assert r_filt.status_code == 200
    assert "alice-model" in r_filt.text
    assert "bob-model" not in r_filt.text


def test_submission_detail_renders(client: TestClient) -> None:
    """POST a manifest, GET /ui/submissions/<run_id>, confirm the
    canonical profile section renders + corroborator section appears."""
    m = _manifest_factory("detail-model")
    m.seal()
    client.post("/submit", json={"manifest": m.to_json()})

    r = client.get(f"/ui/submissions/{m.run_id}")
    assert r.status_code == 200
    body = r.text
    assert m.run_id[:16] in body
    assert "detail-model" in body
    assert "canonical generalization profile" in body
    assert "Manifest.verify()" in body
    assert "corroborators" in body


def test_submission_detail_404(client: TestClient) -> None:
    r = client.get("/ui/submissions/nonexistent-id")
    assert r.status_code == 404
    assert "no submission" in r.text


def test_compare_renders_side_by_side(client: TestClient) -> None:
    """POST two distinct manifests; GET /ui/compare?a=<id_a>&b=<id_b>
    should show both run_ids and the per-dim rows."""
    m_a = _manifest_factory("comp-a")
    m_a.seal()
    client.post("/submit", json={"manifest": m_a.to_json()})
    m_b = _manifest_factory("comp-b")
    m_b.seal()
    client.post("/submit", json={"manifest": m_b.to_json()})

    r = client.get(f"/ui/compare?a={m_a.run_id}&b={m_b.run_id}")
    assert r.status_code == 200
    body = r.text
    assert m_a.run_id[:16] in body
    assert m_b.run_id[:16] in body
    # The canonical-dim rows appear
    for dim in ("language", "visuals", "physics", "embodiment"):
        assert dim in body


def test_compare_picker_when_no_params(client: TestClient) -> None:
    """No params → render the dropdown form."""
    m = _manifest_factory("picker")
    m.seal()
    client.post("/submit", json={"manifest": m.to_json()})

    r = client.get("/ui/compare")
    assert r.status_code == 200
    # The form has both <select name="a"> and <select name="b">
    assert 'name="a"' in r.text and 'name="b"' in r.text


def test_about_page_renders(client: TestClient) -> None:
    r = client.get("/ui/about")
    assert r.status_code == 200
    body = r.text
    assert "CONTRACT_VERSION" in body
    assert "compare page is asserting" in body
    # The trust-model phrase may be split by <strong> tags; check both spellings.
    assert "cryptographically tamper-evident" in body
