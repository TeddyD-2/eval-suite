"""Manifest signing contract — v0 attestation.

**In plain words.** Pins down that the Ed25519 signature on a
manifest actually verifies, and that tampering with the manifest
breaks the verification. If this ever fails, anyone can forge a
submission's attribution.
"""

from __future__ import annotations

from eval_suite.manifest import (
    SCHEMA_VERSION,
    CalibrationRef,
    CellResultPayload,
    HardwareRef,
    Manifest,
    ModelRef,
    SimulatorRef,
)
from eval_suite.signing import generate_keypair, sign, to_hex, verify


def _example_manifest() -> Manifest:
    return Manifest(
        schema_version=SCHEMA_VERSION,
        code_sha="abcd1234",
        container_digest="",
        model=ModelRef(name="example", checkpoint_sha256="aa" * 32, family="example"),
        simulator=SimulatorRef(name="sim", commit="deadbeef"),
        task_name="t",
        embodiment="e",
        trials_per_cell=10,
        cells=[CellResultPayload(cell_id="e/t/k=v", axes={"k": "v"}, n_trials=10,
                                 successes=5, wilson_ci_low=0.2, wilson_ci_high=0.8)],
        hardware=HardwareRef(gpu="g", cuda="c", driver="d"),
        seeds=[0, 1, 2],
        calibration=CalibrationRef(tier="C"),
    )


def test_raw_signing_roundtrip() -> None:
    priv, pub = generate_keypair()
    assert len(priv) == 32 and len(pub) == 32
    payload = '{"hello":"world","n":42}'
    sig = sign(payload, priv)
    assert len(sig) == 128  # 64 bytes hex-encoded
    assert verify(payload, sig, pub)
    # tampering detection
    assert not verify(payload + " ", sig, pub)
    assert not verify(payload, "00" * 64, pub)


def test_manifest_sign_then_verify() -> None:
    priv, pub = generate_keypair()
    m = _example_manifest()
    m.seal()
    assert m.verify()  # unsigned manifest still verifies (tier-1 maintainer run)
    m.sign(priv, to_hex(pub), identity="alice@example.com")
    assert m.submitter_signature is not None
    assert m.submitter_public_key == to_hex(pub)
    assert m.submitter_identity == "alice@example.com"
    assert m.verify()  # signed manifest also verifies


def test_manifest_sign_detects_tampering() -> None:
    priv, pub = generate_keypair()
    m = _example_manifest()
    m.seal()
    m.sign(priv, to_hex(pub), identity="alice@example.com")
    assert m.verify()
    # Tamper with the cells — recompute the hash. The content-hash check
    # passes (we re-sealed), but the signature was over the OLD content.
    m.cells[0] = CellResultPayload(cell_id="e/t/k=v", axes={"k": "v"}, n_trials=10,
                                   successes=9, wilson_ci_low=0.5, wilson_ci_high=1.0)
    m.seal()
    assert not m.verify()  # signature invalid for tampered content


def test_manifest_sign_requires_seal_first() -> None:
    priv, pub = generate_keypair()
    m = _example_manifest()
    # Don't seal — sign should refuse.
    try:
        m.sign(priv, to_hex(pub), identity="x")
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "seal()" in str(e)


def test_manifest_json_roundtrip_preserves_signature() -> None:
    priv, pub = generate_keypair()
    m = _example_manifest()
    m.seal()
    m.sign(priv, to_hex(pub), identity="bob")
    payload = m.to_json()
    m2 = Manifest.from_json(payload)
    assert m2.verify()
    assert m2.submitter_identity == "bob"
    assert m2.submitter_signature == m.submitter_signature
