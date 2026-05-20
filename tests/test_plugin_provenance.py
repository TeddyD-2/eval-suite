"""v0 plugin-provenance sidecar contract tests.

**In plain words.** Pins down the "which pip packages produced
this run" sidecar. A future reviewer can trust the paper trail
because the tamper-detection here works.

Covers:
- Sidecar serializes + round-trips through JSON.
- Unsigned sidecar verifies True.
- Sign / verify roundtrip with Ed25519 keypair.
- Tampering (changing any signed field) breaks verify().
- build_for_run resolves the (Task, Policy, Adapter) instances back to
  their entry-point packages.
"""

from __future__ import annotations

from pathlib import Path

from eval_suite.adapters import GymAdapter
from eval_suite.contracts import CONTRACT_VERSION
from eval_suite.plugin_provenance import PluginProvenance, PluginRef, build_for_run
from eval_suite.policies.mock import MockPolicy
from eval_suite.signing import generate_keypair, to_hex
from eval_suite.tasks.mock import MockTask


def _example_sidecar() -> PluginProvenance:
    return PluginProvenance(
        contract_version=CONTRACT_VERSION,
        target_run_id="abc123" * 11,  # fake 66-char hex
        components={
            "task":    PluginRef(package_name="eval-suite", package_version="0.1.0",
                                 entry_point_name="mock", contract_version_target="1.0.0"),
            "policy":  PluginRef(package_name="eval-suite", package_version="0.1.0",
                                 entry_point_name="mock", contract_version_target="1.0.0"),
            "adapter": PluginRef(package_name="eval-suite", package_version="0.1.0",
                                 entry_point_name="gym", contract_version_target="1.0.0"),
        },
        eval_suite_version="0.1.0",
    )


def test_sidecar_round_trips_through_json(tmp_path: Path) -> None:
    sidecar = _example_sidecar()
    sidecar.save(tmp_path / "plugin_provenance.json")
    loaded = PluginProvenance.load(tmp_path / "plugin_provenance.json")
    assert loaded.target_run_id == sidecar.target_run_id
    assert loaded.contract_version == sidecar.contract_version
    assert set(loaded.components) == {"task", "policy", "adapter"}
    assert loaded.components["task"].entry_point_name == "mock"


def test_unsigned_sidecar_verifies() -> None:
    sidecar = _example_sidecar()
    assert sidecar.verify() is True


def test_sign_verify_roundtrip() -> None:
    sidecar = _example_sidecar()
    priv, pub = generate_keypair()
    sidecar.sign(priv, to_hex(pub), identity="alice@example.com")
    assert sidecar.submitter_signature is not None
    assert sidecar.submitter_public_key == to_hex(pub)
    assert sidecar.submitter_identity == "alice@example.com"
    assert sidecar.verify() is True


def test_tampering_breaks_signed_verify() -> None:
    sidecar = _example_sidecar()
    priv, pub = generate_keypair()
    sidecar.sign(priv, to_hex(pub), identity="alice@example.com")
    # Tamper with one component's version
    sidecar.components["task"] = PluginRef(
        package_name="eval-suite", package_version="9.9.9",  # ← changed
        entry_point_name="mock", contract_version_target="1.0.0",
    )
    assert sidecar.verify() is False


def test_build_for_run_resolves_entry_points() -> None:
    """build_for_run should match each instance's class back to its entry-point."""
    task = MockTask()
    policy = MockPolicy()
    adapter = GymAdapter()
    sidecar = build_for_run(
        task=task, policy=policy, adapter=adapter,
        target_run_id="deadbeef" * 8, contract_version=CONTRACT_VERSION,
    )
    assert sidecar.target_run_id == "deadbeef" * 8
    assert sidecar.components["task"].package_name == "eval-suite-stdlib"
    assert sidecar.components["task"].entry_point_name == "mock"
    assert sidecar.components["policy"].package_name == "eval-suite-stdlib"
    assert sidecar.components["adapter"].entry_point_name == "gym"
