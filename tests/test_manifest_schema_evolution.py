"""Schema-evolution contract tests for the 0.2.0 → 0.3.0 transition.

The 0.3.0 bump adds `Manifest.success_criterion` to support declarative
goal definitions for splat scenes (factory engineer writes a region +
picks a predicate; no new Task subclass). The bump is only safe to
release if it's TRULY ADDITIVE — i.e., every pre-existing 0.1.0/0.2.0
manifest continues to verify byte-identically under its original
hashing rules.

These tests pin the property explicitly. A regression here means the
substrate's content-addressing contract is broken; do not ship a build
that fails this file.
"""

from __future__ import annotations

import json
from typing import Any

from eval_suite.hashing import canonical_json, hash_dict
from eval_suite.manifest import (
    SCHEMA_VERSION,
    CalibrationRef,
    CellResultPayload,
    HardwareRef,
    Manifest,
    ModelRef,
    SimulatorRef,
)

# Reference run_id captured from the codebase at SCHEMA_VERSION="0.2.0",
# BEFORE the 0.3.0 bump. The fixture below MUST hash to this value at
# every later SCHEMA_VERSION. If the legacy-schema dispatch in
# `_hashable_payload` ever changes the bytes that flow into the hash for
# this fixture, this constant must NOT be updated — instead, the dispatch
# logic itself is wrong and must be reverted.
EXPECTED_0_2_0_RUN_ID = (
    "0cabc6678888e4ef82d7a94adff0a42900fa9e81c3c6493ecfc354e941034f66"
)


def _build_fixture_manifest(
    schema_version: str,
    *,
    success_criterion: dict[str, Any] | None = None,
) -> Manifest:
    """Deterministic Manifest with fixed inputs. Caller picks the schema_version
    and optional success_criterion; everything else is identical so the
    schema dispatch + new field are the only sources of hash drift."""
    return Manifest(
        schema_version=schema_version,
        code_sha="deadbeef" * 5,
        container_digest="sha256:cafef00d" + "0" * 56,
        model=ModelRef(
            name="fixture-model",
            checkpoint_sha256="a" * 64,
            huggingface_revision=None,
            family="mock",
        ),
        simulator=SimulatorRef(
            name="simpler-env",
            commit="abc1234",
            auxiliary_commits={"maniskill2_real2sim": "def5678"},
        ),
        task_name="fixture_task",
        embodiment="fixture_embodiment",
        trials_per_cell=3,
        cells=[
            CellResultPayload(
                cell_id="e/t/k=v",
                axes={"k": "v"},
                n_trials=3,
                successes=2,
                wilson_ci_low=0.2076,
                wilson_ci_high=0.9385,
            ),
        ],
        hardware=HardwareRef(gpu="RTX 3090", cuda="12.4", driver="580.126.20"),
        seeds=[0, 1, 2],
        calibration=CalibrationRef(tier="C"),
        notes="fixture",
        canonical_axis_map={"k": "visuals"},
        success_criterion=success_criterion,
    )


def test_current_schema_version_is_0_3_0() -> None:
    """Guard against accidental downgrade or future bump landing without
    updating this file's other tests."""
    assert SCHEMA_VERSION == "0.3.0"


def test_schema_0_2_0_fixture_still_verifies() -> None:
    """The load-bearing byte-identical regression.

    A Manifest constructed at schema_version="0.2.0" with the fixture inputs
    above MUST hash to EXPECTED_0_2_0_RUN_ID. If this fails, the
    `_hashable_payload` dispatch has drifted and any 0.2.0 manifest in the
    wild will fail `verify()` after upgrade.
    """
    m = _build_fixture_manifest("0.2.0")
    m.seal()
    assert m.run_id == EXPECTED_0_2_0_RUN_ID, (
        f"0.2.0 byte-identity REGRESSED. Got {m.run_id}, expected "
        f"{EXPECTED_0_2_0_RUN_ID}. The schema 0.2→0.3 transition has "
        "altered the canonical-JSON bytes for an existing 0.2.0 manifest. "
        "This is a substrate contract break — revert the offending change."
    )
    assert m.verify() is True


def test_schema_0_2_0_fixture_via_json_round_trip() -> None:
    """Same property via the on-disk JSON path: load a 0.2.0 manifest JSON
    that omits `success_criterion`, verify it reproduces the expected
    run_id and passes verify()."""
    m = _build_fixture_manifest("0.2.0")
    m.seal()
    json_payload = m.to_json()
    loaded = Manifest.from_json(json_payload)
    assert loaded.run_id == EXPECTED_0_2_0_RUN_ID
    assert loaded.verify() is True
    # A 0.2.0 manifest that was written before 0.3.0 existed wouldn't carry
    # the key at all; drop it from the JSON and re-load to prove the
    # default-None round-trip works for unknown-key JSON too.
    parsed = json.loads(json_payload)
    parsed.pop("success_criterion", None)
    j_without_key = json.dumps(parsed)
    loaded2 = Manifest.from_json(j_without_key)
    assert loaded2.run_id == EXPECTED_0_2_0_RUN_ID
    assert loaded2.verify() is True


def test_0_3_0_manifest_without_success_criterion_omits_key_from_hash() -> None:
    """A 0.3.0 manifest with success_criterion=None must omit the key from
    the canonical-JSON bytes that feed the hash. If it serialized as
    `null` instead of being omitted, a future 0.3.0 client that learns to
    write the key explicitly would produce a different run_id for the
    same logical manifest. Tested by checking the canonical-JSON
    of `_hashable_payload()` directly."""
    m = _build_fixture_manifest("0.3.0", success_criterion=None)
    payload = m._hashable_payload()
    assert "success_criterion" not in payload, (
        "success_criterion=None leaked into _hashable_payload — the canonical "
        "hash will carry `success_criterion: null` and break the property "
        "that present-and-null equals omitted."
    )
    rendered = canonical_json(payload)
    assert '"success_criterion"' not in rendered


def test_0_3_0_with_success_criterion_changes_run_id() -> None:
    """The whole point of the substrate change: two sweeps of the same
    scene with different predicates produce distinct run_ids. Without
    this property, a factory engineer who runs the same scene under
    different goals would collide on run_id."""
    m_none = _build_fixture_manifest("0.3.0", success_criterion=None)
    m_a = _build_fixture_manifest(
        "0.3.0",
        success_criterion={
            "kind": "robot_reached_region",
            "params": {"region_name": "behind_truck", "tolerance": 0.5},
        },
    )
    m_b = _build_fixture_manifest(
        "0.3.0",
        success_criterion={
            "kind": "maintained_clearance",
            "params": {"region_name": "truck", "min_distance": 0.3},
        },
    )
    m_none.seal()
    m_a.seal()
    m_b.seal()
    assert m_none.run_id != m_a.run_id
    assert m_a.run_id != m_b.run_id
    assert m_none.run_id != m_b.run_id
    # All three must verify against their own sealed hash.
    assert m_none.verify() and m_a.verify() and m_b.verify()


def test_0_3_0_None_hashes_identically_to_key_omitted_from_json() -> None:
    """The additive-evolution property in its most direct form: a 0.3.0
    manifest with success_criterion=None and a 0.3.0 JSON that doesn't
    carry the key at all must produce the same run_id. This is what
    makes the bump truly additive — a JSON payload written by an old
    client (that didn't know about 0.3.0's new field) and the same
    payload re-emitted by a new client (which writes `null`) verify
    against the same hash."""
    m = _build_fixture_manifest("0.3.0", success_criterion=None)
    m.seal()

    # JSON path 1: what a NEW 0.3.0 client writes — includes the key as null.
    j_with_null = m.to_json()
    parsed_with = json.loads(j_with_null)
    assert "success_criterion" in parsed_with
    assert parsed_with["success_criterion"] is None

    # JSON path 2: what an OLD client wrote — the key is absent entirely.
    parsed_without = dict(parsed_with)
    parsed_without.pop("success_criterion")
    j_without_key = json.dumps(parsed_without)

    loaded_with = Manifest.from_json(j_with_null)
    loaded_without = Manifest.from_json(j_without_key)
    assert loaded_with.run_id == loaded_without.run_id == m.run_id
    assert loaded_with.verify() and loaded_without.verify()


def test_0_3_0_with_success_criterion_round_trips_through_json() -> None:
    """Constructing a 0.3.0 manifest with a real predicate, serializing,
    re-loading, and re-sealing must produce the same run_id. Catches
    accidental serialization drift (e.g., nested dict ordering)."""
    crit = {
        "kind": "robot_reached_region",
        "params": {
            "region_name": "behind_truck",
            "tolerance": 0.5,
            "fall_height": 0.15,
        },
    }
    m = _build_fixture_manifest("0.3.0", success_criterion=crit)
    m.seal()
    j = m.to_json()
    loaded = Manifest.from_json(j)
    assert loaded.run_id == m.run_id
    assert loaded.verify() is True
    # Param key reordering in the JSON source must NOT change the run_id
    # because canonical_json sorts keys.
    parsed = json.loads(j)
    parsed["success_criterion"]["params"] = dict(
        reversed(list(parsed["success_criterion"]["params"].items()))
    )
    reordered_json = json.dumps(parsed)
    loaded_reordered = Manifest.from_json(reordered_json)
    assert loaded_reordered.run_id == m.run_id


def test_hashable_payload_excludes_run_id_and_submitter_fields_regardless_of_schema() -> None:
    """Regression guard: across every schema, run_id + submitter_* are
    excluded from the hash. (These fields are populated AFTER the hash is
    computed, so including them would make `verify()` impossible.)"""
    for schema in ("0.1.0", "0.2.0", "0.3.0"):
        m = _build_fixture_manifest(schema)
        m.run_id = "should-be-excluded"
        m.submitter_signature = "should-be-excluded"
        m.submitter_public_key = "should-be-excluded"
        m.submitter_identity = "should-be-excluded"
        payload = m._hashable_payload()
        for k in (
            "run_id",
            "submitter_signature",
            "submitter_public_key",
            "submitter_identity",
        ):
            assert k not in payload, f"{schema}: {k} leaked into _hashable_payload"


def test_schema_0_1_0_excludes_canonical_axis_map() -> None:
    """0.1.0 predates canonical_axis_map. The legacy dispatch must still
    pop the field for that schema regardless of whether the dataclass
    field carries a value (an in-memory 0.1.0 manifest constructed with
    a non-empty map would otherwise hash differently from one constructed
    with the empty default — that would silently break 0.1.0 byte-identity)."""
    m = _build_fixture_manifest("0.1.0")
    payload = m._hashable_payload()
    assert "canonical_axis_map" not in payload
    assert "success_criterion" not in payload


def test_canonical_json_is_byte_stable_across_two_constructions() -> None:
    """The canonical JSON of two independently-constructed identical
    manifests must be byte-identical. (If `asdict` ever started emitting
    non-deterministic ordering, this would catch it.)"""
    m1 = _build_fixture_manifest("0.3.0", success_criterion={"kind": "x", "params": {"a": 1, "b": 2}})
    m2 = _build_fixture_manifest("0.3.0", success_criterion={"kind": "x", "params": {"b": 2, "a": 1}})
    j1 = canonical_json(m1._hashable_payload())
    j2 = canonical_json(m2._hashable_payload())
    assert j1 == j2
    assert hash_dict(m1._hashable_payload()) == hash_dict(m2._hashable_payload())
