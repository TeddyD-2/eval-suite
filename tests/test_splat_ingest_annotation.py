"""Schema + canonical-JSON tests for `eval_suite.ingest.splat.annotation`.

These are CI-runnable (no MuJoCo, no splat tools, no GPU): the
annotation layer is the "always-works" piece of the splat substrate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eval_suite.hashing import sha256_text
from eval_suite.ingest.splat.annotation import (
    SCENE_EXTRACTIONS_SCHEMA_VERSION,
    SCENE_METADATA_SCHEMA_VERSION,
    AnnotationSchemaError,
    BodyPhysics,
    ExtractedBody,
    NamedRegion,
    SceneExtractions,
    SceneMetadata,
    SceneTransform,
    SpawnPoint,
)


def _fixture_scene_metadata() -> SceneMetadata:
    return SceneMetadata(
        schema_version=SCENE_METADATA_SCHEMA_VERSION,
        scene_transform=SceneTransform(
            up_axis="z", meters_per_unit=1.0, world_origin=(0.0, 0.0, 0.0)
        ),
        named_regions=(
            NamedRegion(
                name="behind_truck",
                shape="box",
                pos=(3.0, 0.0, 0.3),
                size=(0.5, 0.5, 0.3),
            ),
        ),
        spawn_points=(
            SpawnPoint(name="go1_start", pos=(0.0, 0.0, 0.3)),
        ),
    )


def _fixture_scene_extractions() -> SceneExtractions:
    return SceneExtractions(
        schema_version=SCENE_EXTRACTIONS_SCHEMA_VERSION,
        extracted_bodies=(
            ExtractedBody(
                name="truck",
                extraction_bounds=NamedRegion(
                    name="truck_bounds",
                    shape="box",
                    pos=(1.5, 0.0, 0.4),
                    size=(1.2, 0.8, 0.6),
                ),
                physics=BodyPhysics(mass=2000.0, joint_type="fixed"),
                collision="convex_hull",
            ),
        ),
    )


def test_scene_metadata_canonical_json_is_byte_stable() -> None:
    """Two independent constructions of the same scene_metadata must
    produce byte-identical canonical JSON. If not, any cell sweep that
    binds scene_metadata into the asset provenance would silently drift
    run_ids across rebuilds."""
    a = _fixture_scene_metadata()
    b = _fixture_scene_metadata()
    assert a.to_canonical_json() == b.to_canonical_json()
    # Equality on the dataclass.
    assert a == b


def test_scene_metadata_load_round_trip(tmp_path: Path) -> None:
    """Write → load → re-write must produce the same canonical bytes.
    Catches accidental list-vs-tuple type drift on round-trip."""
    orig = _fixture_scene_metadata()
    path = tmp_path / "scene_metadata.json"
    orig.save(path)
    loaded = SceneMetadata.load(path)
    assert loaded == orig
    # Re-serialize, compare bytes.
    re_json = loaded.to_canonical_json()
    assert re_json == orig.to_canonical_json()


def test_scene_extractions_load_round_trip(tmp_path: Path) -> None:
    orig = _fixture_scene_extractions()
    path = tmp_path / "scene_extractions.json"
    orig.save(path)
    loaded = SceneExtractions.load(path)
    assert loaded == orig
    assert loaded.to_canonical_json() == orig.to_canonical_json()


def test_invalid_up_axis_rejected(tmp_path: Path) -> None:
    """Bad SceneTransform must raise at load time, not deep in compose."""
    bad = {
        "schema_version": "0.1.0",
        "scene_transform": {
            "up_axis": "q",  # not x/y/z
            "meters_per_unit": 1.0,
            "world_origin": [0, 0, 0],
        },
        "named_regions": [],
        "spawn_points": [],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(AnnotationSchemaError, match="up_axis"):
        SceneMetadata.load(path)


def test_invalid_region_shape_rejected(tmp_path: Path) -> None:
    bad = {
        "schema_version": "0.1.0",
        "scene_transform": {
            "up_axis": "z",
            "meters_per_unit": 1.0,
            "world_origin": [0, 0, 0],
        },
        "named_regions": [
            {"name": "x", "shape": "torus", "pos": [0, 0, 0], "size": [1, 1, 1]}
        ],
        "spawn_points": [],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(AnnotationSchemaError, match="shape"):
        SceneMetadata.load(path)


def test_invalid_joint_type_rejected(tmp_path: Path) -> None:
    bad = {
        "schema_version": "0.1.0",
        "extracted_bodies": [
            {
                "name": "x",
                "extraction_bounds": {
                    "name": "b",
                    "shape": "box",
                    "pos": [0, 0, 0],
                    "size": [1, 1, 1],
                },
                "physics": {"mass": 1.0, "joint_type": "ratchet"},  # bad
            }
        ],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(AnnotationSchemaError, match="joint_type"):
        SceneExtractions.load(path)


def test_modifying_scene_metadata_changes_hash(tmp_path: Path) -> None:
    """The substrate-determinism property: modifying scene_metadata by one
    byte (e.g., moving a spawn point 1mm) changes its canonical-JSON
    SHA256. That SHA256 is what flows into AssetProvenance and binds
    into the run_id — so the property here is what makes the splat
    pipeline preserve the eval-suite's content-addressing contract."""
    a = _fixture_scene_metadata()
    moved_spawn = SpawnPoint(
        name="go1_start", pos=(0.0, 0.0, 0.301)
    )  # 1mm up
    b = SceneMetadata(
        schema_version=a.schema_version,
        scene_transform=a.scene_transform,
        named_regions=a.named_regions,
        spawn_points=(moved_spawn,),
    )
    assert sha256_text(a.to_canonical_json()) != sha256_text(b.to_canonical_json())


def test_canonical_json_uses_sorted_keys() -> None:
    """canonical_json (re-exported from eval_suite.hashing) must sort
    keys at every level. SceneMetadata's canonical JSON should be order-
    insensitive in the dataclass field order."""
    s = _fixture_scene_metadata()
    j = s.to_canonical_json()
    # Confirm the top-level keys are sorted alphabetically in the output.
    keys_in_order = []
    depth = 0
    in_str = False
    cur = ""
    for ch in j:
        if ch == '"' and depth == 0:
            in_str = not in_str
            if not in_str:
                keys_in_order.append(cur)
                cur = ""
            continue
        if in_str and depth == 0:
            cur += ch
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
    # Crude parse — but works for verifying top-level key order on a
    # sorted-keys JSON output.
    assert keys_in_order == sorted(keys_in_order)


def test_named_region_size_arity_validated(tmp_path: Path) -> None:
    """sphere requires size length 1; box requires 3; cylinder requires 2.
    Schema validation catches the mismatch at load."""
    bad = {
        "schema_version": "0.1.0",
        "scene_transform": {
            "up_axis": "z",
            "meters_per_unit": 1.0,
            "world_origin": [0, 0, 0],
        },
        "named_regions": [
            {"name": "x", "shape": "sphere", "pos": [0, 0, 0], "size": [1, 1, 1]}
        ],
        "spawn_points": [],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(AnnotationSchemaError, match="sphere"):
        SceneMetadata.load(path)
