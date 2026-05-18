"""Asset-provenance tests for the TNT Truck splat demo's sidecar.

Mirrors the Namaqualand provenance tests at
tests/test_usd_scan_contract.py:106-176 but binds against the splat
sidecar shape (scene_metadata + scene_extractions + composed MJCF +
static + collision-hull OBJs).

The provenance sidecar's `verify()` re-hashes each declared asset on
disk and asserts SHA256 byte-equality. Modifying any registered asset
must cause verify() to return False; signing must round-trip.
"""

from __future__ import annotations

from pathlib import Path

from eval_suite.asset_provenance import AssetProvenance, AssetRef, write_for_run
from eval_suite.hashing import sha256_file
from eval_suite.signing import generate_keypair


def _write_dummy_asset(path: Path, content: bytes = b"asset-content") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return sha256_file(path)


def test_provenance_binds_to_run_id_and_detects_tamper(tmp_path: Path) -> None:
    """Provenance sidecar bound to a known run_id; mutating any registered
    asset SHA256 causes verify() to return False."""
    scene_metadata_path = tmp_path / "scene_metadata.json"
    scene_extractions_path = tmp_path / "scene_extractions.json"
    composed_mjcf_path = tmp_path / "MJCF" / "scene.xml"
    static_mesh_path = tmp_path / "visuals" / "splat_scene.obj"
    hull_mesh_path = tmp_path / "collision_hull" / "splat_scene_hull.obj"

    md_sha = _write_dummy_asset(scene_metadata_path, b'{"schema_version": "0.1.0"}')
    ex_sha = _write_dummy_asset(scene_extractions_path, b'{"schema_version": "0.1.0"}')
    mjcf_sha = _write_dummy_asset(composed_mjcf_path, b"<mujoco/>")
    mesh_sha = _write_dummy_asset(static_mesh_path, b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    hull_sha = _write_dummy_asset(hull_mesh_path, b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

    fake_run_id = "a" * 64
    assets = [
        AssetRef(
            role="scene_metadata",
            path=str(scene_metadata_path.relative_to(tmp_path)),
            sha256=md_sha,
            origin_url="local",
            license="CC-BY-4.0",
        ),
        AssetRef(
            role="scene_extractions",
            path=str(scene_extractions_path.relative_to(tmp_path)),
            sha256=ex_sha,
            origin_url="local",
            license="CC-BY-4.0",
        ),
        AssetRef(
            role="composed_mjcf",
            path=str(composed_mjcf_path.relative_to(tmp_path)),
            sha256=mjcf_sha,
            origin_url="local",
            license="CC-BY-4.0",
        ),
        AssetRef(
            role="splat_scene_mesh",
            path=str(static_mesh_path.relative_to(tmp_path)),
            sha256=mesh_sha,
            origin_url="local",
            license="CC-BY-4.0",
        ),
        AssetRef(
            role="splat_scene_collision_hull",
            path=str(hull_mesh_path.relative_to(tmp_path)),
            sha256=hull_sha,
            origin_url="local",
            license="CC-BY-4.0",
        ),
    ]
    sidecar = write_for_run(
        target_run_id=fake_run_id,
        assets=assets,
        output_dir=tmp_path,
    )
    assert sidecar.verify(repo_root=tmp_path) is True

    # Tamper one byte of the scene_metadata file → verify must fail.
    scene_metadata_path.write_bytes(b'{"schema_version": "0.1.0", "modified": true}')
    loaded = AssetProvenance.load(tmp_path / "asset_provenance.json")
    assert loaded.verify(repo_root=tmp_path) is False


def test_provenance_optional_signing_roundtrip(tmp_path: Path) -> None:
    """Sign / verify / reload roundtrip with Ed25519. Mirrors the existing
    Namaqualand provenance signing test."""
    scene_metadata_path = tmp_path / "scene_metadata.json"
    md_sha = _write_dummy_asset(scene_metadata_path, b'{"schema_version": "0.1.0"}')

    fake_run_id = "b" * 64
    assets = [
        AssetRef(
            role="scene_metadata",
            path=str(scene_metadata_path.relative_to(tmp_path)),
            sha256=md_sha,
            origin_url="local",
            license="CC-BY-4.0",
        )
    ]
    sidecar = write_for_run(
        target_run_id=fake_run_id, assets=assets, output_dir=tmp_path
    )

    priv, pub = generate_keypair()
    sidecar.sign(priv, pub.hex(), identity="test@example.com")
    sidecar.save(tmp_path / "asset_provenance.json")
    loaded = AssetProvenance.load(tmp_path / "asset_provenance.json")
    assert loaded.submitter_signature is not None
    assert loaded.verify(repo_root=tmp_path) is True

    # Bad signature → verify False.
    loaded.submitter_signature = "00" * 64
    assert loaded.verify(repo_root=tmp_path) is False
