"""Asset provenance sidecar.

Sits **alongside** `manifest.json` in the sweep output directory and
records the source assets (e.g. USD scans, converted MJCF, mesh files,
intermediate artifacts) that fed the run. Deliberately *not* part of
the manifest: the manifest schema stays at 0.2.0 and its `run_id` hash
chain stays intact. Mirrors the `plugin_provenance.json` pattern.

What it captures:

  - Per asset: role tag, path (relative to repo root), SHA256, origin URL,
    license, exact conversion command that produced it, free-form notes.
  - The target manifest's `run_id` — so the sidecar is bound to one
    specific manifest. If a reviewer downloads a tarball with
    `manifest.json` + `asset_provenance.json` + the source/converted
    files, `target_run_id` must equal `manifest.run_id`.
  - Optional Ed25519 signature over the canonical-JSON of the above
    fields, using the same signing primitive as `manifest.sign()` and
    `plugin_provenance.sign()`.

What `verify()` adds beyond signature checking:

  - Re-hashes each declared asset path on disk and asserts byte-equality
    against the recorded SHA256. This is the load-bearing claim of the
    sidecar — a reviewer downloading a tarball can prove the run used
    *these specific bytes*. If an asset is missing or tampered with,
    `verify()` returns False without raising.

What it intentionally does NOT do:

  - It does not feed the manifest's content hash. Adding or removing
    asset entries must not invalidate the manifest's `run_id`. Source
    code changes feed the hash via `code_sha`; the assets themselves
    live in the sidecar and evolve independently.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .hashing import canonical_json, sha256_file
from .signing import sign as _sign
from .signing import verify as _verify


@dataclass(frozen=True)
class AssetRef:
    """One asset that contributed to the run.

    `role` is a short tag describing the asset's place in the pipeline
    ("source_usd", "visual_mesh", "collision_hull", "composed_mjcf", ...).
    Roles are not enumerated — Task authors describe their own pipelines.

    `path` is relative to the sweep's `repo_root` (passed to `verify`)
    so the sidecar can be moved between checkouts without breaking the
    on-disk hash check.

    `conversion_command` is the exact shell/Python invocation that
    produced this asset from its predecessor. Empty string for source
    assets that weren't generated locally (downloaded files).
    """

    role: str
    path: str
    sha256: str
    origin_url: str
    license: str
    conversion_command: str = ""
    notes: str = ""


@dataclass
class AssetProvenance:
    """Sidecar JSON living next to `manifest.json` in the sweep output dir."""

    target_run_id: str
    assets: list[AssetRef]
    # Optional Ed25519 signature over the canonical-JSON of the above
    # two fields. Same primitive as `manifest.sign` and
    # `plugin_provenance.sign`.
    submitter_signature: str | None = field(default=None)
    submitter_public_key: str | None = field(default=None)
    submitter_identity: str | None = field(default=None)

    def _hashable_payload(self) -> dict[str, object]:
        """The dict that gets signed. Excludes the signature fields themselves."""
        d = asdict(self)
        for k in ("submitter_signature", "submitter_public_key", "submitter_identity"):
            d.pop(k, None)
        return d

    def sign(self, private_key_bytes: bytes, public_key_hex: str, identity: str) -> AssetProvenance:
        canonical = canonical_json(self._hashable_payload())
        self.submitter_signature = _sign(canonical, private_key_bytes)
        self.submitter_public_key = public_key_hex
        self.submitter_identity = identity
        return self

    def verify(self, repo_root: Path | None = None) -> bool:
        """True iff every recorded asset's on-disk SHA256 matches the
        recorded value AND any present signature is valid for the public
        key over the canonical payload.

        `repo_root` is the base directory that asset paths are relative
        to. Defaults to the current working directory. Returns False
        (never raises) if an asset is missing, tampered with, or the
        signature is malformed.

        Unsigned sidecars pass the signature check trivially — the
        sidecar is informational even without a signature; the asset
        re-hashing is the load-bearing claim.
        """
        root = (repo_root or Path.cwd()).resolve()
        for ref in self.assets:
            asset_path = root / ref.path
            if not asset_path.is_file():
                return False
            try:
                if sha256_file(asset_path) != ref.sha256:
                    return False
            except OSError:
                return False
        if self.submitter_signature is None:
            return True
        if self.submitter_public_key is None:
            return False
        canonical = canonical_json(self._hashable_payload())
        return _verify(canonical, self.submitter_signature, bytes.fromhex(self.submitter_public_key))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> AssetProvenance:
        obj = json.loads(Path(path).read_text())
        return cls(
            target_run_id=obj["target_run_id"],
            assets=[AssetRef(**a) for a in obj["assets"]],
            submitter_signature=obj.get("submitter_signature"),
            submitter_public_key=obj.get("submitter_public_key"),
            submitter_identity=obj.get("submitter_identity"),
        )


def write_for_run(
    *,
    target_run_id: str,
    assets: list[AssetRef],
    output_dir: Path,
    filename: str = "asset_provenance.json",
) -> AssetProvenance:
    """Build + save the asset sidecar bound to `target_run_id`. Returns
    the object for further actions (e.g. signing). The caller is
    responsible for computing the per-asset SHA256 values (typically via
    `eval_suite.hashing.sha256_file`).
    """
    sidecar = AssetProvenance(target_run_id=target_run_id, assets=list(assets))
    sidecar.save(Path(output_dir) / filename)
    return sidecar
