"""Manifest dataclass — every input that could change a result.

The manifest is the reproducibility deliverable: shipped alongside the
results CSV, anyone with the manifest can verify that a rerun on the
same inputs would produce a byte-identical hash.

`run_id` is computed by canonicalizing the manifest (excluding the
`run_id` field itself) and SHA256-ing the bytes. Changing any input —
different ckpt SHA, different sim commit, different seed list — yields
a new run_id.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .hashing import canonical_json, hash_dict
from .signing import sign as _sign
from .signing import verify as _verify

SCHEMA_VERSION = "0.3.0"

# Schema 0.2.0 added `canonical_axis_map` to the hashed payload so a
# published profile is provably linked to the mapping that produced it.
# Schema 0.3.0 adds `success_criterion` so a parametric scene re-evaluated
# under a different declarative goal produces a distinct run_id (the
# factory-engineer wedge: change the predicate, get a new run_id, no
# new Task subclass needed). `_hashable_payload` dispatches per-feature
# on the version-set membership so old manifests continue to verify
# byte-identically under the rules that produced their run_id.

# Schemas that predate `canonical_axis_map` in the hashed payload.
# DO NOT add 0.2.0 here — that would change 0.2.0 hashing semantics.
_LEGACY_SCHEMAS = frozenset({"0.1.0"})

# Schemas that predate `success_criterion` in the hashed payload.
# Both 0.1.0 and 0.2.0 are listed: the field is omitted from the hash
# (and from `_hashable_payload`) for these schemas regardless of whether
# the dataclass field is set, so loading a pre-0.3.0 manifest still
# produces its original run_id.
_SCHEMAS_WITHOUT_SUCCESS_CRITERION = frozenset({"0.1.0", "0.2.0"})


@dataclass(frozen=True)
class ModelRef:
    name: str  # e.g. "rt1-converged", "octo-base"
    checkpoint_sha256: str  # sha256 of the ckpt directory
    huggingface_revision: str | None = None  # for HF-loaded models like Octo
    family: str = ""  # "rt1", "octo", "mock" — used for grouping in the notebook


@dataclass(frozen=True)
class SimulatorRef:
    name: str  # "simpler-env"
    commit: str  # git SHA
    auxiliary_commits: dict[str, str] = field(default_factory=dict)
    # ^ e.g. {"maniskill2_real2sim": "ef7a4d4..."}


@dataclass(frozen=True)
class HardwareRef:
    gpu: str  # "RTX 3090"
    cuda: str  # "12.4"
    driver: str  # "580.126.20"


@dataclass(frozen=True)
class CellResultPayload:
    """JSON-serializable subset of CellResult. Lives in the manifest."""

    cell_id: str  # the slug from CellId
    axes: dict[str, str]  # axis-level mapping
    n_trials: int
    successes: int
    wilson_ci_low: float
    wilson_ci_high: float


@dataclass(frozen=True)
class CalibrationRef:
    tier: str  # "A" / "B" / "C"
    real_perf_source: str = ""  # e.g. "SimplerEnv paper Table 3"
    real_perf_value: float | None = None


@dataclass
class Manifest:
    """The reproducibility record for a single sweep run.

    The `run_id` is set at construction time to a placeholder, then
    populated by `seal()` which computes the content-addressed hash. A
    sealed manifest is immutable in spirit; mutating any field after
    seal() invalidates `verify()`.
    """

    schema_version: str
    code_sha: str
    container_digest: str  # e.g. "sha256:<docker image digest>" or "" if not in a container
    model: ModelRef
    simulator: SimulatorRef
    task_name: str
    embodiment: str
    trials_per_cell: int
    cells: list[CellResultPayload]
    hardware: HardwareRef
    seeds: list[int]
    calibration: CalibrationRef
    run_id: str = ""  # populated by seal()
    notes: str = ""
    # Optional submitter attestation. When present, signs the same
    # canonical-JSON payload that produced run_id, using the keypair the
    # submitter registered with the portal. `verify()` checks both the
    # content hash AND the signature when present.
    submitter_signature: str | None = None
    submitter_public_key: str | None = None
    submitter_identity: str | None = None  # e.g. email / github handle / org
    # Task's canonical-axis taxonomy at sweep time. Bound into the
    # content hash so the published profile is provably linked to the
    # mapping that produced it. Only included in the hash for schema
    # 0.2.0+; 0.1.0 manifests exclude it (back-compat — see
    # `_hashable_payload`).
    canonical_axis_map: dict[str, str] = field(default_factory=dict)
    # Task's declarative success criterion at sweep time, as a registry
    # entry like {"kind": "robot_reached_region",
    # "params": {"region_name": "behind_truck", "tolerance": 0.5}}.
    # Bound into the content hash so two sweeps of the same scene with
    # different goals (e.g. reach_region vs maintain_clearance_from)
    # produce distinct run_ids. None for Tasks whose env returns success
    # from `env.step()` directly (current Namaqualand / SimplerEnv
    # semantics) — when None, the field is OMITTED from the canonical
    # JSON so a 0.3.0 manifest without a declared criterion hashes
    # byte-identically to a 0.3.0 JSON that omits the key entirely.
    # Only included in the hash for schema 0.3.0+ (see
    # `_SCHEMAS_WITHOUT_SUCCESS_CRITERION`).
    success_criterion: dict[str, Any] | None = None

    def _hashable_payload(self) -> dict[str, Any]:
        """The dict that gets hashed for run_id.

        Always excludes `run_id` and the submitter_* fields (signature is
        computed AFTER seal() so it can sign over the canonical contents).

        Per-feature legacy dispatch keeps each pre-existing schema verifying
        byte-identically under its original rules:

          - Schema 0.1.0: excludes `canonical_axis_map` AND
            `success_criterion`.
          - Schema 0.2.0: includes `canonical_axis_map`, excludes
            `success_criterion`.
          - Schema 0.3.0+: includes both. `success_criterion=None` is
            omitted from the canonical JSON (not serialized as `null`),
            so a 0.3.0 manifest with the default value hashes
            byte-identically to a 0.3.0 JSON that doesn't carry the key.

        `asdict()` emits `None` as JSON `null`, NOT as an omitted key —
        the explicit `.pop("success_criterion", None)` is what produces
        omitted-key semantics.
        """
        payload = asdict(self)
        for k in ("run_id", "submitter_signature", "submitter_public_key", "submitter_identity"):
            payload.pop(k, None)
        if self.schema_version in _LEGACY_SCHEMAS:
            payload.pop("canonical_axis_map", None)
        if self.schema_version in _SCHEMAS_WITHOUT_SUCCESS_CRITERION:
            payload.pop("success_criterion", None)
        elif self.success_criterion is None:
            payload.pop("success_criterion", None)
        return payload

    def seal(self) -> Manifest:
        """Compute and stamp the content-addressed run_id. Idempotent."""
        digest = hash_dict(self._hashable_payload())
        # Replace the field in-place; manifest is mutable until sealed.
        self.run_id = digest
        return self

    def sign(self, private_key_bytes: bytes, public_key_hex: str, identity: str) -> Manifest:
        """Sign the sealed manifest. Must be called AFTER seal().

        The signature is over the canonical JSON of the hashable payload
        (same bytes that produced run_id) — so a verifier with only the
        manifest + public key can reconstruct what was signed without
        guessing about whitespace or key order.
        """
        if not self.run_id:
            raise ValueError("seal() must be called before sign()")
        canonical_payload = canonical_json(self._hashable_payload())
        self.submitter_signature = _sign(canonical_payload, private_key_bytes)
        self.submitter_public_key = public_key_hex
        self.submitter_identity = identity
        return self

    def verify(self) -> bool:
        """Returns True iff:
        - run_id matches the recomputed content hash, AND
        - if submitter_signature is present, the signature is valid for
          the submitter_public_key over the same canonical payload.

        A manifest WITHOUT a signature still verifies on content alone
        (tier-1 maintainer-run sweeps); a manifest WITH a signature must
        pass both checks.
        """
        if not self.run_id:
            return False
        if hash_dict(self._hashable_payload()) != self.run_id:
            return False
        if self.submitter_signature is None:
            return True
        if self.submitter_public_key is None:
            return False  # signature without key — malformed
        canonical_payload = canonical_json(self._hashable_payload())
        return _verify(canonical_payload, self.submitter_signature, bytes.fromhex(self.submitter_public_key))

    def to_json(self) -> str:
        # Pretty-print on disk; canonical form is used only for hashing/signing.
        return json.dumps(asdict(self), indent=2, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str) -> Manifest:
        obj = json.loads(payload)
        return cls(
            schema_version=obj["schema_version"],
            code_sha=obj["code_sha"],
            container_digest=obj["container_digest"],
            model=ModelRef(**obj["model"]),
            simulator=SimulatorRef(
                name=obj["simulator"]["name"],
                commit=obj["simulator"]["commit"],
                auxiliary_commits=obj["simulator"].get("auxiliary_commits", {}),
            ),
            task_name=obj["task_name"],
            embodiment=obj["embodiment"],
            trials_per_cell=obj["trials_per_cell"],
            cells=[CellResultPayload(**c) for c in obj["cells"]],
            hardware=HardwareRef(**obj["hardware"]),
            seeds=list(obj["seeds"]),
            calibration=CalibrationRef(**obj["calibration"]),
            run_id=obj.get("run_id", ""),
            notes=obj.get("notes", ""),
            submitter_signature=obj.get("submitter_signature"),
            submitter_public_key=obj.get("submitter_public_key"),
            submitter_identity=obj.get("submitter_identity"),
            canonical_axis_map=dict(obj.get("canonical_axis_map", {})),
            success_criterion=obj.get("success_criterion"),
        )
