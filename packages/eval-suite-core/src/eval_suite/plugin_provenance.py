"""Plugin provenance sidecar.

**In plain words.** A sweep usually involves several pip-installed
plugin packages: the task is one package, the policy is another, the
simulator bridge is a third. This file is the paper trail that says
"this run used version X of package A and version Y of package B."
If a result later turns out to be wrong, the sidecar lets a reviewer
nail down exactly which plugin version produced it. Lives next to the
manifest, evolves separately from it so the manifest's reproducibility
hash stays stable across plugin-only changes.


Sits **alongside** `manifest.json` in the sweep output directory and
records which pip-installed plugin packages produced each component of
the run. Deliberately *not* part of the manifest: the manifest schema
stays at 0.2.0 and its `run_id` hash chain stays intact. Provenance
information evolves independently as a sidecar file.

What it captures:

  - The eval-suite `CONTRACT_VERSION` at sweep time (so a future
    breaking-change reviewer can spot 1.x vs 2.x runs).
  - Per-component: the pip distribution name + version + the
    entry-point short name. For the Task / Policy / Adapter that ran.
  - The target manifest's `run_id` — so the sidecar is bound to one
    specific manifest. If a reviewer downloads a tarball with both
    `manifest.json` and `plugin_provenance.json`, the sidecar's
    `target_run_id` must equal the manifest's `run_id`.
  - The eval-suite package version itself (from `importlib.metadata`).
  - Optional Ed25519 signature over the canonical-JSON of the above,
    using the same signing primitive as `manifest.sign()`. A signed
    sidecar proves "this set of plugin versions produced this run_id"
    in addition to the manifest's existing "these inputs produced
    this run_id" claim.

What it intentionally does NOT do:

  - It does not feed the manifest's content hash. A plugin author
    bumping a version number must not invalidate the manifest's
    `run_id`. The plugin's CODE changes feed the hash via `code_sha`;
    the plugin's metadata is sidecar.
  - It does not promise tamper-evidence at the portal level. The
    portal's append-only `ledger.jsonl` is the (poor-man's) audit trail
    for that; Sigstore migration is v1.0.
"""

from __future__ import annotations

import importlib.metadata
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .hashing import canonical_json
from .signing import sign as _sign
from .signing import verify as _verify


@dataclass(frozen=True)
class PluginRef:
    """One installed plugin that contributed to the run.

    `package_name` is the pip distribution name (e.g. `"eval-suite"`,
    `"my-house-tasks"`). `package_version` is whatever
    `importlib.metadata.version(package_name)` returned at sweep time.
    `entry_point_name` is the short name the plugin registered under
    (the left side of `=` in the pyproject.toml entry-point block).
    `contract_version_target` is the eval-suite `CONTRACT_VERSION` the
    plugin claimed to be written against — usually the latest at
    publish time. Drift between this and the runtime `CONTRACT_VERSION`
    is a signal for the reviewer.
    """

    package_name: str
    package_version: str
    entry_point_name: str
    contract_version_target: str


@dataclass
class PluginProvenance:
    """Sidecar JSON living next to `manifest.json` in the sweep output dir."""

    contract_version: str          # eval-suite CONTRACT_VERSION at sweep time
    target_run_id: str             # the manifest.run_id this sidecar attests to
    components: dict[str, PluginRef]  # "task" / "policy" / "adapter" → PluginRef
    eval_suite_version: str        # importlib.metadata.version("eval-suite-core")
    # Optional Ed25519 signature over the canonical-JSON of the above
    # four fields. Same Ed25519 primitive as manifest.sign().
    submitter_signature: str | None = field(default=None)
    submitter_public_key: str | None = field(default=None)
    submitter_identity: str | None = field(default=None)

    def _hashable_payload(self) -> dict[str, object]:
        """The dict that gets signed. Excludes the signature fields themselves."""
        d = asdict(self)
        for k in ("submitter_signature", "submitter_public_key", "submitter_identity"):
            d.pop(k, None)
        return d

    def sign(self, private_key_bytes: bytes, public_key_hex: str, identity: str) -> PluginProvenance:
        canonical = canonical_json(self._hashable_payload())
        self.submitter_signature = _sign(canonical, private_key_bytes)
        self.submitter_public_key = public_key_hex
        self.submitter_identity = identity
        return self

    def verify(self) -> bool:
        """True iff (a) any present signature is valid for the public key
        over the same canonical payload that was signed.

        Unsigned sidecars always verify True — the sidecar is informational
        even without a signature. Plugin authors who want non-repudiation
        sign it; plugin authors who don't, don't.
        """
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
    def load(cls, path: Path) -> PluginProvenance:
        obj = json.loads(Path(path).read_text())
        return cls(
            contract_version=obj["contract_version"],
            target_run_id=obj["target_run_id"],
            components={k: PluginRef(**v) for k, v in obj["components"].items()},
            eval_suite_version=obj["eval_suite_version"],
            submitter_signature=obj.get("submitter_signature"),
            submitter_public_key=obj.get("submitter_public_key"),
            submitter_identity=obj.get("submitter_identity"),
        )


def _package_version(package_name: str, default: str = "unknown") -> str:
    """Look up an installed distribution's version, or return a default."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return default


def build_for_run(
    *,
    task: object,
    policy: object,
    adapter: object,
    target_run_id: str,
    contract_version: str,
) -> PluginProvenance:
    """Resolve the (Task / Policy / Adapter) instances back to the pip
    packages they came from and build a PluginProvenance.

    Resolution strategy: walk the entry-points catalog, find the one
    whose target class matches the instance's `type(...)`, and read its
    `dist` metadata. If no entry-point matches (e.g. the user constructed
    the class manually and didn't register it), record `package_name`
    as the class's `__module__.partition('.')[0]` and version as
    "unregistered".
    """
    from .registry import list_adapters, list_policies, list_tasks

    def _resolve(instance: object, entries: list[object]) -> PluginRef:
        cls = type(instance)
        for e in entries:
            ref = getattr(e, "entry_point_ref", None)
            if ref is None:
                continue
            mod, _, attr = ref.partition(":")
            if cls.__module__ == mod and cls.__name__ == attr:
                return PluginRef(
                    package_name=getattr(e, "package_name", "unknown"),
                    package_version=getattr(e, "package_version", "unknown"),
                    entry_point_name=getattr(e, "name", "unknown"),
                    contract_version_target=contract_version,
                )
        # Fallback: instance wasn't registered as an entry-point.
        return PluginRef(
            package_name=cls.__module__.partition(".")[0],
            package_version=_package_version(cls.__module__.partition(".")[0]),
            entry_point_name=cls.__name__,
            contract_version_target=contract_version,
        )

    return PluginProvenance(
        contract_version=contract_version,
        target_run_id=target_run_id,
        components={
            "task": _resolve(task, list_tasks()),  # type: ignore[arg-type]
            "policy": _resolve(policy, list_policies()),  # type: ignore[arg-type]
            "adapter": _resolve(adapter, list_adapters()),  # type: ignore[arg-type]
        },
        eval_suite_version=_package_version("eval-suite-core"),
    )
