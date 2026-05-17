"""File-based submission registry.

The portal accepts signed manifests and writes them to a directory
tree keyed by `(run_id, submitter_pk)` so the same `run_id` can be
submitted by multiple submitters (the Cross-device corroboration
case):

  <root>/<run_id>/<pk_short>/manifest.json   — verbatim manifest
  <root>/<run_id>/<pk_short>/meta.json       — portal metadata

`pk_short` is the first 16 hex chars of the submitter's public key, or
the literal "anonymous-<ts>" when no signature is present.

Plus an index file at `<root>/index.json` — list of every accepted +
rejected submission record, used for fast listing without walking the
tree.

Plus an append-only ledger at `<root>/ledger.jsonl` — one JSON line
per accept/reject event in chronological order. Append-only **by
convention** (nothing else writes); not cryptographically tamper-
evident. v1.0 replaces with a Sigstore transparency log.

This is intentionally a flat filesystem; v1.0 swaps it for a real
database. Keeps the portal trivially `git clone`-deployable.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ..manifest import Manifest


def filter_submissions(
    subs: list[Submission],
    *,
    model: str | None = None,
    task: str | None = None,
    embodiment: str | None = None,
    submitter: str | None = None,
    accepted: bool | None = None,
) -> list[Submission]:
    """In-memory substring/exact filter over a Submission list.

    Text fields (model, task, embodiment, submitter) are matched as
    case-insensitive substring contains — `model="oct"` matches
    `"octo-base"`. None / empty-string for a text field means no filter
    on that field. `accepted=None` means no filter on accepted; True or
    False filter to that value.

    The HTML UI uses this in `/ui/submissions` handler after `list_submissions()`
    because the file-based store has no index. v1.0 (DB-backed) will
    keep the same signature; only the implementation moves.
    """
    def _matches(s: Submission) -> bool:
        if model and model.lower() not in (s.model_name or "").lower():
            return False
        if task and task.lower() not in (s.task_name or "").lower():
            return False
        if embodiment and embodiment.lower() not in (s.embodiment or "").lower():
            return False
        if submitter and submitter.lower() not in (s.submitter_identity or "").lower():
            return False
        if accepted is not None and s.accepted != accepted:
            return False
        return True

    return [s for s in subs if _matches(s)]


@dataclass(frozen=True)
class Submission:
    run_id: str
    embodiment: str
    task_name: str
    submitter_identity: str
    submitter_pk: str            # hex public key, or "anonymous-<ts>" when unsigned
    ingest_ts: float
    accepted: bool
    reject_reason: str | None
    model_name: str = ""         # v0: populated from manifest.model.name on accept()


def _pk_short(pk_hex: str | None) -> str:
    if not pk_hex:
        return f"anonymous-{int(time.time() * 1000)}"
    return pk_hex[:16]


class SubmissionStore:
    """File-backed registry of accepted submissions.

    Multiple submissions per `run_id` are allowed (keyed by submitter_pk)
    so the portal can answer "which submitters have corroborated this
    run_id?" via `list_for_run_id(run_id)`. v0 stores that keyed only
    by run_id are migrated implicitly on next write.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._index_path = self._root / "index.json"
        self._ledger_path = self._root / "ledger.jsonl"

    def accept(self, manifest: Manifest) -> Submission:
        run_id = manifest.run_id
        if not run_id:
            raise ValueError("manifest is not sealed (empty run_id)")
        if not manifest.verify():
            raise ValueError("manifest does not verify (hash or signature mismatch)")
        pk = manifest.submitter_public_key
        pk_short = _pk_short(pk)
        sub_dir = self._root / run_id / pk_short
        sub_dir.mkdir(parents=True, exist_ok=True)
        (sub_dir / "manifest.json").write_text(manifest.to_json())
        sub = Submission(
            run_id=run_id,
            embodiment=manifest.embodiment,
            task_name=manifest.task_name,
            submitter_identity=manifest.submitter_identity or "anonymous",
            submitter_pk=pk or pk_short,
            ingest_ts=time.time(),
            accepted=True,
            reject_reason=None,
            model_name=manifest.model.name,
        )
        (sub_dir / "meta.json").write_text(json.dumps(asdict(sub), indent=2, sort_keys=True))
        self._update_index(sub)
        self._append_ledger(sub)
        return sub

    def reject(self, manifest_payload: str, reason: str, attempted_identity: str | None) -> Submission:
        # Synthesize a placeholder run_id keyed on ingest time so we can
        # still record the attempt.
        ts = time.time()
        run_id = f"rejected-{int(ts * 1000)}"
        pk_short = f"anonymous-{int(ts * 1000)}"
        sub_dir = self._root / run_id / pk_short
        sub_dir.mkdir(parents=True, exist_ok=True)
        (sub_dir / "manifest.json").write_text(manifest_payload)
        sub = Submission(
            run_id=run_id,
            embodiment="?",
            task_name="?",
            submitter_identity=attempted_identity or "anonymous",
            submitter_pk=pk_short,
            ingest_ts=ts,
            accepted=False,
            reject_reason=reason,
        )
        (sub_dir / "meta.json").write_text(json.dumps(asdict(sub), indent=2, sort_keys=True))
        self._update_index(sub)
        self._append_ledger(sub)
        return sub

    def list_submissions(self) -> list[Submission]:
        if not self._index_path.exists():
            return []
        raw = json.loads(self._index_path.read_text())
        # Pre-existing index.json entries don't have `model_name`;
        # default to "" so loads stay backwards-compatible.
        return [Submission(**{**{"model_name": ""}, **r}) for r in raw]

    def list_for_run_id(self, run_id: str) -> list[Submission]:
        """All submissions matching a given run_id, sorted by ingest_ts.

        Cross-device corroboration: when multiple distinct submitters
        submit the same run_id (because they re-ran the same pinned-input
        sweep on different devices), this lets the portal surface that as
        corroboration.
        """
        return [s for s in self.list_submissions() if s.run_id == run_id]

    def get(self, run_id: str, submitter_pk: str | None = None) -> tuple[Submission, Manifest] | None:
        """Resolve one (run_id, submitter) pair. If submitter_pk is None,
        returns the first match for run_id (preserving single-tenant
        callers)."""
        candidates = self.list_for_run_id(run_id)
        if not candidates:
            return None
        if submitter_pk is not None:
            candidates = [s for s in candidates if s.submitter_pk == submitter_pk]
            if not candidates:
                return None
        sub = candidates[0]
        sub_dir = self._root / sub.run_id / _pk_short(sub.submitter_pk if sub.submitter_pk and not sub.submitter_pk.startswith("anonymous-") else None)
        # The _pk_short() result on accept() is what's on disk; recompute
        # the same way for retrieval. For anonymous-<ts> the value IS the
        # pk_short, so we use it directly.
        if sub.submitter_pk.startswith("anonymous-"):
            sub_dir = self._root / sub.run_id / sub.submitter_pk
        else:
            sub_dir = self._root / sub.run_id / sub.submitter_pk[:16]
        meta_path = sub_dir / "meta.json"
        manifest_path = sub_dir / "manifest.json"
        if not (meta_path.exists() and manifest_path.exists()):
            return None
        manifest = Manifest.from_json(manifest_path.read_text())
        return sub, manifest

    def _update_index(self, sub: Submission) -> None:
        existing = self.list_submissions()
        # Replace any existing entry for the same (run_id, submitter_pk)
        # so re-submission updates the timestamp instead of duplicating.
        existing = [s for s in existing if not (s.run_id == sub.run_id and s.submitter_pk == sub.submitter_pk)]
        existing.append(sub)
        existing.sort(key=lambda s: s.ingest_ts, reverse=True)
        self._index_path.write_text(json.dumps([asdict(s) for s in existing], indent=2, sort_keys=True))

    def _append_ledger(self, sub: Submission) -> None:
        """Append a single JSON line to the append-only ledger.

        Trust model: append-only **by convention** (this code only
        appends, never rewrites). It is NOT cryptographically tamper-
        evident — the portal operator could rewrite or hide entries.
        The signature proves the submitter held the key (Ed25519 over the
        manifest). It does NOT prove the ledger of accepted submissions
        is complete. Sigstore in v1.0 closes that hole.
        """
        line = json.dumps({
            "ts": sub.ingest_ts,
            "run_id": sub.run_id,
            "submitter_identity": sub.submitter_identity,
            "submitter_pk": sub.submitter_pk,
            "accepted": sub.accepted,
            "reject_reason": sub.reject_reason,
        }, sort_keys=True)
        with self._ledger_path.open("a") as f:
            f.write(line + "\n")

    @property
    def ledger_path(self) -> Path:
        return self._ledger_path
