"""SHA256 helpers for content-addressing manifests and checkpoint trees."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(root: Path) -> str:
    """Hash a directory tree: SHA256 over the sorted list of
    "relpath:filehash" lines. Order-stable; symlinks followed.

    This is what we use to fingerprint downloaded model checkpoints. Two
    checkpoint dirs with identical contents (same files, same bytes) hash
    to the same value regardless of timestamps or download order.
    """
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    lines: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            lines.append(f"{rel}:{sha256_file(path)}")
    return sha256_text("\n".join(lines))


def canonical_json(obj: Any) -> str:
    """JSON with sorted keys and tight separators — bytes-stable across runs."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_dict(obj: dict[str, Any]) -> str:
    """SHA256 of the canonical JSON of `obj`. Used for `run_id`."""
    return sha256_text(canonical_json(obj))
