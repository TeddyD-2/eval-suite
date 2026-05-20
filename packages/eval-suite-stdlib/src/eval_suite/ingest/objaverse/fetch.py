"""Objaverse-XL fetch + license enforcement.

**In plain words.** The downloader. Checks the asset's license
against the allowed list first; if it's not on the allowlist, the
fetch is refused with a clear error before any bytes hit the disk.


Lazy-imports the `objaverse` client. The license allowlist is checked
*before* download so a refused asset doesn't even hit the local cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Default to commercial-friendly subset (the eval-suite is MIT and we
# don't want adopters' downstream products to inherit a viral or
# non-commercial constraint from a sim asset).
DEFAULT_LICENSE_ALLOWLIST: tuple[str, ...] = ("CC-BY", "CC-BY-4.0", "CC0", "CC0-1.0", "PUBLIC")


class ObjaverseUnavailableError(RuntimeError):
    pass


class AssetLicenseRefusedError(RuntimeError):
    pass


@dataclass(frozen=True)
class FetchedAsset:
    uid: str
    local_path: Path
    license: str
    metadata: dict[str, str]


def fetch_asset(
    *,
    uid: str,
    license_allowlist: tuple[str, ...] = DEFAULT_LICENSE_ALLOWLIST,
    cache_dir: Path | None = None,
) -> FetchedAsset:
    """Resolve a single Objaverse-XL asset by uid.

    Raises AssetLicenseRefusedError if the asset's declared license
    isn't in `license_allowlist`. Raises ObjaverseUnavailableError if
    the `objaverse` client isn't installed.
    """
    try:
        import objaverse
    except ImportError as e:
        raise ObjaverseUnavailableError(
            "`objaverse` not installed. `pip install 'eval-suite-stdlib[objaverse]'`."
        ) from e

    if cache_dir is not None:
        objaverse.objaverse_dir = str(cache_dir)
    annotations = objaverse.load_annotations([uid])
    if uid not in annotations:
        raise KeyError(f"Objaverse uid {uid!r} not found in annotations index")
    meta = annotations[uid]
    declared_license = _normalize_license(meta.get("license"))
    if declared_license not in license_allowlist:
        raise AssetLicenseRefusedError(
            f"Objaverse asset {uid!r} declared license {declared_license!r} "
            f"not in allowlist {license_allowlist!r}. Override --license-allowlist "
            f"at your own risk."
        )
    paths = objaverse.load_objects([uid])
    local = Path(paths[uid])
    return FetchedAsset(
        uid=uid,
        local_path=local,
        license=declared_license,
        metadata={k: str(v) for k, v in meta.items() if isinstance(v, (str, int, float, bool))},
    )


def _normalize_license(raw: object) -> str:
    """Normalize Objaverse's license strings to the allowlist's vocabulary."""
    if not isinstance(raw, str):
        return "UNKNOWN"
    s = raw.strip().upper().replace("_", "-").replace(" ", "-")
    aliases = {
        "BY": "CC-BY",
        "BY-4.0": "CC-BY-4.0",
        "CCBY": "CC-BY",
        "CCBY40": "CC-BY-4.0",
        "CC-BY40": "CC-BY-4.0",
    }
    return aliases.get(s, s)
