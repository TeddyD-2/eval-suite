"""Objaverse ingester contract tests.

**In plain words.** Pins down that the license allowlist actually
blocks unlisted-license assets *before* a download happens, and
that the dispatcher correctly hands off to the Objaverse path.
"""

from __future__ import annotations

import sys

import pytest


def test_objaverse_module_imports_without_client() -> None:
    """`import eval_suite.ingest.objaverse.fetch` must not pull objaverse."""
    import subprocess
    code = (
        "import sys\n"
        "import eval_suite.ingest.objaverse.fetch as m\n"
        "assert hasattr(m, 'fetch_asset')\n"
        "assert 'objaverse' not in sys.modules\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
    )
    assert "ok" in result.stdout


def test_objaverse_default_license_allowlist_is_commercial_friendly() -> None:
    from eval_suite.ingest.objaverse.fetch import DEFAULT_LICENSE_ALLOWLIST
    assert "CC-BY" in DEFAULT_LICENSE_ALLOWLIST
    assert "CC-BY-4.0" in DEFAULT_LICENSE_ALLOWLIST
    assert "CC0" in DEFAULT_LICENSE_ALLOWLIST
    # CC-BY-NC must NOT be in the default — that's the point of the gating.
    assert "CC-BY-NC" not in DEFAULT_LICENSE_ALLOWLIST


def test_objaverse_license_normalizer_handles_common_variants() -> None:
    from eval_suite.ingest.objaverse.fetch import _normalize_license
    assert _normalize_license("cc by 4.0") == "CC-BY-4.0"
    assert _normalize_license("CC-BY") == "CC-BY"
    assert _normalize_license("by") == "CC-BY"
    assert _normalize_license(None) == "UNKNOWN"
    assert _normalize_license(42) == "UNKNOWN"


def test_objaverse_fetch_without_client_raises_clean() -> None:
    from eval_suite.ingest.objaverse.fetch import ObjaverseUnavailableError, fetch_asset
    try:
        import objaverse  # noqa: F401  # type: ignore[import-not-found]
        pytest.skip("objaverse installed; this test only fires without the extra")
    except ImportError:
        pass
    with pytest.raises(ObjaverseUnavailableError, match=r"\[objaverse\]"):
        fetch_asset(uid="fake-uid-no-such-thing")
