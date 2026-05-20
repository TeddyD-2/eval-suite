"""Top-level ingest CLI dispatcher tests.

**In plain words.** Confirms `python -m eval_suite.ingest --help`
works without any heavy converter library installed, and that the
three sub-commands (splat / rgbd / objaverse) all dispatch
correctly.

The dispatcher routes `python -m eval_suite.ingest <source> ...` to
the right subpackage. We exercise:

  1. `--help` works without any extras installed (no torch / open3d /
     objaverse pull).
  2. Unknown sources are rejected.
  3. Forward-to-subcommand dispatch reaches the right module.
"""

from __future__ import annotations

import subprocess
import sys


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "eval_suite.ingest", *args],
        capture_output=True, text=True, check=False,
    )


def test_ingest_help_works_without_extras() -> None:
    result = _run_cli(["--help"])
    assert result.returncode == 0
    assert "splat" in result.stdout
    assert "rgbd" in result.stdout
    assert "objaverse" in result.stdout


def test_ingest_help_does_not_import_heavy_modules() -> None:
    code = (
        "import sys\n"
        "import eval_suite.ingest.cli as m\n"
        "m._build_parser()\n"
        "forbidden = ['open3d', 'objaverse', 'torch', 'lerobot', 'mujoco', 'mujoco_playground']\n"
        "found = [k for k in sys.modules if k in forbidden or "
        "         any(k.startswith(f + '.') for f in forbidden)]\n"
        "assert not found, f'help path must not pull heavy deps; got {found}'\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
    )
    assert "ok" in result.stdout


def test_ingest_unknown_source_rejected() -> None:
    result = _run_cli(["banana"])
    assert result.returncode != 0
    assert "unknown ingest source" in result.stderr or "invalid choice" in result.stderr


def test_ingest_rgbd_help_works_without_open3d() -> None:
    """The rgbd CLI must parse --help without Open3D installed."""
    result = _run_cli(["rgbd", "--help"])
    assert result.returncode == 0
    assert "frames_dir" in result.stdout
    assert "TSDF" in result.stdout or "scene_metadata" in result.stdout


def test_ingest_objaverse_help_works_without_client() -> None:
    """The objaverse CLI must parse --help without the objaverse client installed."""
    result = _run_cli(["objaverse", "--help"])
    assert result.returncode == 0
    assert "uid" in result.stdout
    assert "license" in result.stdout
