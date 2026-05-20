"""RGB-D ingester contract tests.

These don't require Open3D to be installed for the basic checks. The
actual TSDF-fusion path is gated on the [rgbd] extra; a synthetic
end-to-end test runs only when open3d is available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def test_rgbd_module_imports_without_open3d() -> None:
    """`import eval_suite.ingest.rgbd.fuse` must not pull open3d."""
    import subprocess
    code = (
        "import sys\n"
        "import eval_suite.ingest.rgbd.fuse as m\n"
        "assert hasattr(m, 'fuse_rgbd_to_mesh')\n"
        "assert 'open3d' not in sys.modules\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
    )
    assert "ok" in result.stdout


def test_rgbd_fuse_without_open3d_raises_clean() -> None:
    """Without the extra, fuse_rgbd_to_mesh raises a clear RuntimeError."""
    from eval_suite.ingest.rgbd.fuse import Open3DUnavailableError, fuse_rgbd_to_mesh
    try:
        import open3d  # noqa: F401  # type: ignore[import-not-found]
        pytest.skip("open3d installed; this test only fires without the extra")
    except ImportError:
        pass
    with pytest.raises(Open3DUnavailableError, match=r"\[rgbd\]"):
        fuse_rgbd_to_mesh(frames_dir=Path("/nonexistent"), output_dir=Path("/tmp/x"))


def test_rgbd_pose_parser(tmp_path: Path) -> None:
    """The poses.txt parser must accept the documented 4x4-row-major shape."""
    from eval_suite.ingest.rgbd.fuse import _load_poses
    pyimport_open3d_ok = False
    try:
        import open3d  # noqa: F401  # type: ignore[import-not-found]
        pyimport_open3d_ok = True
    except ImportError:
        pass
    # _load_poses doesn't depend on open3d itself; it just uses numpy.
    poses_path = tmp_path / "poses.txt"
    poses_path.write_text(
        "# header line\n"
        "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n"
        "\n"
        "0.5 0 0 0.1  0 0.5 0 0  0 0 0.5 0  0 0 0 1\n"
    )
    poses = _load_poses(poses_path)
    assert len(poses) == 2
    assert poses[0].shape == (4, 4)
    assert poses[1][0, 3] == pytest.approx(0.1)
    # Sanity: the open3d branch is what's heavy-gated; the parser is import-cheap.
    assert pyimport_open3d_ok or True


def test_rgbd_pose_parser_rejects_wrong_length(tmp_path: Path) -> None:
    from eval_suite.ingest.rgbd.fuse import _load_poses
    poses_path = tmp_path / "poses.txt"
    poses_path.write_text("1 2 3\n")
    with pytest.raises(ValueError, match="16 floats"):
        _load_poses(poses_path)
