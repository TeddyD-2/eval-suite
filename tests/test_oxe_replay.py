"""OXE replay contract tests.

Don't require `tensorflow-datasets` to be installed — exercise:

  1. Registration via entry-points.
  2. Lazy-load: importing the module doesn't pull TF.
  3. Error surface: construction without the right extra raises
     RuntimeError with an install hint.
  4. The unknown-format guard.

End-to-end replay against a real OXE / LeRobot dataset runs in the
dedicated `lerobot-ci.yml` workflow (it shares the extras).
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from eval_suite.registry import get_policy, list_policies


def test_oxe_replay_policy_is_registered() -> None:
    names = {e.name for e in list_policies()}
    assert "oxe_replay" in names


def test_oxe_replay_module_imports_without_heavy_deps() -> None:
    code = (
        "import sys\n"
        "import eval_suite.policies.oxe_replay as m\n"
        "assert hasattr(m, 'OXEReplayPolicy')\n"
        "forbidden = ['tensorflow', 'tensorflow_datasets', 'datasets']\n"
        "found = [k for k in sys.modules if k in forbidden or "
        "         any(k.startswith(f + '.') for f in forbidden)]\n"
        "assert not found, f'module import must not pull TF/datasets; got {found}'\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True,
    )
    assert "ok" in result.stdout, result.stdout


def test_oxe_replay_construction_without_extra_raises_runtime_error() -> None:
    cls = get_policy("oxe_replay")
    try:
        import tensorflow_datasets  # type: ignore[import-not-found]  # noqa: F401
        pytest.skip("tensorflow-datasets is installed; this test only fires without the extra")
    except ImportError:
        pass
    with pytest.raises(RuntimeError, match=r"\[oxe\]"):
        cls(dataset_id="bridge_dataset", episode_id=0)


def test_oxe_replay_unknown_format_rejected() -> None:
    cls = get_policy("oxe_replay")
    with pytest.raises(ValueError, match="unknown OXEReplayPolicy format"):
        cls(dataset_id="anything", episode_id=0, format="not-a-real-format")
