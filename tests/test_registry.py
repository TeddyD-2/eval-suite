"""v0 registry contract tests.

Covers:
- Listing returns the in-tree plugins without loading them (lazy-load).
- A broken plugin doesn't kill the listing call.
- Ambiguous short names raise EvalSuiteAmbiguousPluginError.
- Qualified `pkg:name` form resolves correctly.
- `--help` doesn't import any of the heavy plugin modules.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from eval_suite.registry import (
    EvalSuiteAmbiguousPluginError,
    EvalSuitePluginNotFoundError,
    PluginEntry,
    get_task,
    list_failed,
    list_tasks,
)


def test_in_tree_plugins_are_discovered() -> None:
    """The five in-tree tasks register under eval_suite.tasks."""
    names = {e.name for e in list_tasks()}
    assert "mock" in names
    assert "google_robot_pick_coke_can" in names
    assert "widowx_spoon_on_towel" in names
    assert "unitree_go1_joystick" in names
    assert "libero_spatial" in names


def test_listing_does_not_load_heavy_modules() -> None:
    """list_tasks() must not import TF/JAX/SimplerEnv/MuJoCo Playground."""
    # Run in a fresh subprocess so we have a clean sys.modules baseline.
    code = (
        "import sys\n"
        "from eval_suite.registry import list_tasks, list_policies, list_adapters\n"
        "list_tasks(); list_policies(); list_adapters()\n"
        "forbidden = ['tensorflow', 'jax', 'simpler_env', 'mujoco_playground', 'octo', 'torch']\n"
        "found = [m for m in forbidden if any(k == m or k.startswith(m + '.') for k in sys.modules)]\n"
        "print('found:', found)\n"
        "assert not found, f'listing must not load heavy modules; found {found}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True,
    )
    assert "found: []" in result.stdout, result.stdout


def test_get_task_short_form_works() -> None:
    cls = get_task("mock")
    assert cls.__name__ == "MockTask"


def test_get_task_qualified_form_works() -> None:
    cls = get_task("eval-suite-stdlib:mock")
    assert cls.__name__ == "MockTask"


def test_get_task_not_found_raises() -> None:
    try:
        get_task("does_not_exist")
    except EvalSuitePluginNotFoundError as e:
        assert "does_not_exist" in str(e)
        return
    raise AssertionError("expected EvalSuitePluginNotFoundError")


def test_ambiguous_short_name_errors_with_candidates() -> None:
    """When two plugins register the same short name, the short form errors
    with a message listing the qualified candidates. The qualified form
    works as the disambiguation path."""
    # Inject two PluginEntries with the same short name into _resolve's input.
    from eval_suite import registry

    fake_entries = [
        PluginEntry(name="duplicate", package_name="pkg-a", package_version="1.0",
                    entry_point_ref="pkg_a:Duplicate", group="eval_suite.tasks"),
        PluginEntry(name="duplicate", package_name="pkg-b", package_version="1.0",
                    entry_point_ref="pkg_b:Duplicate", group="eval_suite.tasks"),
    ]
    with patch.object(registry, "_collect", return_value=fake_entries):
        try:
            registry._resolve("eval_suite.tasks", "duplicate")
        except EvalSuiteAmbiguousPluginError as e:
            msg = str(e)
            assert "pkg-a:duplicate" in msg and "pkg-b:duplicate" in msg
            return
        raise AssertionError("expected EvalSuiteAmbiguousPluginError")


def test_failed_plugin_does_not_break_listing() -> None:
    """If one plugin's entry-point metadata is corrupt, listing the other
    plugins still returns them; the broken one lands in list_failed()."""
    from eval_suite import registry

    # Simulate one entry whose dist resolution raises.
    class _BrokenEntry:
        name = "broken_plugin"
        value = "broken_pkg:Broken"
        @property
        def dist(self) -> object:  # pragma: no cover (intentional)
            raise RuntimeError("simulating bad plugin metadata")

    class _GoodEntry:
        name = "mock"
        value = "eval_suite.tasks.mock:MockTask"
        @property
        def dist(self) -> object:
            import importlib.metadata
            return importlib.metadata.distribution("eval-suite-stdlib")

    def _fake_entry_points(group: str) -> list:  # type: ignore[type-arg]
        if group == "eval_suite.tasks":
            return [_BrokenEntry(), _GoodEntry()]
        return []

    with patch("importlib.metadata.entry_points", side_effect=_fake_entry_points):
        entries = registry.list_tasks()
        assert any(e.name == "mock" for e in entries), "good plugin must still appear"
        failed = list_failed()
        assert any("broken_plugin" in name for name, _ in failed["eval_suite.tasks"]), (
            "broken plugin must appear in list_failed()"
        )


def test_help_does_not_trigger_plugin_load(tmp_path: Path) -> None:
    """`python -m eval_suite.cli --help` must NOT import heavy plugin modules.

    This catches the regression class where someone adds
    `from .tasks.simpler_env import GoogleRobotPickCokeCan` at the top
    of cli.py "just for type hints" and inadvertently makes --help slow.
    """
    code = (
        "import sys\n"
        "from eval_suite import cli\n"
        "try:\n"
        "    cli.main(['--help'])\n"
        "except SystemExit:\n"
        "    pass\n"
        "forbidden = ['tensorflow', 'jax', 'simpler_env', 'mujoco_playground', 'octo']\n"
        "found = [m for m in forbidden if any(k == m or k.startswith(m + '.') for k in sys.modules)]\n"
        "print('found:', found)\n"
        "assert not found, f'--help must not load heavy modules; found {found}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True,
    )
    assert "found: []" in result.stdout, result.stdout
