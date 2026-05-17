"""v0 conformance kit self-test.

Runs `full_battery(MockTask, MockPolicy, GymAdapter, tmp_path)` to
prove eval-suite is its own first plugin: every helper in the
conformance kit works against the in-tree reference implementations.
If this test passes, an external plugin author can be confident the
helpers work against any class that satisfies the contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from eval_suite.adapters import GymAdapter
from eval_suite.conformance import (
    full_battery,
    roundtrip_determinism,
    verify_adapter,
    verify_policy,
    verify_task,
)
from eval_suite.policies.mock import MockPolicy
from eval_suite.tasks.mock import MockTask


def test_verify_task_passes_for_mock() -> None:
    verify_task(MockTask)


def test_verify_policy_passes_for_mock() -> None:
    verify_policy(MockPolicy)


def test_verify_adapter_passes_for_gym() -> None:
    verify_adapter(GymAdapter)


def test_roundtrip_determinism_passes() -> None:
    roundtrip_determinism(MockTask, MockPolicy, GymAdapter)


def test_full_battery_passes(tmp_path: Path) -> None:
    full_battery(
        MockTask, MockPolicy, GymAdapter, tmp_path,
        task_kwargs={"n_cells": 2},
        trials_per_cell=2, cells_to_sweep=2,
    )


def test_verify_task_rejects_non_task_class() -> None:
    class NotATask:
        pass

    with pytest.raises(AssertionError):
        verify_task(NotATask)
