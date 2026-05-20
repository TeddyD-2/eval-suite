"""Conformance test for the external plugin demo.

**In plain words.** The pytest a third-party plugin author would
copy into their own repo to confirm their code satisfies the
suite's protocols *before* they publish. If this passes, the
plugin is contract-clean.


This is the test a third-party plugin author would write in their own
repo before publishing. It calls `eval_suite.conformance.full_battery`
to confirm the Task / Policy / Adapter triple satisfies the v0
contract.

Run with: `pytest examples/external_plugin_demo/tests/test_conformance.py`
"""

from __future__ import annotations

from pathlib import Path

from eval_suite.adapters import GymAdapter
from eval_suite.conformance import (
    full_battery,
    roundtrip_determinism,
    verify_policy,
    verify_task,
)
from plugin_demo import HouseholdMockTask, IndustrialArmMockPolicy


def test_household_task_satisfies_protocol() -> None:
    verify_task(HouseholdMockTask)


def test_industrial_policy_satisfies_protocol() -> None:
    verify_policy(IndustrialArmMockPolicy)


def test_roundtrip_is_deterministic() -> None:
    roundtrip_determinism(HouseholdMockTask, IndustrialArmMockPolicy, GymAdapter)


def test_full_battery(tmp_path: Path) -> None:
    full_battery(
        HouseholdMockTask, IndustrialArmMockPolicy, GymAdapter, tmp_path,
        trials_per_cell=2, cells_to_sweep=2,
    )
