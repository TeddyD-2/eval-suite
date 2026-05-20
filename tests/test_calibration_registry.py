"""Calibration registry contract — v0 JSON file format.

**In plain words.** Pins down the shape of `real_perf.json` (the
published real-robot numbers the suite ships with). If this test
ever fails, the calibration overlay in the notebook and the
calibration demo can't trust the registry anymore.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eval_suite.analysis import _load_real_perf, _load_real_perf_full


def test_default_registry_loads() -> None:
    """The shipped calibration/real_perf.json parses and contains the
    canonical v0 tier-B entry (Octo on Google Robot pick coke can)."""
    table = _load_real_perf()
    assert "google_robot_pick_coke_can" in table
    assert "octo-base" in table["google_robot_pick_coke_can"]
    assert abs(table["google_robot_pick_coke_can"]["octo-base"] - 0.293) < 1e-6


def test_registry_provenance_fields() -> None:
    entries = _load_real_perf_full()
    assert len(entries) >= 5  # SimplerEnv paper Table 3 + Table 4 entries
    for e in entries:
        assert {"task_key", "model_key", "value", "source", "contributor"}.issubset(e.keys())
        assert isinstance(e["value"], int | float)
        assert 0.0 <= float(e["value"]) <= 1.0


def test_env_var_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """v2.0's customer-deployment pipeline writes to an alternate registry
    location via EVAL_SUITE_CALIBRATION_REGISTRY — make sure the lookup
    honors it."""
    custom = tmp_path / "custom_registry.json"
    custom.write_text(json.dumps({
        "entries": [
            {
                "task_key": "factory_x_widgets",
                "model_key": "our-custom-model",
                "value": 0.91,
                "n_trials": 200,
                "source": "internal Q1 deployment",
                "source_url": "",
                "hardware": "Acme Robotics arm v3 (real, factory floor)",
                "date_iso": "2026-04-01",
                "contributor": "factory-x-deployment-team",
                "notes": "first batch of customer-deployment telemetry",
            }
        ]
    }))
    monkeypatch.setenv("EVAL_SUITE_CALIBRATION_REGISTRY", str(custom))
    table = _load_real_perf()
    assert "factory_x_widgets" in table
    assert table["factory_x_widgets"]["our-custom-model"] == 0.91


def test_missing_registry_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_SUITE_CALIBRATION_REGISTRY", str(tmp_path / "no-such-file.json"))
    assert _load_real_perf() == {}
    assert _load_real_perf_full() == []
