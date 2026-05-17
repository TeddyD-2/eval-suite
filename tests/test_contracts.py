"""Contract tests — MockPolicy + MockTask + GymAdapter round-trip.

These tests run without any GPU, simulator, or TF/JAX. They verify that:
- MockPolicy and MockTask satisfy the Protocol contracts (runtime check).
- The Adapter drives a rollout end-to-end against MockTask.
- The Manifest seals, verifies, and round-trips through JSON.
- The Wilson CI and worst-axis aggregation produce expected values.

This is the contract the GitHub Actions workflow runs on every push.
"""

from __future__ import annotations

from eval_suite._types import CellResult
from eval_suite.adapters import GymAdapter
from eval_suite.contracts import Adapter, Manifest, Policy, Task
from eval_suite.manifest import (
    SCHEMA_VERSION,
    CalibrationRef,
    CellResultPayload,
    HardwareRef,
    ModelRef,
    SimulatorRef,
)
from eval_suite.manifest import (
    Manifest as ManifestImpl,
)
from eval_suite.policies import MockPolicy
from eval_suite.statistics import per_axis_means, wilson_ci, worst_axis
from eval_suite.tasks import MockTask


def test_mock_policy_satisfies_protocol() -> None:
    pol = MockPolicy()
    assert isinstance(pol, Policy)
    assert pol.name == "mock-zero"
    assert pol.checkpoint_id.startswith("mock:")


def test_mock_task_satisfies_protocol() -> None:
    task = MockTask(n_cells=3)
    assert isinstance(task, Task)
    assert task.n_cells == 3
    assert task.embodiment == "mock"
    cell_0 = task.cell_id(0)
    assert cell_0.axes == {"axis_a": "level0", "axis_b": "level0"}


def test_adapter_satisfies_protocol() -> None:
    adapter = GymAdapter()
    assert isinstance(adapter, Adapter)


def test_adapter_rollout_against_mock_task() -> None:
    pol = MockPolicy(terminate_after=3)
    task = MockTask(n_cells=2, max_episode_steps=10)
    adapter = GymAdapter()
    result = adapter.rollout(pol, task, cell=0, seed=42)
    assert result.cell == task.cell_id(0)
    assert result.seed == 42
    assert not result.success  # mock env never reports success
    assert 1 <= result.num_steps <= 10
    assert result.elapsed_wall_seconds >= 0.0


def test_wilson_ci_known_values() -> None:
    # 14 successes out of 20: Wilson 95% should bracket 0.70
    low, high = wilson_ci(14, 20)
    assert 0.4 <= low <= 0.55
    assert 0.85 <= high <= 0.92
    # 0/20: lower bound 0, upper bound modest
    low0, high0 = wilson_ci(0, 20)
    assert low0 == 0.0
    assert 0.10 <= high0 <= 0.20
    # 20/20: tight upper bound, lower around 0.83
    lowN, highN = wilson_ci(20, 20)
    assert highN == 1.0
    assert 0.80 <= lowN <= 0.90
    # Empty: full unit interval
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_per_axis_and_worst_axis() -> None:
    from eval_suite._types import CellId

    cells = [
        CellResult(
            cell=CellId(embodiment="x", task="t", axes={"a": "lo", "b": "x"}),
            n_trials=10, successes=9, wilson_ci_low=0.5, wilson_ci_high=1.0,
        ),
        CellResult(
            cell=CellId(embodiment="x", task="t", axes={"a": "hi", "b": "x"}),
            n_trials=10, successes=2, wilson_ci_low=0.0, wilson_ci_high=0.5,
        ),
        CellResult(
            cell=CellId(embodiment="x", task="t", axes={"a": "lo", "b": "y"}),
            n_trials=10, successes=8, wilson_ci_low=0.4, wilson_ci_high=0.95,
        ),
    ]
    per_axis = per_axis_means(cells)
    assert set(per_axis.keys()) == {"a", "b"}
    assert abs(per_axis["a"]["lo"] - (0.9 + 0.8) / 2) < 1e-9
    assert abs(per_axis["a"]["hi"] - 0.2) < 1e-9

    axis_name, axis_score = worst_axis(per_axis)
    # axis "a" has level-mean (0.85 + 0.20) / 2 = 0.525
    # axis "b" has level-mean (0.55 + 0.80) / 2 = 0.675
    # so worst axis is "a"
    assert axis_name == "a"
    assert abs(axis_score - 0.525) < 1e-9


def _example_manifest() -> ManifestImpl:
    return ManifestImpl(
        schema_version=SCHEMA_VERSION,
        code_sha="deadbeef",
        container_digest="",
        model=ModelRef(name="rt1-converged", checkpoint_sha256="abc123", family="rt1"),
        simulator=SimulatorRef(name="simpler-env", commit="06accac"),
        task_name="pick_coke_can",
        embodiment="google_robot",
        trials_per_cell=20,
        cells=[
            CellResultPayload(
                cell_id="google_robot/pick_coke_can/orientation=upright",
                axes={"orientation": "upright"},
                n_trials=20, successes=14, wilson_ci_low=0.49, wilson_ci_high=0.85,
            ),
        ],
        hardware=HardwareRef(gpu="RTX 3090", cuda="12.4", driver="580.126.20"),
        seeds=list(range(20)),
        calibration=CalibrationRef(tier="B", real_perf_source="SimplerEnv paper", real_perf_value=0.83),
    )


def test_manifest_seal_verify_roundtrip() -> None:
    m = _example_manifest()
    assert not m.verify()  # unsealed
    m.seal()
    assert m.run_id  # populated
    assert m.verify()
    # round-trip through JSON
    payload = m.to_json()
    m2 = ManifestImpl.from_json(payload)
    assert m2.verify()
    assert m2.run_id == m.run_id


def test_manifest_run_id_content_addressed() -> None:
    m1 = _example_manifest()
    m1.seal()
    m2 = _example_manifest()
    m2.seal()
    assert m1.run_id == m2.run_id  # same inputs → same id

    m3 = _example_manifest()
    # Tweak one input and rehash
    m3.cells[0] = CellResultPayload(
        cell_id="google_robot/pick_coke_can/orientation=upright",
        axes={"orientation": "upright"},
        n_trials=20, successes=15, wilson_ci_low=0.49, wilson_ci_high=0.85,
    )
    m3.seal()
    assert m3.run_id != m1.run_id  # any input change → new id


def test_manifest_protocol() -> None:
    m = _example_manifest()
    m.seal()
    assert isinstance(m, Manifest)
