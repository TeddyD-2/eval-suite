"""Conformance kit.

**In plain words.** This is the "is my plugin actually compatible?"
checker for plugin authors. Anyone writing a new task, model, or
simulator bridge calls these helpers from their own test suite to
confirm their code satisfies the suite's four Protocols *before* they
publish their pip package. Without this kit, plugin authors would
discover incompatibilities at run time on someone else's machine;
with it, they catch them on their own `pytest` run.


Plugin authors call these helpers from their own pytest suites to
confirm their Task / Policy / Adapter satisfies the eval-suite
Protocols before publishing. The library shape (vs. a CLI tool) is
deliberate: every plugin author already runs pytest; the kit slots in
as one extra import.

Typical use, in a third-party package's `tests/test_conformance.py`:

    from eval_suite.conformance import verify_task, full_battery
    from my_pkg import MyKitchenTask, MyArmPolicy

    def test_my_task_satisfies_contract():
        verify_task(MyKitchenTask)

    def test_my_kit_round_trips(tmp_path):
        full_battery(MyKitchenTask, MyArmPolicy, MyAdapter, tmp_path)

What each helper checks:

- `verify_task(cls)`        — class is instantiable; satisfies the Task
                              Protocol's required properties + methods;
                              cell_ids are CellId instances with correct
                              embodiment/task identifiers; build_env(0)
                              returns something with reset/step.
- `verify_policy(cls)`      — class is instantiable; reset(str) / step(obs)
                              work; step returns Action or JointAction.
- `verify_adapter(cls)`     — class is instantiable; rollout(...) returns
                              a RolloutResult with the right field types.
- `roundtrip_determinism(...)` — run the same (cell=0, seed=0) twice and
                              assert identical RolloutResult (modulo wall
                              time and rendering artifacts). Catches hidden
                              global state — the #1 cause of cross-device
                              run_id divergence.
- `full_battery(...)`       — all of the above + a 4-trial run_sweep
                              that asserts manifest.verify() passes and
                              the sidecar provenance binds to it.

Conservative semantics: passing the battery doesn't certify your plugin
"correct" — it certifies your plugin satisfies the structural contract
the suite assumes. A plugin can pass the battery and still produce
bad evaluations (broken success criteria, leaky rewards). The battery
is the floor, not the ceiling.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .._types import Action, CellId, JointAction, Observation, RolloutResult
from ..contracts import Adapter, Policy, Task
from ..manifest import Manifest
from ..plugin_provenance import PluginProvenance
from ..sweep import run_sweep


def verify_task(cls: type, *, n_cells_to_check: int = 3, init_kwargs: dict[str, Any] | None = None) -> None:
    """Check that a class satisfies the Task Protocol.

    Constructs an instance, calls the required properties/methods, and
    asserts shapes. Builds the first `n_cells_to_check` envs to catch
    cell-index-out-of-bounds bugs.

    Raises AssertionError with a specific message on failure.
    """
    inst = cls(**(init_kwargs or {}))
    assert isinstance(inst, Task), (
        f"{cls.__name__} does not satisfy the Task Protocol "
        f"(runtime_checkable isinstance check failed)"
    )
    assert isinstance(inst.name, str) and inst.name, f"{cls.__name__}.name must be a non-empty str"
    assert isinstance(inst.embodiment, str) and inst.embodiment, (
        f"{cls.__name__}.embodiment must be a non-empty str"
    )
    n = inst.n_cells
    assert isinstance(n, int) and n > 0, f"{cls.__name__}.n_cells must be a positive int"
    assert isinstance(inst.max_episode_steps, int) and inst.max_episode_steps > 0
    for i in range(min(n, n_cells_to_check)):
        cid = inst.cell_id(i)
        assert isinstance(cid, CellId), f"cell_id({i}) must return CellId"
        assert cid.embodiment == inst.embodiment, "CellId.embodiment must match Task.embodiment"
        assert cid.task == inst.name, "CellId.task must match Task.name"
        assert isinstance(cid.axes, dict), "CellId.axes must be dict[str, str]"
        env = inst.build_env(i)
        assert hasattr(env, "reset") and callable(env.reset), "Task.build_env returned object without reset()"
        assert hasattr(env, "step") and callable(env.step), "Task.build_env returned object without step()"
        # Best-effort close — some envs don't have it.
        close = getattr(env, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def verify_policy(cls: type, *, init_kwargs: dict[str, Any] | None = None) -> None:
    """Check that a class satisfies the Policy Protocol.

    Constructs an instance, calls reset() + step(), and asserts the
    return type is Action or JointAction.
    """
    inst = cls(**(init_kwargs or {}))
    assert isinstance(inst, Policy), (
        f"{cls.__name__} does not satisfy the Policy Protocol"
    )
    assert isinstance(inst.name, str) and inst.name
    assert isinstance(inst.checkpoint_id, str)  # may be empty for placeholder policies
    inst.reset("conformance check instruction")
    obs = Observation(image=np.zeros((1, 1, 3), dtype=np.uint8), instruction="conformance check")
    action = inst.step(obs)
    assert isinstance(action, Action | JointAction), (
        f"{cls.__name__}.step() returned {type(action).__name__}; "
        f"must be Action (7-DoF EEF) or JointAction (N-DoF joint targets)"
    )


def verify_adapter(cls: type, *, init_kwargs: dict[str, Any] | None = None) -> None:
    """Check that a class satisfies the Adapter Protocol.

    Constructs an instance and asserts the rollout() signature. Doesn't
    run an actual rollout here — `full_battery` does that.
    """
    inst = cls(**(init_kwargs or {}))
    assert isinstance(inst, Adapter), f"{cls.__name__} does not satisfy the Adapter Protocol"
    assert isinstance(inst.name, str) and inst.name


def roundtrip_determinism(
    task_cls: type, policy_cls: type, adapter_cls: type,
    *, task_kwargs: dict[str, Any] | None = None,
    policy_kwargs: dict[str, Any] | None = None,
    adapter_kwargs: dict[str, Any] | None = None,
    n_iterations: int = 2, cell: int = 0, seed: int = 0,
) -> None:
    """Run the same (cell, seed) pair `n_iterations` times; assert
    identical RolloutResult (modulo wall time).

    Catches hidden global state — Task that caches between cells in a
    `/tmp` file, Policy that mutates a shared RNG without reseeding,
    etc. These bugs are the most common cause of "my manifest's run_id
    is different on a different machine."
    """
    task = task_cls(**(task_kwargs or {}))
    policy = policy_cls(**(policy_kwargs or {}))
    adapter = adapter_cls(**(adapter_kwargs or {}))
    results: list[RolloutResult] = []
    for _ in range(n_iterations):
        results.append(adapter.rollout(policy=policy, task=task, cell=cell, seed=seed))
    base = results[0]
    for i, r in enumerate(results[1:], start=1):
        assert r.success == base.success, (
            f"determinism: iteration {i} success={r.success} differs from iteration 0 ({base.success})"
        )
        assert r.num_steps == base.num_steps, (
            f"determinism: iteration {i} num_steps={r.num_steps} differs from iteration 0 ({base.num_steps})"
        )


def full_battery(
    task_cls: type, policy_cls: type, adapter_cls: type, tmp_path: Path,
    *, task_kwargs: dict[str, Any] | None = None,
    policy_kwargs: dict[str, Any] | None = None,
    adapter_kwargs: dict[str, Any] | None = None,
    trials_per_cell: int = 2, cells_to_sweep: int = 2,
) -> None:
    """Run the full conformance battery against a (Task, Policy, Adapter) triple.

    Asserts each individual Protocol satisfaction, plus a full
    `run_sweep` end-to-end with manifest verify() and provenance
    sidecar binding.

    Pass `tmp_path` from a pytest fixture or create one yourself.
    """
    verify_task(task_cls, init_kwargs=task_kwargs)
    verify_policy(policy_cls, init_kwargs=policy_kwargs)
    verify_adapter(adapter_cls, init_kwargs=adapter_kwargs)
    roundtrip_determinism(
        task_cls, policy_cls, adapter_cls,
        task_kwargs=task_kwargs, policy_kwargs=policy_kwargs, adapter_kwargs=adapter_kwargs,
    )

    task = task_cls(**(task_kwargs or {}))
    policy = policy_cls(**(policy_kwargs or {}))
    adapter = adapter_cls(**(adapter_kwargs or {}))
    cells = list(range(min(cells_to_sweep, task.n_cells)))
    manifest = run_sweep(
        policy=policy, task=task, adapter=adapter,
        seeds=list(range(trials_per_cell)),
        cells=cells, output_dir=tmp_path,
        code_sha="conformance",
    )
    assert manifest.verify(), "Manifest.verify() must return True after run_sweep"

    # The sidecar must bind to this manifest's run_id.
    sidecar_path = tmp_path / "plugin_provenance.json"
    assert sidecar_path.exists(), "run_sweep must write plugin_provenance.json next to manifest"
    sidecar = PluginProvenance.load(sidecar_path)
    assert sidecar.target_run_id == manifest.run_id, (
        f"sidecar target_run_id ({sidecar.target_run_id}) must equal manifest.run_id ({manifest.run_id})"
    )
    assert sidecar.verify(), "PluginProvenance.verify() must return True for unsigned sidecar"

    # CSV exists and has at least the per-trial rows.
    csv_path = tmp_path / "trials.csv"
    assert csv_path.exists(), "run_sweep must write trials.csv"
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == trials_per_cell * len(cells), (
        f"trials.csv should have {trials_per_cell * len(cells)} rows, has {len(rows)}"
    )

    # Manifest JSON round-trips: load it back from disk and verify.
    reloaded = Manifest.from_json((tmp_path / "manifest.json").read_text())
    assert reloaded.verify(), "Manifest loaded from disk must verify()"
    assert reloaded.run_id == manifest.run_id, "Manifest run_id must round-trip through JSON"
