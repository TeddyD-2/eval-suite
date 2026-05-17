"""End-to-end sweep contract test — Mock policy + Mock task → CSV + manifest.

This is the test that the GitHub Actions workflow runs to verify the
reproducibility substrate is real on every commit. No GPU, no SimplerEnv,
no real model.

Asserts:
1. The sweep loop completes against a Mock policy + Mock task.
2. `trials.csv` is emitted with the expected schema and N×cells rows.
3. `manifest.json` is emitted, sealable, verifiable.
4. The manifest's `run_id` matches the recomputed hash byte-identically.
5. The CSV parses cleanly (round-trip through csv.DictReader).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from eval_suite.adapters import GymAdapter
from eval_suite.manifest import Manifest
from eval_suite.policies import MockPolicy
from eval_suite.sweep import CSV_COLUMNS, run_sweep
from eval_suite.tasks import MockTask


def test_sweep_emits_valid_artifacts(tmp_path: Path) -> None:
    n_cells = 3
    trials = 4
    task = MockTask(n_cells=n_cells, max_episode_steps=5)
    policy = MockPolicy()
    adapter = GymAdapter()

    manifest = run_sweep(
        policy=policy,
        task=task,
        adapter=adapter,
        seeds=range(trials),
        output_dir=tmp_path,
        code_sha="ci-test",
    )

    csv_path = tmp_path / "trials.csv"
    manifest_path = tmp_path / "manifest.json"
    assert csv_path.exists()
    assert manifest_path.exists()

    # --- CSV checks
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == n_cells * trials
    assert {r["model_name"] for r in rows} == {"mock-zero"}
    assert {int(r["cell_index"]) for r in rows} == set(range(n_cells))
    assert set(rows[0].keys()) == set(CSV_COLUMNS)
    for r in rows:
        # Per-row payloads parse cleanly.
        assert json.loads(r["cell_axes"]) == dict(json.loads(r["cell_axes"]))
        assert int(r["success"]) in (0, 1)
        float(r["elapsed_wall_seconds"])

    # --- Manifest checks
    payload = manifest_path.read_text()
    loaded = Manifest.from_json(payload)
    assert loaded.verify(), "manifest run_id must match recomputed hash"
    assert loaded.run_id == manifest.run_id
    assert loaded.trials_per_cell == trials
    assert len(loaded.cells) == n_cells
    assert loaded.code_sha == "ci-test"
    assert loaded.embodiment == "mock"
    # Cell IDs in the manifest match the task's cell_id() output.
    expected_cell_slugs = {task.cell_id(i).slug for i in range(n_cells)}
    assert {c.cell_id for c in loaded.cells} == expected_cell_slugs


def test_sweep_run_id_is_deterministic(tmp_path: Path) -> None:
    """Two sweeps with identical config → identical run_id."""
    def _run(out: Path) -> str:
        manifest = run_sweep(
            policy=MockPolicy(),
            task=MockTask(n_cells=2, max_episode_steps=3),
            adapter=GymAdapter(),
            seeds=[0, 1, 2],
            output_dir=out,
            code_sha="deterministic-test",
        )
        return manifest.run_id

    r1 = _run(tmp_path / "run1")
    r2 = _run(tmp_path / "run2")
    assert r1 == r2
