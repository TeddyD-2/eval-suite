# Authoring an eval-suite plugin

This guide walks through publishing a `Task`, `Policy`, or `Adapter` as a separate pip package that the eval-suite CLI picks up automatically. After `pip install your-plugin`, the in-tree CLI sees your plugin without any modifications to eval-suite source.

A complete reference plugin lives at [`examples/external_plugin_demo/`](../examples/external_plugin_demo/). Read it alongside this guide.

## The 60-second version

1. Implement the relevant Protocol from [`eval_suite/contracts.py`](../packages/eval-suite-core/src/eval_suite/contracts.py) (Task, Policy, Adapter — see signatures below).
2. Declare your class as an entry-point in your package's `pyproject.toml`:

   ```toml
   [project.entry-points."eval_suite.tasks"]
   my_task = "my_pkg.tasks:MyTask"
   ```

3. From your tests, call the conformance battery:

   ```python
   from eval_suite.conformance import full_battery, verify_task
   from my_pkg import MyTask

   def test_my_task_conformance():
       verify_task(MyTask)
   ```

4. `pip install your-package` — eval-suite picks it up.

## Concepts

### The four contracts

eval-suite is built around four `typing.Protocol` definitions:

| Protocol | What it does | Required methods/properties |
|---|---|---|
| `Task` | Defines the variant grid for one benchmark task family. | `name`, `embodiment`, `n_cells`, `cell_id(i)`, `build_env(i)`, `max_episode_steps` |
| `Policy` | Maps observations to actions. | `name`, `checkpoint_id`, `reset(instruction)`, `step(obs) -> ActionLike` |
| `Adapter` | Drives one (Policy + Task) rollout and produces a `RolloutResult`. | `name`, `rollout(policy, task, cell, seed)` |
| `Manifest` | Reproducibility receipt. | `run_id`, `to_json()`, `from_json()`, `verify()` |

Implementations are duck-typed — a class doesn't have to inherit from the Protocol, it just needs the right shape.

### Optional hooks

Several capabilities are opt-in. eval-suite discovers them via `getattr` and falls back to defaults when absent:

- `Task.instruction_for(env)` — return a per-cell language instruction.
- `Task.extract_image(env, obs)` — return an RGB image for video capture.
- `Task.canonical_axis_map: dict[str, CanonicalDim]` — map cell axis names to the four canonical buckets (`"language"`, `"visuals"`, `"physics"`, `"embodiment"`) so the four-bucket profile chart works for your Task.
- `Task.ACTION_SPACE_HINT: Literal["eef_7dof", "joint_target", "custom"]` — coarse hint for the pre-flight compatibility check.
- `Adapter.can_drive(policy, task) -> CompatibilityReport` — pre-flight compatibility check. Conservative: `ok=True` means "no known reason to refuse," not a guarantee.

You don't have to implement these. Plain Protocol-satisfying classes work; the optional hooks let you exercise extra features.

### CONTRACT_VERSION

`eval_suite/contracts.py` declares `CONTRACT_VERSION = "1.0.0"`. Your plugin should depend on `eval-suite-core` (the substrate) in its `pyproject.toml` and treat the contract major version as the compatibility boundary. When a future MAJOR `CONTRACT_VERSION` bump lands, eval-suite-core will surface a deprecation warning for one MINOR version before removing the old surface.

Required methods on the four Protocols are the stability surface; optional hooks are not.

## Step-by-step

### 1. Implement your Task

```python
# my_pkg/tasks.py
from eval_suite._types import CanonicalDim, CellId

class MyKitchenTask:
    canonical_axis_map: dict[str, CanonicalDim] = {
        "room":   "visuals",
        "object": "language",
    }

    def __init__(self, *, max_episode_steps: int = 200) -> None:
        self._max_episode_steps = max_episode_steps

    @property
    def name(self) -> str: return "my_kitchen"
    @property
    def embodiment(self) -> str: return "kitchen_arm"
    @property
    def n_cells(self) -> int: return 3
    @property
    def max_episode_steps(self) -> int: return self._max_episode_steps

    def cell_id(self, cell: int) -> CellId:
        ...

    def build_env(self, cell: int):
        # return a gymnasium-shaped env (reset → (obs, info); step → (obs, reward, success, truncated, info))
        ...
```

Implementation notes:

- `cell_id(i)` must return a `CellId` whose `embodiment` matches `Task.embodiment` and `task` matches `Task.name`. The conformance kit checks this.
- `build_env(i)` must return something with `reset(seed=...)` and `step(action)` that follow the gymnasium 5-tuple convention. The Adapter wraps this; you don't have to use gymnasium directly.
- For non-7-DoF-EEF tasks (quadrupeds, bimanual, etc.), have your Task's underlying env accept `JointAction.vector` straight through. The Adapter dispatches; see `MujocoPlaygroundAdapter` for the joint-space pattern.

### 2. Declare entry-points

```toml
# pyproject.toml
[project.entry-points."eval_suite.tasks"]
my_kitchen = "my_pkg.tasks:MyKitchenTask"
```

The left side is the short name (`my_kitchen`); right is the import path. When two installed packages claim the same short name, the eval-suite CLI errors with `EvalSuiteAmbiguousPluginError` and lists qualified forms (`pkg-a:my_kitchen`, `pkg-b:my_kitchen`); users disambiguate by passing the qualified form.

### 3. Run the conformance battery

```python
# tests/test_conformance.py
from pathlib import Path
from eval_suite.adapters import GymAdapter
from eval_suite.conformance import full_battery, verify_task, verify_policy
from eval_suite.policies.mock import MockPolicy
from my_pkg.tasks import MyKitchenTask

def test_my_task_satisfies_protocol():
    verify_task(MyKitchenTask)

def test_my_kit_round_trip(tmp_path: Path):
    full_battery(MyKitchenTask, MockPolicy, GymAdapter, tmp_path,
                 trials_per_cell=2, cells_to_sweep=2)
```

The conformance kit catches the most common shape mistakes (wrong CellId fields, missing methods, non-determinism). It does *not* guarantee your Task is "correct" in the deeper sense — it only checks structural contract satisfaction. A real Task can pass the battery and still produce bad evaluations (broken success criteria, leaky rewards). The battery is the floor.

### 4. Install and run

```bash
pip install -e .                                  # your package
python -m eval_suite.cli list                     # see your plugin in the catalog
python -m eval_suite.cli sweep \
    --task my_kitchen \
    --policy mock \
    --adapter gym \
    --trials 10 \
    --output-dir results/run/
```

Each sweep writes three artifacts to `output-dir`:

- `manifest.json` — sealed reproducibility receipt (content-hashed).
- `trials.csv` — one row per rollout.
- `plugin_provenance.json` — sidecar recording which pip packages produced this run.

The sidecar will record your package name + version as the source of the `task` component. That's the "I know exactly which plugin this came from" property.

## Submitting to a portal

A portal is a small FastAPI service (built into eval-suite at `eval_suite/portal/`) that accepts signed manifests. Run your own:

```bash
pip install eval-suite-core[portal]
uvicorn eval_suite.portal.app:app --port 8000
```

Sign and submit:

```python
from eval_suite.manifest import Manifest
from eval_suite.signing import generate_keypair, to_hex
import httpx

manifest = Manifest.from_json(open("results/run/manifest.json").read())
priv, pub = generate_keypair()  # in real use, load your existing keypair
manifest.sign(priv, to_hex(pub), identity="me@example.com")
httpx.post("http://127.0.0.1:8000/submit", json={"manifest": manifest.to_json()})
```

The portal records your submission and, if another submitter has already submitted the same `run_id`, surfaces the cross-device corroboration via `GET /submissions?run_id=<the run_id>`.

## Trust model

What v0 guarantees:

- **Manifest integrity at submission.** The content hash + Ed25519 signature mean the submitter held the key and the inputs they declared are bit-for-bit consistent with the recorded numbers.
- **Submitter identity at submission.** When the portal enforces an allow-list, the submitter's claimed identity is overwritten with the allow-list canonical identity — spoofing an identity string the submitter doesn't have a key for fails closed.
- **Cross-device corroboration.** Two submitters submitting the same `run_id` is recorded as independent corroboration. Useful for catching forged numbers (a single submitter who lies has only their own claim; corroborated runs require independent submitters to lie identically).

What v0 does **not** guarantee:

- **Ledger completeness.** The portal's `ledger.jsonl` is append-only **by convention** (only the portal code appends; nothing else writes). It is *not* cryptographically tamper-evident — the portal operator could selectively rewrite or hide entries. Sigstore in v1 closes this hole.
- **Plugin-author identity.** A plugin is trusted because pip + PyPI trusts its author (same trust model as any third-party Python package). eval-suite does not run plugin code in a sandbox in v0; the v1 sandboxed-execution path is named in the EXTENSION roadmap.
- **Compatibility certification.** `Adapter.can_drive()` is a conservative check (`ok=True` means "no known reason to refuse," not "guaranteed to work"). The runtime `TypeError` in `GymAdapter._flatten_action` is the actual safety net.

For the strongest form of the cross-device claim, submitters should publish their manifests and their package versions (`plugin_provenance.json`) under their own GitHub release, so independent reviewers can verify the chain without trusting any portal.

## Reading more

- [`eval_suite/contracts.py`](../packages/eval-suite-core/src/eval_suite/contracts.py) — the Protocol definitions, with full docstrings.
- [`examples/external_plugin_demo/`](../examples/external_plugin_demo/) — complete reference plugin.
- [`eval_suite/conformance/__init__.py`](../packages/eval-suite-core/src/eval_suite/conformance/__init__.py) — the conformance kit source, including what each helper checks.
- [EXTENSION.md](../takehome/EXTENSION.md) — design doc; the plugin substrate is in the roadmap section.
