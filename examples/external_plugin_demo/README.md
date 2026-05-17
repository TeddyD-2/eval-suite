# eval-suite-plugin-demo

A reference external eval-suite plugin. Demonstrates the v0 plug-and-play substrate end-to-end: a `Task` and a `Policy` declared in a separate pip-installable package, picked up by the in-tree `eval-suite` CLI via `importlib.metadata` entry-points.

The Task (`HouseholdMockTask`) has kitchen/pantry/living-room cell labels and instructions like `"pick up the cup"`. The Policy (`IndustrialArmMockPolicy`) has a 7-DoF EEF output shape and an industrial-arm-themed name. **The labels are the point**, not the simulation: the env underneath is `MockEnv` so this runs in seconds on a laptop with no GPU. A real third-party plugin would build a real env in its `build_env` and the same contract (Protocol, registry, manifest, sidecar provenance) would carry through unchanged.

## Install

```bash
# from the eval-suite repo root, with .venv-ci activated
pip install -e examples/external_plugin_demo/
```

After this, the eval-suite CLI sees the new plugins:

```bash
python -m eval_suite.cli list
# # Tasks
#   google_robot_pick_coke_can  (eval-suite-stdlib@0.1.0)        -> ...
#   household_mock              (eval-suite-plugin-demo@0.1.0)   -> plugin_demo.household_task:HouseholdMockTask
#   ...
# # Policies
#   ...
#   industrial_arm_mock         (eval-suite-plugin-demo@0.1.0)   -> plugin_demo.industrial_policy:IndustrialArmMockPolicy
```

## Run a sweep with mix-and-match plugins

```bash
python -m eval_suite.cli sweep \
    --task household_mock \
    --policy industrial_arm_mock \
    --adapter gym \
    --trials 2 \
    --output-dir results/demo_run/
```

The Task came from this package (`eval-suite-plugin-demo`), the Policy came from this package, the Adapter came from `eval-suite`. After the sweep:

```bash
cat results/demo_run/plugin_provenance.json
# {
#   "contract_version": "1.0.0",
#   "target_run_id": "<sha256>",
#   "components": {
#     "task":    {"package_name": "eval-suite-plugin-demo", "package_version": "0.1.0", ...},
#     "policy":  {"package_name": "eval-suite-plugin-demo", "package_version": "0.1.0", ...},
#     "adapter": {"package_name": "eval-suite-stdlib",      "package_version": "0.1.0", ...}
#   },
#   "eval_suite_version": "0.1.0"
# }
```

That's the mix-and-match property the v0 substrate provides: anyone can publish a Task + Policy combination and the manifest provenance records exactly where each piece came from.

## Conformance test

```bash
pytest examples/external_plugin_demo/tests/test_conformance.py
```

This is what a real plugin author would commit to their own CI before publishing — it calls `eval_suite.conformance.full_battery` to confirm the Task / Policy / Adapter triple satisfies the v0 contract.

## What this demo deliberately doesn't do

- **No real simulation.** The env is `MockEnv`; rollouts always fail. A real Task wraps a real sim (SimplerEnv, MuJoCo Playground, RoboCasa, ...). See `eval_suite/tasks/simpler_env.py` or `eval_suite/tasks/unitree_go1.py` for examples of real wrappers.
- **No real model.** The Policy emits a constant tiny perturbation. A real Policy calls into a model checkpoint.
- **No submission to a portal.** That's a separate step (`POST /submit` to a portal running the v0 endpoints). The submitter signs the manifest with their Ed25519 key.

This demo focuses on the *substrate* — proving the mechanism that lets external plugins plug in and that their provenance is recorded. Real-sim wrappers for SimplerEnv, LIBERO, MuJoCo Playground, and a USD scan ingestion path already ship in `packages/eval-suite-stdlib/src/eval_suite/tasks/`.
