# eval-suite-stdlib

The in-tree reference plugins for [eval-suite-core](../eval-suite-core/). Ships:

- **Tasks:** `GoogleRobotPickCokeCan`, `WidowXSpoonOnTowel`, `LIBEROSpatial`, `UnitreeGo1Joystick`, `NamaqualandScanTask`, `ParametricSplatTask` (factories: `tnt_truck_splat_task_factory`, `mock_tnt_truck_splat_task_factory`), `MockTask`.
- **Policies:** `SimplerEnvPolicy` (Octo + RT-1 wrappers), `RandomLocomotionPolicy`, `MockPolicy`.
- **Adapters:** `GymAdapter` (drives SimplerEnv and LIBERO), `MujocoPlaygroundAdapter` (drives MJX / Unitree Go1).
- **Ingest pipelines:** `eval_suite.ingest.splat` — Gaussian-splat → MJCF converter behind the `[splat]` pip extra, with declarative scene-annotation schemas (`NamedRegion`, `SpawnPoint`, `ExtractedBody`, `SceneTransform`) and a registry of declarative success predicates (`Survived`, `RobotReachedRegion`, `MaintainedClearance`). Pairs with `ParametricSplatTask` so a non-developer adopter can configure a goal-driven evaluation on a captured scene without writing code. See [`src/eval_suite/ingest/splat/README.md`](src/eval_suite/ingest/splat/README.md) for install + run instructions.

Structurally identical to a third-party plugin — same `Protocol` satisfaction, same `[project.entry-points."eval_suite.{tasks,policies,adapters}"]` registration that any external package would use. The wedge: the in-tree plugins use the same plug-in path as external ones, so the substrate is exercised by code that travels the same loader/dispatch path third parties do.

Optional extras:

- `[splat]` — `trimesh`, `fast-simplification`, `scipy`, `manifold3d` for the Gaussian-splat ingest path. Heavy GPU tooling (SuGaR, Nerfstudio) is documented but not pip-pinned; invoked via subprocess by the convert pipeline.

A worked external-plugin example lives at `../../examples/external_plugin_demo/`.

See the repo-root `README.md` for the user-facing vision and quick-start, and `takehome/EXTENSION.md` for the design doc.
