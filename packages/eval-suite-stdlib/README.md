# eval-suite-stdlib

The in-tree reference plugins for [eval-suite-core](../eval-suite-core/). Ships:

- **Tasks:** `GoogleRobotPickCokeCan`, `WidowXSpoonOnTowel`, `LIBEROSpatial`, `UnitreeGo1Joystick`, `NamaqualandScanTask`, `MockTask`.
- **Policies:** `SimplerEnvPolicy` (Octo + RT-1 wrappers), `RandomLocomotionPolicy`, `MockPolicy`.
- **Adapters:** `GymAdapter` (drives SimplerEnv and LIBERO), `MujocoPlaygroundAdapter` (drives MJX / Unitree Go1).

Structurally identical to a third-party plugin — same `Protocol` satisfaction, same `[project.entry-points."eval_suite.{tasks,policies,adapters}"]` registration that any external package would use. The wedge: the in-tree plugins use the same plug-in path as external ones, so the substrate is exercised by code that travels the same loader/dispatch path third parties do.

A worked external-plugin example lives at `../../examples/external_plugin_demo/`.

See the repo-root `README.md` for the user-facing vision and quick-start, and `takehome/EXTENSION.md` for the design doc.
