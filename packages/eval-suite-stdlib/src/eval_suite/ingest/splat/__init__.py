"""Gaussian-splat → MJCF ingest pipeline.

End-to-end: a Gaussian-splat `.ply` checkpoint plus declarative
annotations become a composed MJCF scene the existing
`MujocoPlaygroundAdapter` can drive.

Pipeline stages:

  1. `convert.splat_to_static_mesh` — runs SuGaR (preferred) or
     Nerfstudio's poisson exporter as a subprocess, decimates the
     resulting mesh via `_mesh_utils.decimate_and_hull`, emits the
     residual static visual + collision-hull OBJs.

  2. `extract.cut_region_from_mesh` (optional) — uses `manifold3d`
     boolean ops to carve declared `ExtractedBody` regions out of the
     static mesh. Each extracted body becomes its own MJCF body with
     its own collision hull and (optionally) joint kinematics.

  3. `convert.compose_with_annotations` — applies the scene_transform,
     bakes named-region sites + extracted bodies, emits the final MJCF.

Heavy GPU tooling (SuGaR, Nerfstudio) is invoked via subprocess; commit
SHAs land in `convert_log.json` next to the converted MJCF. Pure-Python
deps (`trimesh`, `manifold3d`) live behind the `[splat]` pip extra and
are imported lazily inside functions so this package is import-safe
without them.

The on-disk artifacts of one conversion bind into the eval-suite
substrate's content-addressed run_id via two paths:

  - `AssetProvenance.assets[]` — entries for source `.ply`, converted
    OBJs, composed MJCF, `scene_metadata.json`, `scene_extractions.json`.
    Their SHA256s flow into the sidecar's `target_run_id` linkage.

  - `Manifest.success_criterion` — the declarative predicate dict
    (e.g., `{"kind": "robot_reached_region", "params": {...}}`) is
    hashed into the manifest at schema 0.3.0. Changing the predicate
    on the same scene produces a distinct run_id.

The factory engineer's user story: capture warehouse with phone →
SuGaR → mesh → write `scene_metadata.json` (regions + spawns +
scene_transform) and `scene_extractions.json` (bodies to carve out) →
pick a predicate from `eval_suite.tasks._success_predicates` →
`python -m eval_suite.ingest.splat ingest ...` → `run_sweep(...)`.
"""
