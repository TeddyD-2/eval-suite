# `eval_suite.ingest.splat`

Gaussian-splat → MJCF ingest path for the eval-suite substrate.

## What this does

End-to-end pipeline: take a Gaussian-splat `.ply` checkpoint of a
real-world environment (a factory floor, warehouse, kitchen) plus
declarative annotations, and produce a composed MJCF the existing
`MujocoPlaygroundAdapter` can drive. The composed MJCF carries:

- The static residual scene mesh (decimated + convex-hulled for
  collision).
- Optional `ExtractedBody` entries: parts of the scene boolean-cut
  out and re-emitted as articulable bodies with their own joints +
  masses.
- Invisible MJCF sites for every `NamedRegion` declared in the
  scene metadata, so success predicates can reference regions by
  name (`RobotReachedRegion("loading_dock")`,
  `MaintainedClearance("conveyor")`).
- Invisible MJCF sites for every `SpawnPoint`, so the task can
  position the robot at a named pose.

The scene's `scene_metadata.json` and `scene_extractions.json` files
are content-hashed into the manifest's `run_id` via
`AssetProvenance.assets[]` entries. The success predicate dict is
hashed into the manifest directly (schema 0.3.0). Two sweeps of the
same scene with different goals (`reach_region` vs
`maintain_clearance_from`) therefore produce distinct `run_id`s — the
factory-engineer wedge.

## Install

The pure-Python deps live behind the `[splat]` extra in
`eval-suite-stdlib`:

```
pip install 'eval-suite-stdlib[splat]'
```

This pulls in `trimesh`, `fast-simplification`, `scipy`, and
`manifold3d` (the modern boolean-mesh-ops library — much better than
`trimesh`'s default for non-watertight splat output).

The heavy GPU tooling — **SuGaR** for splat → surface-mesh extraction,
or **Nerfstudio** with its `ns-export poisson` fallback — must be
installed separately. The CUDA pins for these change frequently, so
they're documented here rather than hard-pinned in `pyproject.toml`.
Pick one:

```
# Preferred: SuGaR (better surface quality on splat input)
git clone https://github.com/Anttwo/SuGaR && cd SuGaR && pip install -e .

# Fallback: Nerfstudio's poisson exporter
pip install nerfstudio
```

The convert pipeline calls these via subprocess. The chosen tool's
commit SHA + arguments + wall-clock are recorded in `convert_log.json`
for provenance.

## Run

One command:

```
python -m eval_suite.ingest.splat ingest \
  path/to/source.ply \
  --scene-metadata     path/to/scene_metadata.json \
  --scene-extractions  path/to/scene_extractions.json \
  --output-dir         path/to/assets/my_scene/
```

Produces:

- `MJCF/scene.xml` — composed MJCF
- `visuals/splat_scene.obj` + per-extracted-body `extracted_<name>.obj`
- `collision_hull/*.obj` (convex hulls)
- `convert_log.json` — per-phase provenance (tool versions, SHAs,
  scene_transform, mesh-repair report)

Then register the scene as a task entry-point (see
`eval_suite.tasks.parametric_splat`) and use it like any other task:

```
from eval_suite.tasks.parametric_splat import ParametricSplatTask, ParametricSplatTaskConfig
from eval_suite.tasks._success_predicates import RobotReachedRegion
from eval_suite.tasks._parametric_variants import STANDARD_LIGHTING_3, STANDARD_CAMERAS_3

task = ParametricSplatTask(ParametricSplatTaskConfig(
    scene_dir=Path("path/to/assets/my_scene/"),
    embodiment="unitree_go1",
    max_episode_steps=200,
    axes={"lighting": STANDARD_LIGHTING_3, "camera": STANDARD_CAMERAS_3},
    success_predicate=RobotReachedRegion(region_name="loading_dock", tolerance=0.5),
    canonical_axis_map={"lighting": "visuals", "camera": "visuals"},
))
```

## Risks

The pipeline's highest-implementation-risk step is the boolean
mesh-extraction (`extract.cut_region_from_mesh`). Splat-derived
meshes are non-watertight surfaces; naive booleans fail on them. The
`repair_mesh` pipeline (fix_normals → fill_holes → `manifold3d`
round-trip) handles most cases. If repair can't produce a watertight
mesh, the function falls back to **anchor-mode** extraction (replace
the carved region with a synthetic primitive matching the declared
bounds), with a logged warning. This always succeeds and loses
geometric fidelity in exchange.

For your scene to actually use real boolean-cut bodies (not synthetic
anchors), check `convert_log.json`'s `mesh_repair.watertight_after_repair`
flag. If False, either:

- Improve the source splat (more training iterations, better camera
  coverage) so SuGaR produces a more closed surface.
- Use `joint_type: "fixed"` for the extracted body so collision
  fidelity matters less than articulation kinematics.
- Accept the synthetic-anchor fallback for that body.

## Coordinate frame

SuGaR's output mesh has arbitrary up-axis and arbitrary scale. The
mandatory `scene_transform` block in `scene_metadata.json` maps mesh
frame onto MuJoCo world frame (Z-up, meters). Without it, your robot
spawns 47 meters above the floor.

The convert pipeline auto-detects (PCA-based up-axis + bbox-based
scale) and writes the auto-detected values into `convert_log.json`.
Review them, override in `scene_metadata.json` if wrong. Auto-detect
on splats is heuristic, not reliable — for production captures, set
the transform explicitly.
