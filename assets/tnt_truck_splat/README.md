# `tnt_truck_splat` — v1 splat trial-generation demo asset

A converted Gaussian-splat of the Tanks-and-Temples "Truck" scene,
composed with declarative annotations into a Go1-driveable MJCF for the
eval-suite substrate.

**License:** Tanks-and-Temples imagery is CC-BY 4.0 (Knapitsch et al.,
SIGGRAPH 2017). Commercial reuse permitted — unblocks
EXTENSION.md §6 partner-startup pilots.

**Purpose:** v1's deliverable per EXTENSION.md §7 is "an initial 3D-splat
trial-generation prototype, single environment, parametric over lighting
+ camera pose, end-to-end through the same Task Protocol." This
directory is that prototype's demo asset. The 9-cell ParametricSplatTask
(3 lighting × 3 camera) evaluates "did Go1 reach behind the truck under
variant lighting/camera conditions" — exercising the annotation
substrate (`NamedRegion`, `SpawnPoint`, `ExtractedBody`, `scene_transform`)
end-to-end on real-world-captured geometry.

## Contents

| File                       | Role                                                                                                                                       |
|----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `README.md`                | This file.                                                                                                                                 |
| `download.sh`              | Fetches the source `.ply` from the official Tanks-and-Temples release URL; verifies SHA256 against `source_manifest.json`.                |
| `source_manifest.json`     | URL + expected SHA256 of the source splat checkpoint. Committed; small (a few KB).                                                          |
| `scene_metadata.json`      | Declarative scene metadata: `scene_transform`, `named_regions`, `spawn_points`. Hashed into the eval-suite manifest via `AssetProvenance`. |
| `scene_extractions.json`   | Declarative bodies to carve out of the static scene: the truck itself. Hashed into the manifest the same way.                              |
| `MJCF/scene.xml`           | Composed MJCF produced by `python -m eval_suite.ingest.splat ingest ...`. **Not committed** — regenerated from source on conversion.       |
| `visuals/*.obj`            | Decimated static + extracted-body visual meshes. **Not committed** — regenerated.                                                          |
| `collision_hull/*.obj`     | Convex hulls for collision. **Not committed** — regenerated.                                                                                |
| `convert_log.json`         | Per-phase provenance from the ingest pipeline (tool versions, SHAs, mesh-repair report). **Not committed** — regenerated.                  |
| `asset_provenance.json`    | Sidecar binding the converted artifacts to the manifest's `target_run_id`. **Not committed** — written by `run_sweep`'s output dir.        |

## Generating the converted artifacts

Heavy GPU prerequisites: install SuGaR (preferred) or Nerfstudio + an
ML CUDA stack. See `packages/eval-suite-stdlib/src/eval_suite/ingest/splat/README.md`.

```
bash download.sh                              # fetches source .ply; verifies SHA
python -m eval_suite.ingest.splat ingest \
  assets/tnt_truck_splat/source.ply \
  --scene-metadata     assets/tnt_truck_splat/scene_metadata.json \
  --scene-extractions  assets/tnt_truck_splat/scene_extractions.json \
  --output-dir         assets/tnt_truck_splat/ \
  --body-name          splat_scene
```

Produces `MJCF/scene.xml`, `visuals/*.obj`, `collision_hull/*.obj`, and
populates `convert_log.json`.

## Running the v1 demo sweep

After conversion:

```python
from eval_suite.tasks.parametric_splat import tnt_truck_splat_task_factory
from eval_suite.policies.random_locomotion import RandomLocomotionPolicy
from eval_suite.adapters.mujoco_playground import MujocoPlaygroundAdapter
from eval_suite.sweep import run_sweep

m = run_sweep(
    task=tnt_truck_splat_task_factory(),
    policy=RandomLocomotionPolicy(),
    adapter=MujocoPlaygroundAdapter(),
    seeds=[0, 1, 2],
    output_dir="/tmp/tnt_truck_sweep",
    code_sha="<HEAD>",
)
assert m.verify()
assert m.schema_version == "0.3.0"
assert m.success_criterion == {"kind": "robot_reached_region",
                               "params": {"region_name": "behind_truck", "tolerance": 0.5}}
```

Expect: 9 cells × 3 seeds = 27 trial rows; sealed manifest with
`verify()` True; mp4s + npz under `/tmp/tnt_truck_sweep/`. The
factory-engineer user story works end-to-end.

## Substrate properties exercised

- `scene_metadata.json` SHA256 → bound into `Manifest.run_id` via
  `AssetProvenance.assets[role="scene_metadata"]`. Editing the goal
  zone by one byte → different run_id.
- `scene_extractions.json` SHA256 → same binding (role
  `"scene_extractions"`).
- The success predicate dict (`{"kind": "robot_reached_region",
  "params": {...}}`) → bound directly into `Manifest.success_criterion`
  at schema 0.3.0. Swapping the predicate produces a distinct run_id.
- `canonical_axis_map = {"lighting": "visuals", "camera": "visuals"}` →
  bound at schema 0.2.0+; lets the analysis notebook project both axes
  onto the canonical visuals dimension.

The composed MJCF SHA256 is recorded in `convert_log.json` and bound
into `AssetProvenance` once the conversion runs. The combined effect:
the manifest's `run_id` is uniquely determined by the (splat source,
scene_metadata, scene_extractions, predicate, axes, seeds, code SHA,
hardware) tuple. That's the eval-suite's reproducibility contract,
preserved unchanged for splat scenes.
