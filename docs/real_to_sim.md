# Real-to-sim asset pipeline

The eval-suite turns a real-world capture into a simulator scene the
existing `ParametricSplatTask` consumes — same MJCF artifact shape,
zero new Task code, zero contract changes. Three sources are
supported, dispatched through one CLI:

```bash
python -m eval_suite.ingest splat     <splat.ply>      ...    # v1 — Gaussian splat
python -m eval_suite.ingest rgbd      <frames_dir>     ...    # v2 — RGB-D capture
python -m eval_suite.ingest objaverse <uid>            ...    # v2 — public asset library
```

All three emit the same artifact triple:

- `visuals/scene.obj` — high-res visual mesh
- `collision_hull/scene_hull.obj` — convex-hull collision proxy
- `MJCF/scene.xml` — composed MJCF with named regions + spawn points + extracted bodies
- `convert_log.json` — per-stage provenance (tool versions, input SHAs, wall clock)

Plus `scene_metadata.json` and (optionally) `scene_extractions.json`
written by the operator. Once the four files are on disk the sweep
runs:

```bash
python -m eval_suite.cli sweep \
    --task my_scene \              # registered via entry_points → tnt_truck_splat_task_factory(scene_dir=...)
    --task-arg scene_dir=<output-dir> \
    --policy lerobot \
    --policy-arg repo_id=lerobot/smolvla-base \
    --adapter mujoco_playground \
    --trials 20 \
    --output-dir results/sweep_X
```

## Source 1: Gaussian splat (v1, already shipping)

See `packages/eval-suite-stdlib/src/eval_suite/ingest/splat/README.md`
for the SuGaR + `manifold3d` install and the `--target-faces`,
`--scene-extractions`, `--body-name` knobs. Original v1 demo asset is
the Tanks-and-Temples Truck.

## Source 2: RGB-D capture (new in v2)

For sensors that produce paired color + depth + pose streams (Intel
RealSense, iPhone LiDAR via Polycam / Record3D, ZED stereo). TSDF
fusion via Open3D's `ScalableTSDFVolume`.

Install: `pip install 'eval-suite-stdlib[rgbd]'`.

Input layout under `<frames_dir>`:

```
color/        0000.png, 0001.png, ...  (HxWx3 uint8 RGB)
depth/        0000.png, 0001.png, ...  (HxW uint16 mm depth; or float32 m)
poses.txt     # one 4x4 row-major T_world_cam per non-blank line
intrinsics.json
```

`intrinsics.json` schema:

```json
{
  "fx": 615.7,
  "fy": 615.7,
  "cx": 326.5,
  "cy": 240.0,
  "width": 640,
  "height": 480,
  "depth_scale": 1000.0
}
```

Run:

```bash
python -m eval_suite.ingest rgbd \
    /path/to/frames \
    --scene-metadata scene_metadata.json \
    --output-dir scenes/my_room \
    --voxel-length 0.01 --sdf-trunc 0.04 --depth-max 3.0
```

The fusion records every input frame's SHA256 + the fusion parameters
into `convert_log.json` so the run is reproducible bit-for-bit given
the same frames and intrinsics.

## Source 3: Objaverse-XL (new in v2)

For procedurally composing public CC-licensed objects into a sim
scene. The fetcher enforces a license allowlist *before* download —
the default (`CC-BY`, `CC-BY-4.0`, `CC0`, `CC0-1.0`) is the
commercial-friendly subset; CC-BY-NC assets are refused with a clear
error unless you opt in via `--license-allowlist`.

Install: `pip install 'eval-suite-stdlib[objaverse]'`.

```bash
python -m eval_suite.ingest objaverse \
    1769ebd99c004a90b8e3f8b9b9c89d6a \
    --scene-metadata scene_metadata.json \
    --output-dir scenes/my_object \
    --license-allowlist CC-BY,CC-BY-4.0,CC0
```

The license + Objaverse uid + every metadata field land in
`convert_log.json`'s `objaverse_fetch` payload — the manifest's
`asset_provenance.json` sidecar then binds the converted mesh's
SHA256 to the run_id.

## End-to-end: RGB-D → MJCF → LeRobot sweep

Combine Phase 1 (LeRobot interop) + Phase 2 (RGB-D ingest):

```bash
# 1. Capture: a RealSense or iPhone scan of your tabletop scene
#    (frames under ./capture/)

# 2. Author scene_metadata.json declaring named regions + spawn points
#    (see packages/eval-suite-stdlib/src/eval_suite/ingest/splat/README.md
#    for the schema — same shape across all three ingest paths).

# 3. Ingest:
python -m eval_suite.ingest rgbd ./capture \
    --scene-metadata scene_metadata.json \
    --output-dir scenes/tabletop

# 4. Sweep a LeRobot policy against it:
python -m eval_suite.cli sweep \
    --task my_scene \
    --task-arg scene_dir=scenes/tabletop \
    --policy lerobot \
    --policy-arg repo_id=lerobot/smolvla-base \
    --adapter mujoco_playground \
    --trials 10 \
    --output-dir results/sweep_tabletop

# 5. Result: results/sweep_tabletop/manifest.json carries
#    asset_provenance.json (RGB-D source SHA, fusion params, mesh SHAs),
#    convert_log.json (per-frame SHAs), and the standard per-cell
#    Wilson CIs. ParametricSplatTask did not need any changes — the
#    RGB-D ingester just produced the same artifact shape it already
#    consumes.
```

This is what "the substrate is right" means in practice: the new
ingest source plugs into an unchanged downstream.
