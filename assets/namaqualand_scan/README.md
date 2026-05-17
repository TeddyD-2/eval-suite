# v0 — Namaqualand Boulder 05 scan

This directory holds the v0 real-world-scan demo asset and the
deterministic conversion script that turns it into MJCF. See
`takehome/EXTENSION.md §2` (asset interchange) for the wedge
framing and `eval_suite/tasks/usd_scan.py` for the Task that consumes
the output.

## Source

| Field | Value |
|---|---|
| Asset slug | `namaqualand_boulder_05` |
| Display name | Namaqualand Boulder 05 |
| Origin (Poly Haven page) | https://polyhaven.com/a/namaqualand_boulder_05 |
| Download URL (USDC, 8k) | https://dl.polyhaven.org/file/ph-assets/Models/usd/8k/namaqualand_boulder_05/namaqualand_boulder_05_8k.usdc |
| License | CC0-1.0 (public domain) |
| Authors / reviewers | James Ray Cock, Rico Cilliers (Poly Haven) |
| Capture region | Karoo desert, South Africa |
| Capture type | Photogrammetric 3D scan |
| Dimensions | 1.36 m × 0.75 m × 0.54 m (boulder scale, Go1-relative) |
| Native polycount | 117,622 |
| Downloaded | 2026-05-16 |

The published Poly Haven MD5 (`bd25f70531f9bf4e80fd399accc64f87`) matches
the downloaded file byte-for-byte. The on-disk SHA256 is the load-bearing
input check (used by `asset_provenance.json`'s `verify()`):

```
source.usdc  SHA256  85dc3dfeda7484928c3ff273a2c5887b7e6a78313449bee53e2cbb50719a7239
```

## Conversion pipeline

Composed three-step pipeline (see `convert.py` and the EXTENSION.md
§2 framing). The intermediate visual mesh comes from `usd2mjcf`; the
collision geom is produced by `trimesh` decimation + convex hull because
`usd2mjcf --generate_collision` (which delegates to `coacd`) produced
**269** convex pieces on this single boulder.

| Step | Tool | Input | Output |
|---|---|---|---|
| A | `usd2mjcf` (LightwheelAI, Apache-2.0) | `source.usdc` | `MJCF/visuals/Boulder.obj` (visual mesh, 89,618 faces) |
| B | `trimesh.simplify_quadric_decimation(face_count=3000)` → `convex_hull` | Step-A OBJ | `MJCF/collision_hull/boulder_hull.obj` (548 faces) |
| C | Local XML emit | A + B | `MJCF/scene.xml` (boulder-only; Go1 attached at sweep time via `mujoco.MjSpec`) |

Recorded SHA256s of the conversion artifacts (these are what
`asset_provenance.json` binds to `manifest.run_id`):

```
MJCF/visuals/Boulder.obj             5d7329a8d290489baf3e6327f30d4b88a2c2075598b19adfdd8753f5711d6c70
MJCF/collision_hull/boulder_hull.obj 78566c84c79228f4da8c9473cb6bbe98a8dee448bd04d1ca727d62adb0102441
MJCF/scene.xml                       37f417e04b1cf96681582a65f05e2e3c097efdc96b4c64f0fffc32e2d28dede7
```

A full machine-readable log (timestamps, tool versions, intermediate
face/vertex counts, MuJoCo load-check) is written to `convert_log.json`
on every run.

## How to regenerate

The conversion runs in a sibling Python 3.10 venv (see CLAUDE.md
"Three venvs" table — `~/usd2mjcf/.venv`):

```bash
# One-time setup (Python 3.10 + usd2mjcf clone + pip install):
git clone --depth=1 https://github.com/LightwheelAI/usd2mjcf.git ~/usd2mjcf/src
python3.10 -m venv --without-pip ~/usd2mjcf/.venv
curl -fsSL https://bootstrap.pypa.io/get-pip.py | ~/usd2mjcf/.venv/bin/python
~/usd2mjcf/.venv/bin/pip install -r ~/usd2mjcf/src/requirements.txt \
    fast_simplification mujoco

# Run conversion (idempotent; rewrites MJCF/ + convert_log.json):
cd /path/to/eval-suite/assets/namaqualand_scan
~/usd2mjcf/.venv/bin/python convert.py
```

The script is deterministic: Step B (`fast_simplification` QEM +
`scipy.spatial.ConvexHull`) is seedless; Step C pins `PYTHONHASHSEED=0`
and writes XML with sorted attributes + JSON with `sort_keys=True`.
Step A (`usd2mjcf` external) is best-effort deterministic — empirically
stable across reruns of the same `usd2mjcf` + `usd-core` version on
this scan. Confirmed by rerunning the full pipeline and observing
byte-identical SHA256s for all three outputs.

## Why a custom collision step (and not `usd2mjcf --generate_collision`)

usd2mjcf's collision path runs `coacd` (Approximate Convex Decomposition).
On Namaqualand Boulder 05, coacd produces **269 convex pieces** —
1.2 MB on disk, `nmesh=270`, hundreds of contact pairs against a 12-leg
Go1. The decimated-convex-hull alternative produces **one** convex
piece (548 faces, 17 KB), which is ~75× smaller on disk, gives `nmesh=2`,
and runs ~4× faster in `mj_step` per 100-step measurement (1.6 ms vs
6 ms with no robot in scene).

The composed pipeline keeps `usd2mjcf` as the canonical USD→visual-mesh
step (where it does real work — parsing USD, applying the
TransformGraph, extracting the OBJ) and replaces only its `coacd` step.
Both tools are exercised in `convert.py`; both are recorded in
`asset_provenance.json`.

## Composition at sweep time

The MJCF in this directory is **boulder-only**. Go1 is composed in at
sweep time by `_ScanSceneCompatEnv` in `eval_suite/tasks/usd_scan.py`
via:

```python
go1_spec = mujoco.MjSpec.from_file(<menagerie_go1_path>)
boulder_spec = mujoco.MjSpec.from_file("MJCF/scene.xml")
anchor = go1_spec.worldbody.add_frame(name="boulder_anchor", pos=[1.0, 0, 0])
go1_spec.attach(boulder_spec, prefix="scan_", frame=anchor)
model = go1_spec.compile()
```

`<menagerie_go1_path>` resolves at runtime to
`mujoco_playground/external_deps/mujoco_menagerie/unitree_go1/go1.xml`
shipped with `mujoco_playground`. This means the boulder asset stays
robot-agnostic — pairing it with a different embodiment is a one-line
edit at the Task level, not a re-export.

The composed model: `nbody=15, ngeom=58, nmesh=7, nu=12, nq=19`. A
30-step `mj_step` rollout from rest with zero control: Go1 stays
upright at `trunk_z ≈ 0.43` m (above the `FALL_HEIGHT_THRESHOLD = 0.15`
m fall criterion).

## Files

```
assets/namaqualand_scan/
├── README.md                              # this file
├── convert.py                             # the three-step pipeline
├── convert_log.json                       # per-run trace (tool versions, SHA256s, timing)
├── source.usdc                            # downloaded Poly Haven asset
└── MJCF/
    ├── scene.xml                          # composed scene used by NamaqualandScanTask
    ├── source.xml                         # usd2mjcf's raw Step-A output (visual-only)
    ├── visuals/
    │   ├── Boulder.obj                    # Step-A visual mesh
    │   └── materials.mtl
    └── collision_hull/
        └── boulder_hull.obj               # Step-B decimated convex hull
```
