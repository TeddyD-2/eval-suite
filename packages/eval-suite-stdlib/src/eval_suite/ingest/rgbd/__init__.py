"""RGB-D capture → MJCF ingest path.

Counterpart to `eval_suite.ingest.splat` for the *other* common source
of real-world scans: a directory of RGB-D frames captured from a sensor
(RealSense, iPhone LiDAR, ZED). The fusion step is Open3D TSDF; the
output `visuals/scene.obj` + `collision_hull/scene_hull.obj` are the
same shape `splat_to_static_mesh` produces, so the existing
`compose_with_annotations` consumes them unmodified — and
`ParametricSplatTask(scene_dir=...)` runs against either source.

Heavy deps (`open3d`) live behind the `[rgbd]` extra and are imported
lazily inside `fuse_rgbd_to_mesh`. The CLI module is import-cheap so
`python -m eval_suite.ingest rgbd --help` works without extras.
"""
