"""RGB-D capture → MJCF ingest path.

**In plain words.** This is the ingest path for anyone with a depth
camera (Intel RealSense, iPhone LiDAR, ZED stereo). Point it at a
folder of paired color + depth + pose frames and it returns a
sim-ready 3D scene. No Gaussian-splat training required — useful
for users who don't have the GPU rig to train SuGaR / Nerfstudio
but do have a depth sensor.


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
