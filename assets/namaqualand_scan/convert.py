#!/usr/bin/env python3
"""v0 — composed USD→MJCF conversion for the Namaqualand Boulder 05 scan.

Three-step pipeline (see `EXTENSION.md §v0`):

  Step A.  usd2mjcf (LightwheelAI, Apache-2.0) parses `source.usdc` and
           emits `MJCF/source.xml` plus `MJCF/visuals/Boulder.obj` — the
           extracted visual mesh. This is usd2mjcf's bread-and-butter
           path: a static USD-as-single-link "robot body" with the geom
           processed via its TransformGraph.

  Step B.  `trimesh.simplify_quadric_decimation(face_count=3000)` on the
           visual OBJ, then `decim.convex_hull`. Produces a single
           low-poly convex hull (≈548 faces, 17 KB) used as the
           collision geom. Replaces usd2mjcf's `--generate_collision`
           coacd path, which on this scan produced 269 convex pieces —
           ~75× the disk + memory footprint.

  Step C.  Emit `MJCF/scene.xml`: a boulder-only scene that uses the
           Step-A visual mesh for rendering (`group=1`, `contype=0`,
           `conaffinity=0`) and the Step-B hull for collision
           (`group=0`). NamaqualandScanTask composes this with a Go1
           robot programmatically at sweep time via `mujoco.MjSpec`,
           so this file stays purely scene-side and remains loadable
           on its own for testing.

Determinism strategy:
  - `PYTHONHASHSEED=0` set in `main()`.
  - Step A: usd2mjcf is external; its output bytes are stable across
    reruns for a given usd2mjcf+usd-core version but not under our
    control. Recorded SHA256 + tool versions in convert_log.json.
  - Step B: `fast_simplification`'s QEM is seedless and deterministic;
    `scipy.spatial.ConvexHull` is deterministic. Tier-2 outputs are
    byte-stable.
  - Step C: `xml.etree.ElementTree.tostring` with sorted attributes
    (via `_sorted_serialize`); JSON output uses `sort_keys=True`.

Run with the usd2mjcf venv (Python 3.10):

    PYTHONPATH=~/usd2mjcf/src ~/usd2mjcf/.venv/bin/python convert.py

The script is idempotent: rerunning it overwrites the MJCF/ tree with
byte-identical output (modulo Step A's external determinism). Source
USD SHA256 is the load-bearing input check regardless.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

DECIM_TARGET_FACES = 3000
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_USD2MJCF_REPO = Path.home() / "usd2mjcf" / "src"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _sorted_serialize(elem: ET.Element) -> str:
    """Serialize an XML element with attributes sorted by key so the
    output is byte-stable across Python's dict-iteration variations.
    """
    for e in elem.iter():
        if e.attrib:
            e.attrib = dict(sorted(e.attrib.items()))
    return ET.tostring(elem, encoding="unicode")


def _run_usd2mjcf(source_usd: Path, output_dir: Path, repo: Path, py: Path) -> Path:
    """Step A. Returns the path to the produced visual OBJ."""
    test_script = repo / "test" / "usd2mjcf_test.py"
    if not test_script.exists():
        raise FileNotFoundError(f"usd2mjcf test script not found at {test_script}")
    env = {**os.environ, "PYTHONPATH": str(repo)}
    subprocess.run(
        [str(py), str(test_script), str(source_usd), "--output_path", str(output_dir)],
        check=True,
        env=env,
    )
    visual = output_dir / "MJCF" / "visuals" / "Boulder.obj"
    if not visual.exists():
        raise FileNotFoundError(f"usd2mjcf did not produce expected visual mesh at {visual}")
    return visual


def _decimate_and_hull(visual_obj: Path, hull_out: Path, target_faces: int) -> dict[str, int]:
    """Step B. Returns {'src_faces', 'src_verts', 'decim_faces', 'hull_faces', 'hull_verts'}."""
    import trimesh

    mesh = trimesh.load(visual_obj, force="mesh")
    src_faces, src_verts = int(len(mesh.faces)), int(len(mesh.vertices))
    decim = mesh.simplify_quadric_decimation(face_count=target_faces)
    decim_faces = int(len(decim.faces))
    hull = decim.convex_hull
    hull_faces, hull_verts = int(len(hull.faces)), int(len(hull.vertices))
    hull_out.parent.mkdir(parents=True, exist_ok=True)
    hull.export(hull_out)
    return {
        "src_faces": src_faces,
        "src_verts": src_verts,
        "decim_faces": decim_faces,
        "hull_faces": hull_faces,
        "hull_verts": hull_verts,
    }


def _emit_scene_xml(visual_rel: str, hull_rel: str, out: Path) -> None:
    """Step C. Boulder-only MJCF; Go1 is attached at sweep time."""
    root = ET.Element("mujoco", {"model": "namaqualand_boulder_05_scene"})
    ET.SubElement(root, "compiler", {"angle": "radian", "coordinate": "local"})
    ET.SubElement(root, "option", {"integrator": "RK4", "timestep": "0.01"})
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", {"name": "boulder_visual", "file": visual_rel})
    ET.SubElement(asset, "mesh", {"name": "boulder_collision", "file": hull_rel})
    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "light", {"pos": "0 0 5", "dir": "0 0 -1", "directional": "true"})
    body = ET.SubElement(wb, "body", {"name": "boulder", "pos": "0 0 0"})
    ET.SubElement(body, "geom", {
        "name": "boulder_vis",
        "mesh": "boulder_visual",
        "type": "mesh",
        "group": "1",
        "contype": "0",
        "conaffinity": "0",
    })
    ET.SubElement(body, "geom", {
        "name": "boulder_col",
        "mesh": "boulder_collision",
        "type": "mesh",
        "group": "0",
    })
    out.write_text('<?xml version="1.0" encoding="utf-8"?>\n' + _sorted_serialize(root) + "\n")


def _tool_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("trimesh", "usd-core", "coacd", "fast_simplification", "mujoco"):
        try:
            from importlib.metadata import version as _v
            versions[name] = _v(name)
        except Exception:
            versions[name] = "unknown"
    return versions


def main() -> int:
    os.environ["PYTHONHASHSEED"] = "0"

    parser = argparse.ArgumentParser(description="v0 composed USD→MJCF conversion.")
    parser.add_argument("--source-usd", type=Path, default=SCRIPT_DIR / "source.usdc")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--usd2mjcf-repo", type=Path, default=DEFAULT_USD2MJCF_REPO)
    parser.add_argument("--usd2mjcf-python", type=Path,
                        default=Path.home() / "usd2mjcf" / ".venv" / "bin" / "python")
    parser.add_argument("--decim-faces", type=int, default=DECIM_TARGET_FACES)
    args = parser.parse_args()

    src_usd: Path = args.source_usd.resolve()
    if not src_usd.is_file():
        raise FileNotFoundError(f"source USD missing: {src_usd}")
    output_dir: Path = args.output_dir.resolve()
    mjcf_dir = output_dir / "MJCF"
    mjcf_dir.mkdir(parents=True, exist_ok=True)
    hull_dir = mjcf_dir / "collision_hull"

    src_sha = sha256_file(src_usd)
    print(f"[convert] source USD SHA256: {src_sha}")

    print("[convert] Step A: usd2mjcf …")
    visual_obj = _run_usd2mjcf(src_usd, output_dir, args.usd2mjcf_repo, args.usd2mjcf_python)
    visual_sha = sha256_file(visual_obj)
    print(f"[convert] visual mesh SHA256: {visual_sha}  ({visual_obj.relative_to(output_dir)})")

    print(f"[convert] Step B: decimate→{args.decim_faces} faces, convex hull …")
    hull_out = hull_dir / "boulder_hull.obj"
    stats = _decimate_and_hull(visual_obj, hull_out, target_faces=args.decim_faces)
    hull_sha = sha256_file(hull_out)
    print(f"[convert] hull SHA256: {hull_sha}  (V={stats['hull_verts']}, F={stats['hull_faces']})")

    print("[convert] Step C: emit composed scene.xml …")
    visual_rel = visual_obj.relative_to(mjcf_dir).as_posix()
    hull_rel = hull_out.relative_to(mjcf_dir).as_posix()
    scene_xml = mjcf_dir / "scene.xml"
    _emit_scene_xml(visual_rel, hull_rel, scene_xml)
    scene_sha = sha256_file(scene_xml)
    print(f"[convert] scene.xml SHA256: {scene_sha}")

    print("[convert] verifying scene.xml loads in MuJoCo …")
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    print(f"[convert] load OK: nbody={m.nbody}, ngeom={m.ngeom}, nmesh={m.nmesh}")

    log: dict[str, Any] = {
        "converted_at_iso": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "tool_versions": _tool_versions(),
        "source": {"path": str(src_usd.relative_to(output_dir)), "sha256": src_sha,
                   "size_bytes": src_usd.stat().st_size},
        "outputs": {
            "visual_mesh": {"path": str(visual_obj.relative_to(output_dir)),
                            "sha256": visual_sha, "size_bytes": visual_obj.stat().st_size,
                            "faces": stats["src_faces"], "vertices": stats["src_verts"]},
            "collision_hull": {"path": str(hull_out.relative_to(output_dir)),
                               "sha256": hull_sha, "size_bytes": hull_out.stat().st_size,
                               "faces": stats["hull_faces"], "vertices": stats["hull_verts"]},
            "composed_mjcf": {"path": str(scene_xml.relative_to(output_dir)),
                              "sha256": scene_sha, "size_bytes": scene_xml.stat().st_size},
        },
        "decimation": {"target_faces": args.decim_faces, "achieved_faces": stats["decim_faces"]},
        "mujoco_load_check": {"nbody": int(m.nbody), "ngeom": int(m.ngeom), "nmesh": int(m.nmesh)},
    }
    log_path = output_dir / "convert_log.json"
    log_path.write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    print(f"[convert] log written: {log_path.relative_to(output_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
