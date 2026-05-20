"""Generic mesh utilities shared across asset-ingest paths.

**In plain words.** Helpers every ingest path needs: simplify a huge
mesh down to something the simulator can run at 60 FPS, compute a
convex hull for collision, hash a file deterministically. Same code
used regardless of whether the source was a splat, an RGB-D scan,
or an Objaverse asset.


Lifted from the patterns in `assets/namaqualand_scan/convert.py` and
generalized so non-Namaqualand ingest paths (splat, future RGB-D) can
reuse them without depending on Namaqualand's CLI script. The Namaqualand
path keeps its own copies deliberately — see `eval_suite.ingest`
docstring for the reasoning.

Heavy deps (`trimesh`, `fast_simplification`, `mujoco`) are imported
lazily inside functions so this module is import-safe in environments
without the `[splat]` or `[mesh]` extras installed.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

from eval_suite.hashing import sha256_file as _core_sha256_file

__all__ = [
    "sha256_file",
    "sorted_serialize",
    "decimate_and_hull",
    "emit_scene_xml",
    "tool_versions",
]


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA256 hex of `path`'s bytes. Re-export of `eval_suite.hashing.sha256_file`
    so ingest code paths can import everything they need from one module."""
    return _core_sha256_file(path, chunk_size=chunk_size)


def sorted_serialize(elem: ET.Element) -> str:
    """Serialize an XML element tree with attributes sorted by key, so the
    output is byte-stable across Python's dict-iteration variations.
    Mutates `elem` in place (rewrites the .attrib dict on every node).
    """
    for e in elem.iter():
        if e.attrib:
            e.attrib = dict(sorted(e.attrib.items()))
    return ET.tostring(elem, encoding="unicode")


def decimate_and_hull(
    visual_obj: Path, hull_out: Path, target_faces: int
) -> dict[str, int]:
    """Load a visual mesh OBJ, decimate to roughly `target_faces`, then
    compute a convex hull and write it to `hull_out`.

    Returns mesh stats: src_faces, src_verts, decim_faces, hull_faces,
    hull_verts. Mirrors the Namaqualand pipeline so the same downstream
    MJCF emission can be reused for splat scenes.
    """
    import trimesh

    loaded = trimesh.load(visual_obj, force="mesh")
    # `trimesh.load` is typed as `Geometry`; with `force="mesh"` it returns
    # a `Trimesh`. Cast so attribute access type-checks.
    mesh: Any = loaded
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


def emit_scene_xml(
    *,
    visual_rel: str,
    hull_rel: str,
    body_name: str,
    out: Path,
    model_name: str | None = None,
) -> None:
    """Write a minimal MJCF scene with one named body that carries a visual
    mesh (group=1, no collision) and a collision hull (group=0). Caller
    provides the relative paths to the two OBJ files.

    `body_name` is configurable so splat scenes don't all ship a body
    called "boulder". `model_name` defaults to `f"{body_name}_scene"`.
    """
    model = model_name or f"{body_name}_scene"
    root = ET.Element("mujoco", {"model": model})
    ET.SubElement(root, "compiler", {"angle": "radian", "coordinate": "local"})
    ET.SubElement(root, "option", {"integrator": "RK4", "timestep": "0.01"})
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", {"name": f"{body_name}_visual", "file": visual_rel})
    ET.SubElement(asset, "mesh", {"name": f"{body_name}_collision", "file": hull_rel})
    wb = ET.SubElement(root, "worldbody")
    ET.SubElement(wb, "light", {"pos": "0 0 5", "dir": "0 0 -1", "directional": "true"})
    body = ET.SubElement(wb, "body", {"name": body_name, "pos": "0 0 0"})
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{body_name}_vis",
            "mesh": f"{body_name}_visual",
            "type": "mesh",
            "group": "1",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{body_name}_col",
            "mesh": f"{body_name}_collision",
            "type": "mesh",
            "group": "0",
        },
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('<?xml version="1.0" encoding="utf-8"?>\n' + sorted_serialize(root) + "\n")


def tool_versions(extra_packages: tuple[str, ...] = ()) -> dict[str, Any]:
    """Pip-installed version of every mesh / splat tool we record in the
    convert log. Unknown / missing packages map to "unknown" rather than
    raising — provenance is best-effort.
    """
    versions: dict[str, Any] = {"python": sys.version.split()[0]}
    candidates = (
        "trimesh",
        "fast_simplification",
        "scipy",
        "manifold3d",
        "mujoco",
        "nerfstudio",
        "gsplat",
    ) + tuple(extra_packages)
    for name in candidates:
        try:
            versions[name] = _pkg_version(name)
        except PackageNotFoundError:
            versions[name] = "unknown"
        except Exception:
            versions[name] = "unknown"
    return versions
