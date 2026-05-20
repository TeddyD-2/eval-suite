"""Boolean mesh ops for extracting articulable bodies from a static
splat-derived scene mesh.

**In plain words.** Splats are great at capturing geometry but they
come out as one giant solid mesh. To let the robot interact with
parts of the scene (a movable cart, a drawer), the suite needs to
*carve those bodies out* of the static mesh so they become separate
articulated objects. This file is the surgeon — the math that
takes the big mesh and a list of regions, and returns the regions
as standalone bodies plus the residual scene.

The highest-implementation-risk piece of the
v1 splat path: splat-derived meshes are non-watertight surfaces, so
naive `trimesh` boolean ops fail on most inputs.

Strategy: pre-pass the mesh through a repair pipeline
(`fix_normals` → `fill_holes` → `manifold3d` round-trip) before any
boolean. `manifold3d` is the modern library that handles non-watertight
inputs better than `trimesh`'s defaults. If repair can't produce a
watertight mesh, the caller falls back to "anchor-mode" extraction
(replace the carved region with a synthetic primitive matching the
declared `bounds.shape` + `bounds.size`), which always succeeds and
loses geometric fidelity in exchange.

All heavy imports (`trimesh`, `manifold3d`, `numpy`) happen lazily
inside functions so this module is import-safe in environments without
the `[splat]` pip extra installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .annotation import NamedRegion

__all__ = [
    "EmptyExtractionError",
    "FullCoverExtractionError",
    "DegenerateResidualError",
    "MeshRepairReport",
    "ExtractionReport",
    "repair_mesh",
    "cut_region_from_mesh",
    "synthetic_anchor_mesh",
]


class EmptyExtractionError(ValueError):
    """Raised when the declared bounds enclose zero faces of the input mesh."""


class FullCoverExtractionError(ValueError):
    """Raised when the declared bounds enclose the entire input mesh
    (residual would be empty)."""


class DegenerateResidualError(ValueError):
    """Raised when the residual mesh has fewer than `min_residual_faces`
    faces after the boolean op (likely an unrecoverable boolean failure)."""


@dataclass(frozen=True)
class MeshRepairReport:
    n_holes_filled: int
    n_normals_fixed: int
    watertight_after_repair: bool
    used_manifold3d_fallback: bool


@dataclass(frozen=True)
class ExtractionReport:
    residual_faces: int
    extracted_faces: int
    fell_back_to_synthetic_anchor: bool
    repair: MeshRepairReport | None = None


def repair_mesh(mesh: Any) -> tuple[Any, MeshRepairReport]:
    """Attempt to make `mesh` watertight via `trimesh`'s built-ins first,
    then `manifold3d` if still non-watertight. Returns the repaired mesh
    and a report.

    Splat-derived meshes are reliably non-watertight (they're surfaces,
    not solids), so the report's `watertight_after_repair` flag is the
    diagnostic users should consult before relying on the boolean output.
    """
    import trimesh

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"repair_mesh expects a trimesh.Trimesh, got {type(mesh)!r}")

    # 1. Normals first — fill_holes uses them.
    n_normals_fixed = 0
    try:
        before = mesh.is_winding_consistent if hasattr(mesh, "is_winding_consistent") else None
        mesh.fix_normals()
        after = mesh.is_winding_consistent if hasattr(mesh, "is_winding_consistent") else None
        if before is False and after is True:
            n_normals_fixed = 1
    except Exception:
        pass

    # 2. Fill holes via trimesh's built-in.
    n_holes_filled = 0
    try:
        if bool(mesh.fill_holes()):
            n_holes_filled = 1
    except Exception:
        pass

    used_manifold = False
    if not bool(getattr(mesh, "is_volume", False)):
        # 3. manifold3d round-trip as last-resort repair.
        try:
            import manifold3d

            verts = mesh.vertices
            faces = mesh.faces
            mf = manifold3d.Manifold(
                manifold3d.Mesh(vert_properties=verts, tri_verts=faces)
            )
            # `mf.as_original()` round-trips through a manifold representation
            # that closes most micro-gaps.
            repaired = mf.to_mesh()
            mesh = trimesh.Trimesh(
                vertices=repaired.vert_properties[:, :3], faces=repaired.tri_verts
            )
            used_manifold = True
        except Exception:
            # manifold3d not available or repair failed; ship as-is.
            pass

    watertight = bool(getattr(mesh, "is_volume", False)) or bool(
        getattr(mesh, "is_watertight", False)
    )
    return mesh, MeshRepairReport(
        n_holes_filled=n_holes_filled,
        n_normals_fixed=n_normals_fixed,
        watertight_after_repair=watertight,
        used_manifold3d_fallback=used_manifold,
    )


def synthetic_anchor_mesh(bounds: NamedRegion) -> Any:
    """Build a primitive trimesh.Trimesh matching `bounds.shape` + `bounds.size`,
    positioned at `bounds.pos`. The "fallback when boolean ops fail"
    escape hatch.
    """
    import numpy as np
    import trimesh

    if bounds.shape == "box":
        sx, sy, sz = bounds.size
        mesh = trimesh.creation.box(extents=(2 * sx, 2 * sy, 2 * sz))
    elif bounds.shape == "sphere":
        (radius,) = bounds.size
        mesh = trimesh.creation.icosphere(radius=radius)
    elif bounds.shape == "cylinder":
        radius, half_h = bounds.size
        mesh = trimesh.creation.cylinder(radius=radius, height=2 * half_h)
    else:  # pragma: no cover — schema validation should prevent this
        raise ValueError(f"unsupported synthetic anchor shape {bounds.shape!r}")
    mesh.apply_translation(np.asarray(bounds.pos, dtype=float))
    return mesh


def _region_to_manifold_primitive(bounds: NamedRegion) -> Any:
    """Convert a NamedRegion into a manifold3d primitive solid. Used as
    the cut tool for boolean difference."""
    import manifold3d
    import numpy as np

    if bounds.shape == "box":
        sx, sy, sz = bounds.size
        m = manifold3d.Manifold.cube([2 * sx, 2 * sy, 2 * sz], center=True)
    elif bounds.shape == "sphere":
        (radius,) = bounds.size
        m = manifold3d.Manifold.sphere(radius=radius, circular_segments=24)
    elif bounds.shape == "cylinder":
        radius, half_h = bounds.size
        m = manifold3d.Manifold.cylinder(
            height=2 * half_h, radius_low=radius, radius_high=radius, circular_segments=24
        )
    else:  # pragma: no cover
        raise ValueError(f"unsupported region shape {bounds.shape!r}")
    # Translate to the region's world-frame center.
    return m.translate(np.asarray(bounds.pos, dtype=float))


def cut_region_from_mesh(
    mesh: Any,
    bounds: NamedRegion,
    *,
    min_residual_faces: int = 50,
    allow_synthetic_anchor_fallback: bool = True,
) -> tuple[Any, Any, ExtractionReport]:
    """Boolean-difference `bounds` out of `mesh`. Returns
    `(residual, extracted, report)`.

    Raises:
      EmptyExtractionError — when `bounds` enclose zero faces of `mesh`.
      FullCoverExtractionError — when `bounds` enclose the entire mesh.
      DegenerateResidualError — when the boolean produces a residual with
        fewer than `min_residual_faces` faces and synthetic-anchor
        fallback is disabled.

    Repair side-effects: the input mesh is run through `repair_mesh`
    first; the resulting watertight (or best-effort) mesh is what feeds
    the boolean. The repair report is attached to the ExtractionReport.

    If `allow_synthetic_anchor_fallback=True` and the boolean fails for
    any reason, the function returns the un-modified `mesh` as the
    residual and a `synthetic_anchor_mesh(bounds)` as the extracted body.
    The report's `fell_back_to_synthetic_anchor` flag records this.
    """
    import numpy as np
    import trimesh

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"cut_region_from_mesh expects a trimesh.Trimesh, got {type(mesh)!r}")

    # Pre-flight: how many faces are inside the bounds primitive?
    # Use a cheap AABB containment test as a sanity check; the boolean
    # itself does the precise work.
    aabb_min = np.asarray(bounds.pos, dtype=float) - np.asarray(
        _aabb_halfsize_for(bounds), dtype=float
    )
    aabb_max = np.asarray(bounds.pos, dtype=float) + np.asarray(
        _aabb_halfsize_for(bounds), dtype=float
    )
    centroids = mesh.triangles_center
    inside_aabb = np.all((centroids >= aabb_min) & (centroids <= aabb_max), axis=1)
    n_inside_aabb = int(inside_aabb.sum())
    if n_inside_aabb == 0:
        raise EmptyExtractionError(
            f"Region {bounds.name!r} contains zero mesh faces "
            f"(AABB centroid pre-check). Bounds: pos={bounds.pos}, "
            f"shape={bounds.shape}, size={bounds.size}."
        )
    if n_inside_aabb == int(len(mesh.faces)):
        raise FullCoverExtractionError(
            f"Region {bounds.name!r} encloses ALL {n_inside_aabb} mesh faces "
            "(residual would be empty). Tighten the bounds."
        )

    repaired, repair_report = repair_mesh(mesh)

    try:
        import manifold3d

        scene_manifold = manifold3d.Manifold(
            manifold3d.Mesh(
                vert_properties=repaired.vertices, tri_verts=repaired.faces
            )
        )
        tool = _region_to_manifold_primitive(bounds)
        residual_manifold = scene_manifold - tool
        extracted_manifold = scene_manifold & tool

        residual_m = residual_manifold.to_mesh()
        extracted_m = extracted_manifold.to_mesh()

        residual = trimesh.Trimesh(
            vertices=residual_m.vert_properties[:, :3], faces=residual_m.tri_verts
        )
        extracted = trimesh.Trimesh(
            vertices=extracted_m.vert_properties[:, :3], faces=extracted_m.tri_verts
        )

        if int(len(residual.faces)) < min_residual_faces:
            if not allow_synthetic_anchor_fallback:
                raise DegenerateResidualError(
                    f"Residual has only {len(residual.faces)} faces after boolean diff; "
                    f"min_residual_faces={min_residual_faces}. Either the mesh repair "
                    "failed or bounds are too aggressive."
                )
            # Fall through to synthetic-anchor path below.
        else:
            return residual, extracted, ExtractionReport(
                residual_faces=int(len(residual.faces)),
                extracted_faces=int(len(extracted.faces)),
                fell_back_to_synthetic_anchor=False,
                repair=repair_report,
            )
    except (EmptyExtractionError, FullCoverExtractionError, DegenerateResidualError):
        raise
    except Exception:
        if not allow_synthetic_anchor_fallback:
            raise

    # Fallback: keep the original mesh as residual, synthesize the body.
    anchor = synthetic_anchor_mesh(bounds)
    return repaired, anchor, ExtractionReport(
        residual_faces=int(len(repaired.faces)),
        extracted_faces=int(len(anchor.faces)),
        fell_back_to_synthetic_anchor=True,
        repair=repair_report,
    )


def _aabb_halfsize_for(bounds: NamedRegion) -> tuple[float, float, float]:
    if bounds.shape == "box":
        return (bounds.size[0], bounds.size[1], bounds.size[2])
    if bounds.shape == "sphere":
        r = bounds.size[0]
        return (r, r, r)
    if bounds.shape == "cylinder":
        r, half_h = bounds.size
        return (r, r, half_h)
    raise ValueError(f"unsupported shape {bounds.shape!r}")  # pragma: no cover
