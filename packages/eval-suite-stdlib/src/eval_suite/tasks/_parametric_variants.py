"""Lighting and camera variants for ParametricSplatTask cells.

**In plain words.** The catalog of "how can I vary the scene" knobs
for a splat-derived task — three lighting setups and three camera
angles, each tagged with the distribution shift it's meant to
probe. A user can pick any subset (or write their own) to define
the cell grid for their own scanned scene.


Each variant has a named hypothesis (per EXTENSION.md §3's
"silent re-bucketing is auditable" rule — every axis level should map
to a deliberate distribution shift, not a magic number).

v1 ships three lighting + three camera variants, mirroring EXTENSION.md
§7's "parametric over lighting + camera pose" commitment. Users can
construct their own variant lists and pass them to
ParametricSplatTaskConfig.axes for non-default cell grids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "LightingVariant",
    "CameraVariant",
    "STANDARD_LIGHTING_3",
    "STANDARD_CAMERAS_3",
]


@dataclass(frozen=True)
class LightingVariant:
    """One lighting cell. Applied to the composed MJCF pre-compile via
    MjSpec mutation of the scene's light element."""

    name: str
    type: Literal["directional", "spot", "ambient_plus_directional"]
    pos: tuple[float, float, float]
    dir: tuple[float, float, float]
    diffuse: tuple[float, float, float]
    ambient: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class CameraVariant:
    """One camera cell. `mount="world"` attaches the camera to the world
    frame at `pos`. `mount="robot_trunk"` attaches it to the Go1 trunk
    body so it follows the robot. `lookat` is used at compile time to
    derive a quaternion via `_lookat_quat` (reused from usd_scan.py)."""

    name: str
    mount: Literal["world", "robot_trunk"]
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float] | None = None
    quat: tuple[float, float, float, float] | None = None


# v1 demo defaults — three lighting, three camera variants.
#
# Each variant has a hypothesis spelled out in the comment. If the
# variant can't be defended on hypothesis grounds, it shouldn't be in
# the cell grid (it would just add compute without information).


STANDARD_LIGHTING_3: list[LightingVariant] = [
    # noon_overhead — baseline matching the lighting most splat captures
    # are taken under (midday). Best-case visual recognition.
    LightingVariant(
        name="noon_overhead",
        type="directional",
        pos=(0.0, 0.0, 5.0),
        dir=(0.0, 0.0, -1.0),
        diffuse=(1.0, 1.0, 1.0),
    ),
    # afternoon_oblique — tests whether the policy overfits to the
    # specific lighting baked into the splat's appearance. Warm color
    # temperature exercises any chromatic-prior on the image side.
    LightingVariant(
        name="afternoon_oblique",
        type="directional",
        pos=(3.0, -3.0, 2.0),
        dir=(-0.5, 0.5, -0.7),
        diffuse=(1.0, 0.85, 0.65),
    ),
    # overcast_ambient — failure mode for image-conditioned policies that
    # rely on shadow cues to localize. Low directional + high ambient
    # eliminates the long shadows that mid-day captures depend on.
    LightingVariant(
        name="overcast_ambient",
        type="ambient_plus_directional",
        pos=(0.0, 0.0, 5.0),
        dir=(0.0, 0.0, -1.0),
        diffuse=(0.4, 0.4, 0.45),
        ambient=(0.5, 0.5, 0.55),
    ),
]


STANDARD_CAMERAS_3: list[CameraVariant] = [
    # third_person_default — matches the Namaqualand baseline
    # (usd_scan.py:61's SCENE_CAMERA_POS). The "what a benchmark video
    # looks like" pose; expected baseline for most policies.
    CameraVariant(
        name="third_person_default",
        mount="world",
        pos=(-1.5, -2.0, 1.0),
        lookat=(0.5, 0.0, 0.3),
    ),
    # low_egocentric — mounted on the robot trunk, looks forward.
    # Tests sim→deploy generalization from external training-time view
    # to on-board deployment-time view, which is what real robots use.
    CameraVariant(
        name="low_egocentric",
        mount="robot_trunk",
        pos=(0.3, 0.0, 0.05),
        lookat=(1.0, 0.0, 0.0),
    ),
    # high_overview — 5m above, looks straight down. Stress-tests the
    # image-extraction path at unusual viewpoints and ensures the
    # rendered cell grid varies visibly. Most image-conditioned policies
    # collapse on this — that's the point.
    CameraVariant(
        name="high_overview",
        mount="world",
        pos=(0.0, 0.0, 5.0),
        lookat=(0.0, 0.0, 0.0),
    ),
]
