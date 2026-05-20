"""Declarative, registry-keyed success predicates serializable into the
manifest's `success_criterion` field (schema 0.3.0+).

**In plain words.** A "predicate" is the rule that decides whether
an episode counts as a success — "robot reached the region behind
the truck within half a meter," "robot maintained clearance from
this body," "robot stayed alive for N steps." This file is the
small library of pre-built rules a non-programmer adopter can pick
from when defining their own task; same scene with a different
predicate produces a different (and provably distinct) run.


A predicate is a tuple of (KIND, params) that:

  1. Describes the success semantics for an episode (e.g., "reach this
     region within tolerance", "maintain clearance from this body").
  2. Serializes to a stable dict shape so the manifest's `_hashable_payload`
     binds it into the run_id deterministically — changing the goal on
     the same scene produces a distinct run_id.
  3. Reconstructs from the dict shape (round-trip).
  4. Evaluates against a per-step env state, returning a `PredicateOutcome`
     the env uses to decide success + early termination.

The factory-engineer wedge: a non-developer adopter picks a predicate
from this library, fills in scalar parameters, and runs a sweep — no
new Task subclass, no manifest schema editing, no eval-suite-core code.

Serialization shape (forced):

```python
{"kind": "<registry key>",
 "params": {"<name>": <scalar>, ...}}     # scalars only — no nested dict/list
```

The scalars-only rule rules out an entire class of "someone smuggled a
numpy.float32 into params" bugs and keeps canonical-JSON byte-stability
trivial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

from .._types import CanonicalDim  # noqa: F401 — kept for downstream type cross-ref
from ..ingest.splat.annotation import NamedRegion, SceneMetadata

__all__ = [
    "PARAMS_SCALAR_TYPES",
    "SuccessCriterionDict",
    "EnvState",
    "PredicateOutcome",
    "SuccessPredicate",
    "Survived",
    "RobotReachedRegion",
    "MaintainedClearance",
    "register_predicate",
    "predicate_from_dict",
    "list_predicates",
]

# Scalars-only contract on the params dict (forced via runtime validation
# inside from_dict). Nested dicts/lists are forbidden so canonical-JSON
# byte-stability stays trivial and you can't smuggle a non-JSON-serializable
# value through the manifest hash.
PARAMS_SCALAR_TYPES = (str, int, float, bool, type(None))

SuccessCriterionDict = dict[str, Any]
"""The on-disk dict shape: {'kind': str, 'params': dict[str, scalar]}."""


@dataclass(frozen=True)
class EnvState:
    """Minimum-viable per-step env state the predicate library consumes.
    Constructed by the env each step; passed to `SuccessPredicate.step`.
    """

    step_idx: int
    max_steps: int
    trunk_pos: tuple[float, float, float]


@dataclass(frozen=True)
class PredicateOutcome:
    """What `SuccessPredicate.step` returns. `done=True` means the predicate
    has determined the terminal outcome before max_steps; `success` is the
    final outcome value at that point. When `done=False`, the env keeps
    stepping; `success` is the at-horizon answer if step_idx reaches
    max_steps without `done` ever firing — but predicates also expose
    `at_horizon()` directly for clarity.
    """

    done: bool
    success: bool


@runtime_checkable
class SuccessPredicate(Protocol):
    """The Protocol every concrete predicate satisfies.

    `KIND` is the registry key (e.g., `"robot_reached_region"`). It MUST
    be unique across the registered set; the registry refuses duplicates
    at import time.

    `reset` is called once at the start of each episode. The
    `scene_metadata` arg lets predicates that reference named regions
    (RobotReachedRegion, MaintainedClearance) resolve the region by name
    against the loaded scene.

    **Optional method discovered via getattr:**

    - `target_position() -> tuple[float, float, float] | None` —
      returns the predicate's "where success points to" coordinate in
      scene frame (region centroid for region-based predicates, None
      for Survived). Used by the mock-env synth-trajectory path so it
      doesn't have to poke private predicate state; analysis or
      visualization layers can also read it.
    """

    KIND: ClassVar[str]

    def to_dict(self) -> SuccessCriterionDict: ...
    @classmethod
    def from_dict(cls, d: SuccessCriterionDict) -> SuccessPredicate: ...
    def reset(self, scene_metadata: SceneMetadata | None) -> None: ...
    def step(self, env_state: EnvState) -> PredicateOutcome: ...
    def at_horizon(self) -> bool: ...


_PREDICATE_REGISTRY: dict[str, type[SuccessPredicate]] = {}


def register_predicate(cls: type[SuccessPredicate]) -> type[SuccessPredicate]:
    """Class decorator. Registers `cls.KIND` → `cls`. Raises `KeyError`
    on duplicate KIND so a typo'd predicate doesn't silently overwrite
    an existing one at import time."""
    if not isinstance(getattr(cls, "KIND", None), str) or not cls.KIND:
        raise TypeError(f"{cls.__name__} must declare a non-empty `KIND: ClassVar[str]`.")
    if cls.KIND in _PREDICATE_REGISTRY:
        existing = _PREDICATE_REGISTRY[cls.KIND]
        if existing is cls:
            return cls  # idempotent re-register (e.g., reload)
        raise KeyError(
            f"SuccessPredicate KIND collision: {cls.KIND!r} already registered "
            f"to {existing.__name__}; cannot also register {cls.__name__}."
        )
    _PREDICATE_REGISTRY[cls.KIND] = cls
    return cls


def predicate_from_dict(d: SuccessCriterionDict) -> SuccessPredicate:
    """Reconstruct a predicate from its on-disk dict shape. Raises
    `ValueError` on malformed input or unknown kind."""
    if not isinstance(d, dict):
        raise ValueError(f"predicate dict must be a dict, got {type(d).__name__}")
    kind = d.get("kind")
    if not isinstance(kind, str) or not kind:
        raise ValueError(f"predicate dict missing/invalid 'kind': {d!r}")
    if kind not in _PREDICATE_REGISTRY:
        raise ValueError(
            f"Unknown predicate kind {kind!r}. Registered: "
            f"{sorted(_PREDICATE_REGISTRY)}"
        )
    return _PREDICATE_REGISTRY[kind].from_dict(d)


def list_predicates() -> tuple[str, ...]:
    """Sorted tuple of registered predicate KIND strings."""
    return tuple(sorted(_PREDICATE_REGISTRY))


def _validate_params(params: dict[str, Any], *, predicate_name: str) -> None:
    """Enforce the scalars-only contract on params."""
    if not isinstance(params, dict):
        raise ValueError(
            f"{predicate_name}.params must be a dict, got {type(params).__name__}"
        )
    for k, v in params.items():
        if not isinstance(k, str):
            raise ValueError(
                f"{predicate_name}.params keys must be str, got {type(k).__name__}"
            )
        if not isinstance(v, PARAMS_SCALAR_TYPES):
            raise ValueError(
                f"{predicate_name}.params[{k!r}] must be a scalar "
                f"(str/int/float/bool/None), got {type(v).__name__}. "
                "Nested dicts/lists are forbidden — the canonical-JSON "
                "byte-stability contract depends on the scalars-only rule."
            )


# ---------------------------------------------------------------------------
# Concrete predicates
# ---------------------------------------------------------------------------


@register_predicate
@dataclass(frozen=True)
class Survived:
    """Back-compat with the Namaqualand success semantics: episode ends
    at max_steps; success iff trunk-z never fell below `fall_height`.
    """

    KIND: ClassVar[str] = "survived"

    fall_height: float = 0.15
    # Stateful per-episode flag; tracked via private mutable proxy below.

    def __post_init__(self) -> None:
        # Frozen dataclass — use object.__setattr__ to install the private
        # mutable state slot that reset() and step() update.
        object.__setattr__(self, "_state", {"fallen": False})

    def to_dict(self) -> SuccessCriterionDict:
        return {
            "kind": self.KIND,
            "params": {"fall_height": float(self.fall_height)},
        }

    @classmethod
    def from_dict(cls, d: SuccessCriterionDict) -> Survived:
        params = d.get("params", {})
        _validate_params(params, predicate_name=cls.__name__)
        return cls(fall_height=float(params.get("fall_height", 0.15)))

    def reset(self, scene_metadata: SceneMetadata | None) -> None:
        self._state["fallen"] = False  # type: ignore[attr-defined]

    def step(self, env_state: EnvState) -> PredicateOutcome:
        if env_state.trunk_pos[2] < self.fall_height:
            self._state["fallen"] = True  # type: ignore[attr-defined]
            return PredicateOutcome(done=True, success=False)
        return PredicateOutcome(done=False, success=False)

    def at_horizon(self) -> bool:
        return not self._state["fallen"]  # type: ignore[attr-defined]

    def target_position(self) -> tuple[float, float, float] | None:
        # Survival has no spatial target — caller should fall back to
        # the robot's start position.
        return None


@register_predicate
@dataclass(frozen=True)
class RobotReachedRegion:
    """Robot reaches a `NamedRegion` (referenced by name in the scene's
    scene_metadata). Success the moment the robot's trunk enters the
    region's shape-aware containment volume (box → AABB; sphere/cylinder →
    inside primitive). Failure at horizon if never entered.

    `tolerance` is shape-dependent additive slack: for spheres it extends
    the radius; for boxes it expands the half-size; for cylinders it
    extends both radius and half-height.
    """

    KIND: ClassVar[str] = "robot_reached_region"

    region_name: str
    tolerance: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "_state", {"region": None, "reached": False})

    def to_dict(self) -> SuccessCriterionDict:
        return {
            "kind": self.KIND,
            "params": {
                "region_name": str(self.region_name),
                "tolerance": float(self.tolerance),
            },
        }

    @classmethod
    def from_dict(cls, d: SuccessCriterionDict) -> RobotReachedRegion:
        params = d.get("params", {})
        _validate_params(params, predicate_name=cls.__name__)
        if "region_name" not in params:
            raise ValueError(f"{cls.__name__}.params missing 'region_name'")
        return cls(
            region_name=str(params["region_name"]),
            tolerance=float(params.get("tolerance", 0.5)),
        )

    def reset(self, scene_metadata: SceneMetadata | None) -> None:
        if scene_metadata is None:
            self._state["region"] = None  # type: ignore[attr-defined]
        else:
            self._state["region"] = scene_metadata.region(self.region_name)  # type: ignore[attr-defined]
        self._state["reached"] = False  # type: ignore[attr-defined]

    def step(self, env_state: EnvState) -> PredicateOutcome:
        region: NamedRegion | None = self._state["region"]  # type: ignore[attr-defined]
        if region is None or not _trunk_in_region(env_state.trunk_pos, region, self.tolerance):
            return PredicateOutcome(done=False, success=False)
        self._state["reached"] = True  # type: ignore[attr-defined]
        return PredicateOutcome(done=True, success=True)

    def at_horizon(self) -> bool:
        return bool(self._state["reached"])  # type: ignore[attr-defined]

    def target_position(self) -> tuple[float, float, float] | None:
        # Centroid of the bound region (None until reset() has resolved
        # the region against scene_metadata).
        region: NamedRegion | None = self._state["region"]  # type: ignore[attr-defined]
        if region is None:
            return None
        return (float(region.pos[0]), float(region.pos[1]), float(region.pos[2]))


@register_predicate
@dataclass(frozen=True)
class MaintainedClearance:
    """Robot maintains a minimum distance from a `NamedRegion` (e.g., a
    keep-out zone, or the projected footprint of an `ExtractedBody`).
    Fails the moment the robot's trunk enters the region (extended by
    `min_distance`); succeeds at horizon if never violated.

    The factory keep-out semantics: this is the predicate that justifies
    shipping the `ExtractedBody` substrate in v1. The TNT Truck demo's
    extracted truck body is a clearance target for a follow-up sweep.
    """

    KIND: ClassVar[str] = "maintained_clearance"

    region_name: str
    min_distance: float = 0.3

    def __post_init__(self) -> None:
        object.__setattr__(self, "_state", {"region": None, "violated": False})

    def to_dict(self) -> SuccessCriterionDict:
        return {
            "kind": self.KIND,
            "params": {
                "region_name": str(self.region_name),
                "min_distance": float(self.min_distance),
            },
        }

    @classmethod
    def from_dict(cls, d: SuccessCriterionDict) -> MaintainedClearance:
        params = d.get("params", {})
        _validate_params(params, predicate_name=cls.__name__)
        if "region_name" not in params:
            raise ValueError(f"{cls.__name__}.params missing 'region_name'")
        return cls(
            region_name=str(params["region_name"]),
            min_distance=float(params.get("min_distance", 0.3)),
        )

    def reset(self, scene_metadata: SceneMetadata | None) -> None:
        if scene_metadata is None:
            self._state["region"] = None  # type: ignore[attr-defined]
        else:
            self._state["region"] = scene_metadata.region(self.region_name)  # type: ignore[attr-defined]
        self._state["violated"] = False  # type: ignore[attr-defined]

    def step(self, env_state: EnvState) -> PredicateOutcome:
        if self._state["violated"]:  # type: ignore[attr-defined]
            # Sticky failure once violated.
            return PredicateOutcome(done=True, success=False)
        region: NamedRegion | None = self._state["region"]  # type: ignore[attr-defined]
        if region is None:
            return PredicateOutcome(done=False, success=True)
        if _trunk_in_region(env_state.trunk_pos, region, self.min_distance):
            self._state["violated"] = True  # type: ignore[attr-defined]
            return PredicateOutcome(done=True, success=False)
        return PredicateOutcome(done=False, success=True)

    def at_horizon(self) -> bool:
        return not bool(self._state["violated"])  # type: ignore[attr-defined]

    def target_position(self) -> tuple[float, float, float] | None:
        # Clearance targets the keep-out region itself; the mock env
        # uses the same fall-toward-target trajectory shape it uses for
        # the reach predicate, which is fine for the synth trajectory.
        region: NamedRegion | None = self._state["region"]  # type: ignore[attr-defined]
        if region is None:
            return None
        return (float(region.pos[0]), float(region.pos[1]), float(region.pos[2]))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _trunk_in_region(
    trunk_pos: tuple[float, float, float],
    region: NamedRegion,
    extra: float,
) -> bool:
    """Shape-aware containment test. `extra` is additive slack."""
    px, py, pz = trunk_pos
    rx, ry, rz = region.pos
    if region.shape == "box":
        sx, sy, sz = region.size
        return (
            abs(px - rx) <= sx + extra
            and abs(py - ry) <= sy + extra
            and abs(pz - rz) <= sz + extra
        )
    if region.shape == "sphere":
        (radius,) = region.size
        d2 = (px - rx) ** 2 + (py - ry) ** 2 + (pz - rz) ** 2
        return d2 <= (radius + extra) ** 2
    if region.shape == "cylinder":
        radius, half_h = region.size
        if abs(pz - rz) > half_h + extra:
            return False
        d2 = (px - rx) ** 2 + (py - ry) ** 2
        return d2 <= (radius + extra) ** 2
    return False  # pragma: no cover — schema validation should prevent this
