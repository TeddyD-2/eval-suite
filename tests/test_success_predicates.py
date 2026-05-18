"""Tests for `eval_suite.tasks._success_predicates`."""

from __future__ import annotations

import pytest
from eval_suite.hashing import canonical_json
from eval_suite.ingest.splat.annotation import (
    NamedRegion,
    SceneMetadata,
    SceneTransform,
    SpawnPoint,
)
from eval_suite.tasks._success_predicates import (
    PARAMS_SCALAR_TYPES,
    EnvState,
    MaintainedClearance,
    RobotReachedRegion,
    Survived,
    list_predicates,
    predicate_from_dict,
    register_predicate,
)


def _fixture_metadata() -> SceneMetadata:
    return SceneMetadata(
        schema_version="0.1.0",
        scene_transform=SceneTransform(
            up_axis="z", meters_per_unit=1.0, world_origin=(0.0, 0.0, 0.0)
        ),
        named_regions=(
            NamedRegion(name="goal", shape="box", pos=(3.0, 0.0, 0.3), size=(0.5, 0.5, 0.3)),
        ),
        spawn_points=(SpawnPoint(name="start", pos=(0.0, 0.0, 0.3)),),
    )


def test_registry_contains_v1_predicates() -> None:
    """The three v1 predicates (Survived, RobotReachedRegion,
    MaintainedClearance) must register at import time."""
    kinds = set(list_predicates())
    assert "survived" in kinds
    assert "robot_reached_region" in kinds
    assert "maintained_clearance" in kinds


def test_predicate_registry_collision_raises() -> None:
    """Registering two predicates with the same KIND must raise KeyError
    at registration (not silently overwrite). The decorator is the gate."""
    from typing import ClassVar

    from eval_suite.tasks._success_predicates import (
        PredicateOutcome,
        SuccessCriterionDict,
    )

    @register_predicate
    class _NewKind:
        KIND: ClassVar[str] = "test_unique_kind_for_collision_test"

        def to_dict(self) -> SuccessCriterionDict:
            return {"kind": self.KIND, "params": {}}

        @classmethod
        def from_dict(cls, d: SuccessCriterionDict) -> _NewKind:
            return cls()

        def reset(self, sm: SceneMetadata | None) -> None:
            return None

        def step(self, env_state: EnvState) -> PredicateOutcome:
            return PredicateOutcome(done=False, success=False)

        def at_horizon(self) -> bool:
            return False

    with pytest.raises(KeyError, match="collision"):

        @register_predicate
        class _DuplicateKind:
            KIND: ClassVar[str] = "test_unique_kind_for_collision_test"

            def to_dict(self) -> SuccessCriterionDict:
                return {"kind": self.KIND, "params": {}}

            @classmethod
            def from_dict(cls, d: SuccessCriterionDict) -> _DuplicateKind:
                return cls()

            def reset(self, sm: SceneMetadata | None) -> None:
                return None

            def step(self, env_state: EnvState) -> PredicateOutcome:
                return PredicateOutcome(done=False, success=False)

            def at_horizon(self) -> bool:
                return False


def test_each_predicate_round_trips_through_dict() -> None:
    """For each v1 predicate, `from_dict(p.to_dict())` reconstructs an
    equivalent object. Equality on the dataclass means identical params
    re-emerge."""
    from eval_suite.tasks._success_predicates import SuccessPredicate

    cases: list[SuccessPredicate] = [
        Survived(fall_height=0.12),
        RobotReachedRegion(region_name="behind_truck", tolerance=0.5),
        MaintainedClearance(region_name="conveyor", min_distance=0.3),
    ]
    for p in cases:
        d = p.to_dict()
        # Discriminator-first.
        assert "kind" in d and "params" in d
        # Scalars-only contract.
        params: dict[str, object] = d["params"]
        for k, v in params.items():
            assert isinstance(v, PARAMS_SCALAR_TYPES), (
                f"{p}: param {k}={v!r} is not a scalar"
            )
        p2 = predicate_from_dict(d)
        # Dataclass equality (excluding the private _state slot).
        assert type(p2) is type(p)
        # Compare salient public fields. All three v1 predicates are frozen
        # dataclasses, so they expose __dataclass_fields__ — but the Protocol
        # type can't statically promise it.
        for field_name in p.__dataclass_fields__:  # type: ignore[attr-defined]
            assert getattr(p, field_name) == getattr(p2, field_name)


def test_canonical_json_byte_stable() -> None:
    """canonical_json on the predicate dict is byte-identical across
    constructions with identical params. Catches accidental float
    formatting drift."""
    p1 = RobotReachedRegion(region_name="goal", tolerance=0.5)
    p2 = RobotReachedRegion(region_name="goal", tolerance=0.5)
    assert canonical_json(p1.to_dict()) == canonical_json(p2.to_dict())


def test_robot_reached_region_eval() -> None:
    """Region at (3, 0, 0.3) with size (0.5, 0.5, 0.3). Robot at (3.2, 0, 0.3)
    is inside → done=True, success=True. Robot at (5, 0, 0.3) is outside
    → done=False until horizon → at_horizon=False."""
    md = _fixture_metadata()
    p = RobotReachedRegion(region_name="goal", tolerance=0.0)
    p.reset(md)

    inside = EnvState(step_idx=1, max_steps=10, trunk_pos=(3.2, 0.0, 0.3))
    outcome = p.step(inside)
    assert outcome.done is True
    assert outcome.success is True
    assert p.at_horizon() is True

    # Fresh predicate; robot stays outside.
    p2 = RobotReachedRegion(region_name="goal", tolerance=0.0)
    p2.reset(md)
    for step in range(1, 6):
        out = p2.step(EnvState(step_idx=step, max_steps=10, trunk_pos=(5.0, 0.0, 0.3)))
        assert out.done is False
    assert p2.at_horizon() is False


def test_maintained_clearance_eval() -> None:
    """Region (3, 0, 0.3) (0.5, 0.5, 0.3) + clearance 0.2. Robot stays
    away → at_horizon=True. Robot enters → done=True success=False
    immediately, sticky after."""
    md = _fixture_metadata()
    safe = MaintainedClearance(region_name="goal", min_distance=0.2)
    safe.reset(md)
    for step in range(1, 6):
        out = safe.step(EnvState(step_idx=step, max_steps=10, trunk_pos=(0.0, 0.0, 0.3)))
        assert out.done is False
        assert out.success is True
    assert safe.at_horizon() is True

    fail = MaintainedClearance(region_name="goal", min_distance=0.2)
    fail.reset(md)
    out1 = fail.step(EnvState(step_idx=1, max_steps=10, trunk_pos=(3.0, 0.0, 0.3)))
    assert out1.done is True
    assert out1.success is False
    # Sticky: subsequent steps still fail.
    out2 = fail.step(EnvState(step_idx=2, max_steps=10, trunk_pos=(0.0, 0.0, 0.3)))
    assert out2.done is True
    assert out2.success is False
    assert fail.at_horizon() is False


def test_survived_eval() -> None:
    """Trunk z below fall_height triggers done=True, success=False
    immediately. Otherwise at_horizon stays True."""
    p = Survived(fall_height=0.15)
    p.reset(None)
    # Standing tall.
    out = p.step(EnvState(step_idx=1, max_steps=10, trunk_pos=(0.0, 0.0, 0.3)))
    assert out.done is False
    assert p.at_horizon() is True
    # Falls.
    out = p.step(EnvState(step_idx=2, max_steps=10, trunk_pos=(0.0, 0.0, 0.05)))
    assert out.done is True
    assert out.success is False
    assert p.at_horizon() is False


def test_predicate_from_dict_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unknown predicate kind"):
        predicate_from_dict({"kind": "no_such_predicate", "params": {}})


def test_predicate_from_dict_rejects_non_scalar_params() -> None:
    """The scalars-only contract is enforced at from_dict time."""
    with pytest.raises(ValueError, match="scalar"):
        RobotReachedRegion.from_dict({
            "kind": "robot_reached_region",
            "params": {"region_name": "x", "tolerance": [0.5, 0.6]},  # list — not scalar
        })


def test_predicate_from_dict_rejects_missing_required_param() -> None:
    with pytest.raises(ValueError, match="region_name"):
        RobotReachedRegion.from_dict({
            "kind": "robot_reached_region",
            "params": {"tolerance": 0.5},
        })
