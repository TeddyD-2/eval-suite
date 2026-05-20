"""TopicSpec contract tests.

These don't require rclpy — the dataclass + YAML / JSON round-trips
+ sidecar signing all live in `eval_suite.ros2.topic_spec`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from eval_suite.ros2 import TopicSpec, write_topic_spec_sidecar


def _example_spec() -> TopicSpec:
    return TopicSpec(
        image_topic="/camera/image_raw",
        image_msg_type="sensor_msgs/Image",
        instruction_topic="/task/instruction",
        instruction_msg_type="std_msgs/String",
        state_topic="/joint_states",
        state_msg_type="sensor_msgs/JointState",
        action_topic="/arm/cmd",
        action_msg_type="geometry_msgs/Twist",
        action_scaling={"world_vector_x": 0.5, "gripper": 1.0},
        rate_hz=20.0,
        watchdog_timeout_s=0.2,
        workspace_bounds={"x": (0.1, 0.6), "y": (-0.3, 0.3), "z": (0.0, 0.5)},
        embodiment="franka_panda",
        description="test spec",
    )


def test_topic_spec_yaml_round_trip(tmp_path: Path) -> None:
    spec = _example_spec()
    yaml_path = tmp_path / "spec.yaml"
    yaml_path.write_text(spec.to_yaml())
    reloaded = TopicSpec.from_yaml_file(yaml_path)
    assert reloaded == spec


def test_topic_spec_canonical_json_is_stable() -> None:
    """Same fields → same canonical bytes (signing precondition)."""
    a = _example_spec()
    b = _example_spec()
    assert a.to_canonical_json() == b.to_canonical_json()


def test_topic_spec_canonical_json_changes_when_field_changes() -> None:
    a = _example_spec()
    b_kwargs = a.__dict__.copy()
    b_kwargs["rate_hz"] = 30.0
    from eval_suite.ros2 import TopicSpec as _TS
    b = _TS(**b_kwargs)
    assert a.to_canonical_json() != b.to_canonical_json()


def test_topic_spec_sidecar_writes_run_id_and_payload(tmp_path: Path) -> None:
    spec = _example_spec()
    out = write_topic_spec_sidecar(spec, run_dir=tmp_path, manifest_run_id="testrun123")
    assert out.exists()
    import json
    payload = json.loads(out.read_text())
    assert payload["manifest_run_id"] == "testrun123"
    assert payload["topic_spec"]["embodiment"] == "franka_panda"


def test_topic_spec_yaml_must_be_mapping(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- not\n- a\n- mapping\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        TopicSpec.from_yaml_file(bad)


def test_bundled_reference_yamls_parse() -> None:
    """All three bundled topic_specs (WidowX, Go1, Franka) must parse cleanly."""
    repo_root = Path(__file__).resolve().parents[1]
    specs_dir = repo_root / "packages/eval-suite-ros2/config/topic_specs"
    for name in ("widowx_bridge.yaml", "unitree_go1.yaml", "franka_panda.yaml"):
        spec = TopicSpec.from_yaml_file(specs_dir / name)
        assert spec.embodiment
        assert spec.image_topic.startswith("/")
        assert spec.action_topic.startswith("/")
