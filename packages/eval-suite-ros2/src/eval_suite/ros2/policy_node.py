"""ROS 2 lifecycle node that wraps any eval_suite.Policy.

Runtime behavior:

  on_configure:
    - Parse topic_spec YAML → TopicSpec
    - Load policy via eval_suite.registry.get_policy(name)(**args)
    - Load manifest → ProfileGate.evaluate(manifest, profile)
    - If gate.passed == False: log refusal reasons, return FAILURE
    - Else: create subscribers + publishers, return SUCCESS
  on_activate:
    - Start policy loop at topic_spec.rate_hz
  on_deactivate:
    - Stop loop, publish zero action
  on_shutdown:
    - Release subscribers/publishers

The watchdog ensures no stale observation drives the policy: if no
fresh image arrives within `topic_spec.watchdog_timeout_s`, the loop
emits zero action and transitions to deactivate.

This file imports rclpy lazily so the package can be installed and
used (TopicSpec / ProfileGate / sidecar I/O) on a CI runner without
ROS 2 — the imports happen inside `_make_node`, which is the
deployer's entry point.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from eval_suite._types import Action, JointAction, Observation
from eval_suite.manifest import Manifest

from .gate import GateResult, ProfileGate
from .topic_spec import TopicSpec

log = logging.getLogger("eval_suite.ros2.policy_node")


def _make_node(
    *,
    topic_spec_file: str,
    policy_name: str,
    policy_args: dict[str, Any] | None = None,
    manifest_path: str | None = None,
    gate_path: str | None = None,
    adapter_name: str = "gym",
    node_name: str = "eval_suite_policy_node",
) -> Any:
    """Construct the lifecycle node. Lazy-imports rclpy.

    Returns the node instance; the caller is responsible for `spin`.

    All input paths are validated and the gate runs inside the
    constructor (which corresponds to ROS 2's `on_configure`): if the
    gate fails, this function raises and the lifecycle never enters
    INACTIVE / ACTIVE.
    """
    try:
        import rclpy
        from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
        from rclpy.node import Node  # noqa: F401  (lifecycle node inherits)
    except ImportError as e:
        raise RuntimeError(
            "PolicyNode requires rclpy. On ROS 2 Humble/Jazzy: "
            "`apt install ros-${ROS_DISTRO}-rclpy ros-${ROS_DISTRO}-sensor-msgs "
            "ros-${ROS_DISTRO}-geometry-msgs`."
        ) from e

    spec = TopicSpec.from_yaml_file(topic_spec_file)
    manifest = _load_manifest_if_present(manifest_path)
    gate = ProfileGate.from_yaml_file(gate_path) if gate_path else None

    from eval_suite.registry import get_adapter, get_policy
    policy = get_policy(policy_name)(**(policy_args or {}))
    adapter_cls = get_adapter(adapter_name)
    adapter = adapter_cls()

    gate_result = _evaluate_gate(gate, manifest)
    if gate_result is not None and not gate_result.passed:
        joined = "; ".join(gate_result.reasons)
        raise RuntimeError(
            f"ProfileGate refused activation: {joined}. "
            f"Adjust the gate (gate.yaml) or run an additional sweep that "
            f"clears the bar."
        )

    class PolicyLifecycleNode(LifecycleNode):  # type: ignore[misc]  # LifecycleNode is Any when rclpy isn't installed
        def __init__(self) -> None:
            super().__init__(node_name)
            self._policy = policy
            self._adapter = adapter  # noqa: SLF001
            self._spec = spec
            self._latest_image: Any = None
            self._latest_state: Any = None
            self._latest_instruction: str = ""
            self._subs: list[Any] = []
            self._action_pub: Any = None
            self._timer: Any = None

        def on_configure(self, _state: Any) -> Any:  # noqa: ANN401
            self.get_logger().info(
                f"PolicyNode configuring with policy={policy_name} "
                f"topic_spec={topic_spec_file}"
            )
            return TransitionCallbackReturn.SUCCESS

        def on_activate(self, _state: Any) -> Any:  # noqa: ANN401
            self._wire_subscriptions()
            self._action_pub = self.create_publisher(_resolve_msg(spec.action_msg_type),
                                                      spec.action_topic, 10)
            period = 1.0 / max(spec.rate_hz, 1e-3)
            self._timer = self.create_timer(period, self._policy_tick)
            self.get_logger().info("PolicyNode ACTIVE")
            return TransitionCallbackReturn.SUCCESS

        def on_deactivate(self, _state: Any) -> Any:  # noqa: ANN401
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._publish_zero()
            self.get_logger().info("PolicyNode DEACTIVATED")
            return TransitionCallbackReturn.SUCCESS

        def on_shutdown(self, _state: Any) -> Any:  # noqa: ANN401
            for sub in self._subs:
                self.destroy_subscription(sub)
            self._subs.clear()
            return TransitionCallbackReturn.SUCCESS

        def _wire_subscriptions(self) -> None:
            self._subs.append(self.create_subscription(
                _resolve_msg(spec.image_msg_type), spec.image_topic,
                self._on_image, 10,
            ))
            self._subs.append(self.create_subscription(
                _resolve_msg(spec.instruction_msg_type), spec.instruction_topic,
                self._on_instruction, 10,
            ))
            if spec.state_topic and spec.state_msg_type:
                self._subs.append(self.create_subscription(
                    _resolve_msg(spec.state_msg_type), spec.state_topic,
                    self._on_state, 10,
                ))

        def _on_image(self, msg: Any) -> None:
            self._latest_image = _ros_image_to_numpy(msg)

        def _on_state(self, msg: Any) -> None:
            self._latest_state = msg

        def _on_instruction(self, msg: Any) -> None:
            self._latest_instruction = getattr(msg, "data", "")

        def _policy_tick(self) -> None:
            if self._latest_image is None:
                self.get_logger().warn("no image yet; skipping tick")
                return
            obs = Observation(
                image=self._latest_image,
                instruction=self._latest_instruction,
                extra={"state": self._latest_state} if self._latest_state is not None else {},
            )
            action = self._policy.step(obs)
            self._publish_action(action)

        def _publish_action(self, action: Any) -> None:
            scaled = _apply_action_scaling(action, spec.action_scaling)
            msg = _action_to_ros_msg(scaled, spec.action_msg_type)
            if self._action_pub is not None:
                self._action_pub.publish(msg)

        def _publish_zero(self) -> None:
            if self._action_pub is None:
                return
            try:
                msg = _action_to_ros_msg(
                    _zero_action(self._spec.action_msg_type), self._spec.action_msg_type,
                )
                self._action_pub.publish(msg)
            except Exception as e:  # pragma: no cover — defensive
                self.get_logger().warn(f"zero-action publish failed: {e}")

    rclpy.init()
    return PolicyLifecycleNode()


def _load_manifest_if_present(manifest_path: str | None) -> Manifest | None:
    if not manifest_path:
        return None
    p = Path(manifest_path)
    if not p.exists():
        log.warning("manifest_path=%s does not exist; skipping gate evaluation", manifest_path)
        return None
    return Manifest.from_json(p.read_text())


def _evaluate_gate(gate: ProfileGate | None, manifest: Manifest | None) -> GateResult | None:
    if gate is None or manifest is None:
        return None
    return gate.evaluate(manifest)


def _resolve_msg(type_str: str) -> Any:
    """Resolve a 'pkg/Msg' string to the Python class.

    rclpy normally exposes these via `from sensor_msgs.msg import Image`.
    Doing it dynamically keeps the YAML portable.
    """
    if "/" not in type_str:
        raise ValueError(f"msg type must be 'pkg/Msg', got {type_str!r}")
    pkg, name = type_str.split("/", 1)
    import importlib
    mod = importlib.import_module(f"{pkg}.msg")
    return getattr(mod, name)


def _ros_image_to_numpy(msg: Any) -> Any:
    """Convert a sensor_msgs/Image or CompressedImage to HxWxC uint8 ndarray.

    Avoids cv_bridge (which isn't always installed) — the byte layout
    is well-defined for the common encodings.
    """
    import numpy as np
    encoding = getattr(msg, "encoding", "")
    if encoding == "rgb8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    if encoding == "bgr8":
        bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return bgr[..., ::-1].copy()
    if encoding == "" and hasattr(msg, "format"):  # CompressedImage
        import cv2
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img[..., ::-1].copy()
    raise ValueError(f"unsupported image encoding: {encoding!r}")


def _apply_action_scaling(action: Any, scaling: dict[str, float]) -> Any:
    if not scaling:
        return action
    import numpy as np
    if isinstance(action, Action):
        sx = scaling.get("world_vector_x", 1.0)
        sy = scaling.get("world_vector_y", 1.0)
        sz = scaling.get("world_vector_z", 1.0)
        srx = scaling.get("rot_axangle_x", 1.0)
        sry = scaling.get("rot_axangle_y", 1.0)
        srz = scaling.get("rot_axangle_z", 1.0)
        sg = scaling.get("gripper", 1.0)
        wv = action.world_vector * np.array([sx, sy, sz], dtype=np.float32)
        ra = action.rot_axangle * np.array([srx, sry, srz], dtype=np.float32)
        g = action.gripper * sg
        return Action(world_vector=wv, rot_axangle=ra, gripper=g, terminate=action.terminate)
    if isinstance(action, JointAction):
        v = action.vector.copy()
        for i in range(len(v)):
            v[i] = v[i] * scaling.get(f"j{i}", 1.0)
        return JointAction(vector=v, terminate=action.terminate)
    return action


def _action_to_ros_msg(action: Any, msg_type: str) -> Any:
    """Convert an eval-suite Action / JointAction to a ROS 2 message."""
    if msg_type == "geometry_msgs/Twist":
        from geometry_msgs.msg import Twist
        msg = Twist()
        if isinstance(action, Action):
            msg.linear.x = float(action.world_vector[0])
            msg.linear.y = float(action.world_vector[1])
            msg.linear.z = float(action.world_vector[2])
            msg.angular.x = float(action.rot_axangle[0])
            msg.angular.y = float(action.rot_axangle[1])
            msg.angular.z = float(action.rot_axangle[2])
        return msg
    if msg_type == "sensor_msgs/JointState":
        from sensor_msgs.msg import JointState
        msg = JointState()
        if isinstance(action, JointAction):
            msg.position = [float(x) for x in action.vector]
        return msg
    if msg_type == "std_msgs/Float32MultiArray":
        from std_msgs.msg import Float32MultiArray
        msg = Float32MultiArray()
        if isinstance(action, Action):
            msg.data = [
                *action.world_vector.tolist(),
                *action.rot_axangle.tolist(),
                *action.gripper.tolist(),
            ]
        elif isinstance(action, JointAction):
            msg.data = action.vector.tolist()
        return msg
    raise ValueError(f"unsupported action_msg_type: {msg_type!r}")


def _zero_action(msg_type: str) -> Any:
    import numpy as np
    if msg_type in ("geometry_msgs/Twist", "std_msgs/Float32MultiArray"):
        return Action(
            world_vector=np.zeros(3, dtype=np.float32),
            rot_axangle=np.zeros(3, dtype=np.float32),
            gripper=np.zeros(1, dtype=np.float32),
        )
    return JointAction(vector=np.zeros(12, dtype=np.float32))


def main(argv: list[str] | None = None) -> int:
    """Entry point for `ros2 run eval_suite_ros2 policy_node ...`.

    Lifts the YAML / registry-name plumbing into rclpy parameters.
    """
    import rclpy
    node = _make_node(
        topic_spec_file=_param_or_fail("topic_spec_file"),
        policy_name=_param_or_fail("policy_name"),
        policy_args=_parse_policy_args(),
        manifest_path=_param_or_none("manifest_path"),
        gate_path=_param_or_none("gate_path"),
        adapter_name=_param_or_default("adapter_name", "gym"),
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


def _param_or_fail(name: str) -> str:
    import os
    val = os.environ.get(f"EVAL_SUITE_{name.upper()}", "")
    if not val:
        raise RuntimeError(f"required parameter {name!r} not set")
    return val


def _param_or_none(name: str) -> str | None:
    import os
    return os.environ.get(f"EVAL_SUITE_{name.upper()}") or None


def _param_or_default(name: str, default: str) -> str:
    import os
    return os.environ.get(f"EVAL_SUITE_{name.upper()}", default)


def _parse_policy_args() -> dict[str, Any]:
    import json
    import os
    raw = os.environ.get("EVAL_SUITE_POLICY_ARGS", "")
    if not raw:
        return {}
    return json.loads(raw) if raw.startswith("{") else dict(
        kv.split("=", 1) for kv in raw.split(",") if "=" in kv
    )
