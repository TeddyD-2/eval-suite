"""eval-suite ROS 2 deployment bridge.

The static layer (TopicSpec, ProfileGate, sidecar I/O) is importable
without rclpy. The rclpy-backed PolicyNode and PolicyActionServer
import lazily; their modules guard their imports so this package can
be used on a CI runner without ROS 2 installed.
"""

from .gate import GateResult, ProfileGate
from .topic_spec import TopicSpec, write_topic_spec_sidecar

__all__ = ["TopicSpec", "write_topic_spec_sidecar", "ProfileGate", "GateResult"]
