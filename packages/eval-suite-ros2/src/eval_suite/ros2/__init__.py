"""eval-suite ROS 2 deployment bridge.

**In plain words.** This is how a policy evaluated in sim gets to a
real robot. A ROS 2 lifecycle node wraps the policy; a YAML topic
spec declares which camera topic feeds it and which arm topic
listens to it; and a `ProfileGate` refuses to let the node go
ACTIVE on the real robot if the attached eval-suite profile is
weaker than the deployer's published bar. Sim evaluation goes from
"a number in a paper" to "an admission contract that determines
whether the robot moves."

The static layer (TopicSpec, ProfileGate, sidecar I/O) is importable
without rclpy. The rclpy-backed PolicyNode and PolicyActionServer
import lazily; their modules guard their imports so this package can
be used on a CI runner without ROS 2 installed.
"""

from .gate import GateResult, ProfileGate
from .topic_spec import TopicSpec, write_topic_spec_sidecar

__all__ = ["TopicSpec", "write_topic_spec_sidecar", "ProfileGate", "GateResult"]
