"""TopicSpec — declarative obs/action ↔ ROS 2 topic mapping.

**In plain words.** A small YAML file that says "the robot's camera
is on /camera/color/image_raw, send actions to /arm/cmd, stop if
no image arrives in 0.5 seconds, never command outside this xyz
box." It's signed and stored next to the manifest so anybody
auditing a deployment can see exactly which topics the robot was
listening to and writing to. Without this, the sim-to-real wiring
is implicit in someone's launch script; with it, the wiring is
auditable.


The deployer wires sim observation fields to real-robot topics and the
policy's action fields to real-robot command topics. This file makes
that mapping a first-class signed artifact alongside the eval-suite
manifest. Two reasons that's load-bearing:

1. **Audit.** A deployer wants to know exactly what their robot was
   listening to and writing to when this manifest's policy was
   running. A signed `topic_spec.json` answers that question.
2. **Reproducibility of deployment.** Two deployers with the same
   manifest + the same topic spec hit the same robot the same way.
   Differences in topic graph become legible at review time, not at
   "robot moved unexpectedly" time.

YAML is the editing surface; canonical JSON is the signing surface.
The sidecar uses the same Ed25519 signing machinery as the manifest
(`eval_suite.signing.sign`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TopicSpec:
    """One declarative obs/action ↔ ROS 2 topic mapping.

    Designed to be portable across ROS 2 distros (Humble, Iron, Jazzy)
    — message-type names are strings (e.g. `"sensor_msgs/Image"`)
    rather than imported Python classes, so the spec can be parsed
    without rclpy installed.
    """

    # --- Observation side -------------------------------------------------
    image_topic: str
    image_msg_type: str  # "sensor_msgs/Image" | "sensor_msgs/CompressedImage"
    instruction_topic: str
    instruction_msg_type: str = "std_msgs/String"
    state_topic: str | None = None
    state_msg_type: str | None = None  # "sensor_msgs/JointState" etc.

    # --- Action side ------------------------------------------------------
    action_topic: str = ""
    action_msg_type: str = ""
    # Per-axis sim→real scaling. Keys are field names from
    # eval_suite._types.Action ("world_vector_x", ...) or joint names
    # for JointAction; values are linear multipliers.
    action_scaling: dict[str, float] = field(default_factory=dict)

    # --- Runtime safety ---------------------------------------------------
    rate_hz: float = 10.0  # policy step rate
    watchdog_timeout_s: float = 0.5
    # Workspace AABB in robot base frame (meters). None = no bound.
    workspace_bounds: dict[str, tuple[float, float]] | None = None

    # --- Provenance -------------------------------------------------------
    embodiment: str = ""  # "widowx_bridge", "unitree_go1", "franka_panda", ...
    description: str = ""

    def to_canonical_json(self) -> str:
        """Stable JSON for signing. Keys sorted; tuples serialized as
        lists; None values preserved (workspace_bounds may legitimately
        be None and we want byte-identical serialization either way).
        """
        payload = asdict(self)
        # Tuples in workspace_bounds become lists in JSON automatically;
        # but only if they're recursively dataclass-converted, which
        # asdict() does for nested dicts only. Force the conversion
        # explicitly so signing is stable.
        if payload.get("workspace_bounds"):
            payload["workspace_bounds"] = {
                k: [float(v[0]), float(v[1])] for k, v in payload["workspace_bounds"].items()
            }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> TopicSpec:
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"topic_spec YAML at {path} must be a mapping, got {type(raw)}")
        # Normalize workspace_bounds: YAML lists → tuples for the dataclass.
        if raw.get("workspace_bounds"):
            raw["workspace_bounds"] = {
                k: (float(v[0]), float(v[1])) for k, v in raw["workspace_bounds"].items()
            }
        return cls(**raw)

    def to_yaml(self) -> str:
        payload = asdict(self)
        # PyYAML serializes tuples as Python objects (not portable);
        # convert to lists for the on-disk form.
        if payload.get("workspace_bounds"):
            payload["workspace_bounds"] = {
                k: [float(v[0]), float(v[1])] for k, v in payload["workspace_bounds"].items()
            }
        return str(yaml.safe_dump(payload, sort_keys=True))


def write_topic_spec_sidecar(
    spec: TopicSpec,
    *,
    run_dir: str | Path,
    manifest_run_id: str,
    sign_key: str | None = None,
) -> Path:
    """Emit `topic_spec.json` next to the manifest in a sweep dir.

    The sidecar carries the spec's canonical JSON + the manifest's
    run_id + (optionally) an Ed25519 signature over the canonical
    bytes. The reader checks: signature verifies, sidecar SHA matches
    the spec.
    """
    run_dir = Path(run_dir)
    payload: dict[str, Any] = {
        "manifest_run_id": manifest_run_id,
        "topic_spec": json.loads(spec.to_canonical_json()),
    }
    if sign_key:
        from eval_suite.signing import sign  # local import; signing isn't a hard dep
        sig = sign(spec.to_canonical_json(), bytes.fromhex(sign_key))
        payload["signature"] = sig
    out_path = run_dir / "topic_spec.json"
    out_path.write_text(json.dumps(payload, sort_keys=True, indent=2))
    return out_path
