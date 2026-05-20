# eval-suite-ros2

ROS 2 deployment bridge for `eval-suite`. Wraps any `eval_suite.Policy`
in a lifecycle node or action server, with the observation/action ↔
topic mapping declared explicitly in a YAML `TopicSpec` that gets
signed into a sidecar alongside the eval-suite manifest. Includes a
`ProfileGate` that refuses to transition the node to ACTIVE when the
attached eval-suite profile fails a deployer-set bar.

See `docs/ros2_deployment.md` for the full design.

## Install

```bash
# eval-suite-ros2 itself + its declarative bits work without rclpy:
pip install -e packages/eval-suite-ros2

# The runtime node + action server need rclpy. On Humble / Jazzy:
sudo apt install ros-${ROS_DISTRO}-rclpy ros-${ROS_DISTRO}-sensor-msgs ros-${ROS_DISTRO}-geometry-msgs
```

## Quick start

```bash
ros2 run eval_suite_ros2 policy_node \
    --ros-args \
    -p topic_spec_file:=packages/eval-suite-ros2/config/topic_specs/franka_panda.yaml \
    -p policy_name:=lerobot \
    -p policy_args:='{repo_id: "lerobot/smolvla-base"}' \
    -p manifest_path:=results/sweep_*/manifest.json \
    -p gate_path:=config/gate.yaml
```

When the lifecycle node configures it parses the topic spec + manifest +
gate, evaluates the gate against the manifest's profile, and only
transitions to ACTIVE if the gate passes. Refusal reasons go to rosout.

## Design

- `TopicSpec` declares which ROS 2 topic each `Observation` field comes
  from and which topic each `Action` field goes to. Bound into a signed
  sidecar so the deployer-recorded mapping is content-addressed
  alongside the eval-suite run_id.
- `PolicyNode` is a `rclpy.lifecycle.Node`. `on_configure` loads the
  policy via the eval-suite plugin registry, parses the topic spec,
  evaluates the gate. `on_activate` starts the policy loop. Watchdog
  + workspace bounds enforced in the loop.
- `PolicyActionServer` exposes the same body as a ROS 2 action server
  for behavior-tree integrations.
- `ProfileGate` is the thin slice: a deployer's `gate.yaml` declares
  things like `worst_dim_min_score: 0.6`, `min_calibration_tier: B`,
  `required_paired_pearson_r: 0.7`. `GateResult.passed` ANDs them all.

Currently the static layer (`TopicSpec`, `ProfileGate`, sidecar I/O,
YAML round-trip) ships and is CI-tested. The `rclpy`-backed runtime
nodes import lazily and are documented as the deployer-side glue.
