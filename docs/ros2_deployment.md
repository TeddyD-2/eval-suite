# ROS 2 deployment path

This page documents how a policy evaluated through `eval-suite` reaches
a real robot through ROS 2: the observation/action ↔ topic mapping, the
lifecycle node that wraps any `eval_suite.Policy`, and the
`ProfileGate` admission controller that refuses activation when the
attached eval-suite profile fails a deployer-set bar.

The substrate ships in the sibling package
`packages/eval-suite-ros2/`. It depends on `eval-suite-core` only — no
new requirement on `eval-suite-stdlib`, so a third-party policy
package can ship its own deployment story without depending on the
in-tree reference plugins.

## Architecture

```
            ┌──────────────────────────┐
            │   eval_suite.Policy      │   ← unchanged from sim
            │   (LeRobot / SimplerEnv  │
            │    / OXEReplay / ... )   │
            └──────────┬───────────────┘
                       │
                       │ Observation, ActionLike
                       ▼
   ┌─────────────────────────────────────────┐
   │  PolicyNode (rclpy.lifecycle.Node)      │
   │  ─ on_configure: load topic_spec.yaml,  │
   │       policy, manifest, gate.yaml       │
   │  ─ on_activate: subscribe + publish     │
   │       at TopicSpec.rate_hz              │
   │  ─ watchdog + workspace_bounds enforced │
   └────┬──────────────┬─────────────────────┘
        │              │
   subscribers      publisher
        │              │
        ▼              ▼
   /camera/...    /arm/cmd
   /joint_states  (geometry_msgs/Twist | sensor_msgs/JointState | ... )
   /task/instruction
```

## TopicSpec — the obs/action ↔ topic mapping

A `TopicSpec` declares which ROS 2 topic each `Observation` field comes
from and which topic each `Action` field goes to. The mapping is
deliberately *declarative* (a YAML file) so the deployment-time wiring
is auditable: two operators with the same manifest + the same topic
spec hit the same robot the same way.

Bundled reference YAMLs at `packages/eval-suite-ros2/config/topic_specs/`:

- `widowx_bridge.yaml` — WidowX 6 arm, Bridge teleop topic graph.
  7-DoF EEF as `std_msgs/Float32MultiArray`.
- `unitree_go1.yaml` — Unitree Go1 quadruped, 12-DoF joint targets as
  `sensor_msgs/JointState`.
- `franka_panda.yaml` — Franka Emika Panda, 7-DoF EEF Cartesian
  velocity as `geometry_msgs/Twist`. Default ROS 2 manipulation
  reference embodiment.

Each spec gets signed into a `topic_spec.json` sidecar alongside the
eval-suite manifest via `write_topic_spec_sidecar(...)`. Same Ed25519
signing machinery as the manifest itself, so deployer tampering is
detectable.

## Per-embodiment mapping table

| Embodiment       | Image topic                 | State topic                 | Action topic                         | Action msg type            |
|------------------|-----------------------------|-----------------------------|--------------------------------------|----------------------------|
| widowx_bridge    | /widowx/camera/image_raw    | /widowx/joint_states        | /widowx/cartesian_command            | std_msgs/Float32MultiArray |
| unitree_go1      | /go1/camera/front/image_raw | /go1/joint_states           | /go1/joint_commands                  | sensor_msgs/JointState     |
| franka_panda     | /camera/color/image_raw     | /franka_state_controller/joint_states | /cartesian_velocity_controller/command | geometry_msgs/Twist |

`Action.world_vector` → `Twist.linear.{x,y,z}`. `Action.rot_axangle` →
`Twist.angular.{x,y,z}`. `Action.gripper` is dropped (separate action
server on Panda; in-band on WidowX); add a sibling YAML for the
gripper if your platform needs it routed.

## Running the lifecycle node

```bash
# Build (colcon — your existing ROS 2 workspace)
cd ~/ros2_ws
colcon build --packages-select eval_suite_ros2

# Source the install
source install/setup.bash

# Run with a topic spec, a policy, a manifest, and a gate
ros2 run eval_suite_ros2 policy_node \
    --ros-args \
    -p topic_spec_file:=$REPO/packages/eval-suite-ros2/config/topic_specs/franka_panda.yaml \
    -p policy_name:=lerobot \
    -p policy_args:='{"repo_id": "lerobot/smolvla-base"}' \
    -p manifest_path:=$REPO/results/sweep_*/manifest.json \
    -p gate_path:=$HOME/franka_gate.yaml
```

## Gating: refusing activation when the profile is too weak

The `ProfileGate` is the **thin slice** that ties the substrate to
deployment trust. A `gate.yaml`:

```yaml
worst_dim_min_score: 0.6
required_canonical_dims: [visuals, physics]
min_calibration_tier: B
required_paired_pearson_r: 0.7         # set if you have a paired_trials.json sidecar (Phase 3)
policy_family_allowlist: [lerobot, simpler_env]
```

`on_activate` evaluates the gate against the attached manifest. If it
fails, the lifecycle transition returns FAILURE and the refusal reasons
land in `rosout`:

```
[INFO] PolicyNode configuring with policy=lerobot topic_spec=...
[ERROR] ProfileGate refused activation:
        worst_dim_score=0.250 below required 0.600;
        calibration tier 'C' below required 'B'.
        Adjust the gate (gate.yaml) or run an additional sweep that
        clears the bar.
```

Try the local demo (no rclpy needed):

```bash
python examples/gate_smolvla_deployment.py
```

The demo synthesizes a fixture manifest matching the README's
RT-1-on-Google-Robot profile shape (weak language axis at 0.27), runs
a strict and a relaxed gate against it, and prints the refusal reasons
in one case + the pass in the other.

## What you trade off by adding a gate

The gate is a forcing function: if the profile doesn't measure the
dimensions your deployment cares about, the gate refuses the
deployment. That's the point — it surfaces a measurement gap *before*
the robot moves. The gate's reasons tell the deployer exactly what
they need to run an additional sweep against (a different cell grid, a
calibration-tier upgrade, a different policy family) to clear it.

## Safety primitives the node enforces

- **Watchdog.** If no fresh image arrives within
  `TopicSpec.watchdog_timeout_s`, the policy loop skips the tick and
  publishes zero action. Repeated misses trigger a deactivate.
- **Workspace bounds.** Per-axis AABB declared in the topic spec
  (`workspace_bounds.{x,y,z}: [lo, hi]`); commands outside are clipped
  with a warning. `null` disables the check.
- **Sim→real action scaling.** Per-axis multiplier so the deployer
  can cap aggressive sim actions on real hardware without re-training.

## What's deliberately NOT in this package

- A canonical ROS 2 *Adapter* (the eval-suite `Adapter` Protocol is
  for sim, not for the real robot loop). The lifecycle node *uses* a
  Policy directly; there's no Adapter on the deployment side because
  the robot isn't a gymnasium env.
- A real-robot rollout recorder. That's deployment telemetry, ingested
  separately into the calibration registry (Phase 3 + v2 of the
  roadmap).
- Sim-time gating. The gate runs at deployment time against the
  sealed manifest; sim-time selection happens in the sweep / portal.
