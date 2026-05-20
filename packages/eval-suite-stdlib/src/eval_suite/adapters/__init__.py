"""Adapter implementations.

**In plain words.** An "adapter" is the bridge between the suite and a
specific simulator. Two ship in the box: `GymAdapter` for any
gymnasium-shaped sim with a 7-DoF end-effector action (SimplerEnv,
LIBERO, RoboCasa, Bridge), and `MujocoPlaygroundAdapter` for MJX-style
legged platforms (Unitree Go1). Together they prove the contract is
sim-agnostic; a new simulator slots in by writing one more adapter.

`GymAdapter` drives any gymnasium-shaped sim with the 7-DoF EEF action
convention (SimplerEnv, LIBERO, RoboCasa, Bridge).

`MujocoPlaygroundAdapter` drives MJX-shaped envs with joint-space
actions — legged platforms (Go1), aerial, multi-arm. The split exists
because the action shape differs, not because the rollout loop differs.
"""

from .gym import GymAdapter
from .mujoco_playground import MujocoPlaygroundAdapter

__all__ = ["GymAdapter", "MujocoPlaygroundAdapter"]
