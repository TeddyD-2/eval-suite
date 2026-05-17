"""Adapter implementations.

`GymAdapter` drives any gymnasium-shaped sim with the 7-DoF EEF action
convention (SimplerEnv, LIBERO, RoboCasa, Bridge).

`MujocoPlaygroundAdapter` drives MJX-shaped envs with joint-space
actions — legged platforms (Go1), aerial, multi-arm. The split exists
because the action shape differs, not because the rollout loop differs.
"""

from .gym import GymAdapter
from .mujoco_playground import MujocoPlaygroundAdapter

__all__ = ["GymAdapter", "MujocoPlaygroundAdapter"]
