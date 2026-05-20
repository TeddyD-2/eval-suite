"""Policy implementations.

**In plain words.** A "policy" is a robot model the suite knows how to
drive. This sub-package is the catalog of wrappers that fit
ready-made models into the suite's expected shape. `SimplerEnvPolicy`
covers Octo and RT-1; `LeRobotPolicy` covers anything on the
HuggingFace LeRobot Hub; `OXEReplayPolicy` plays back a recorded
real-world episode as if it were a model (used to pair sim and real
trajectories). The two `Mock`/`Random` policies are cheap stand-ins
used in CI tests where loading a real model would be slow.

`MockPolicy` (7-DoF zero EEF) and `RandomLocomotionPolicy` (joint-space)
are import-cheap and used by CI. `SimplerEnvPolicy` lives in
`.simpler_env` and is imported lazily (it pulls TF / JAX / SAPIEN).
"""

from .mock import MockPolicy
from .random_locomotion import RandomLocomotionPolicy

__all__ = ["MockPolicy", "RandomLocomotionPolicy"]
