"""Task implementations.

`MockTask` is import-cheap and used by CI.
`GoogleRobotPickCokeCan` and `WidowXSpoonOnTowel` are lazy-imported via
`.simpler_env` because they bring SimplerEnv + ManiSkill2 with them.
"""

from .mock import MockTask

__all__ = ["MockTask"]
