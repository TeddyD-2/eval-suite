"""Task implementations.

**In plain words.** A "task" tells the suite what conditions to test
the robot under and when to call an episode a success. This
sub-package is the catalog of tasks that ship in the box —
pick-the-coke-can (Google Robot), spoon-on-towel (WidowX),
joystick-walking (Go1), three LIBERO scenes, the Namaqualand
boulder scan, the TNT-Truck Gaussian splat, and a `MockTask` used in
CI. New tasks plug in here or as a third-party pip package without
needing any changes to the suite's substrate.

`MockTask` is import-cheap and used by CI.
`GoogleRobotPickCokeCan` and `WidowXSpoonOnTowel` are lazy-imported via
`.simpler_env` because they bring SimplerEnv + ManiSkill2 with them.
"""

from .mock import MockTask

__all__ = ["MockTask"]
