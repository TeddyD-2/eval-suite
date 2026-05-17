"""Reference external eval-suite plugin.

Demonstrates the v0 plugin substrate:

- `HouseholdMockTask` — a Task with kitchen / pantry / living-room cell
  labels (the env underneath is a no-op MockEnv; the *labels* are the
  point, not the simulation).
- `IndustrialArmMockPolicy` — a Policy with an industrial-arm name
  (the actions underneath are zero EEF; again, the *label* is the
  point).

After `pip install -e .` from this directory, the in-tree
eval-suite CLI sees these plugins:

    python -m eval_suite.cli list                 # both appear
    python -m eval_suite.cli sweep \\
        --task household_mock \\
        --policy industrial_arm_mock \\
        --adapter gym \\
        --trials 2 \\
        --output-dir results/demo/

The sealed manifest's sidecar `plugin_provenance.json` records this
package as the source of the Task and the Policy, and `eval-suite` as
the source of the Adapter. That's the mix-and-match contract working
end-to-end.
"""

from .household_task import HouseholdMockTask
from .industrial_policy import IndustrialArmMockPolicy

__all__ = ["HouseholdMockTask", "IndustrialArmMockPolicy"]
