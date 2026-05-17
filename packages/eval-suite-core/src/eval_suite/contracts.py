"""Interface contracts for the eval-suite.

Four Protocols define the substrate:

- `Policy` — anything that maps observations to actions. Octo and RT-1 (via
  SimplerEnv) implement this; a MockPolicy for CI implements it; a future
  third-party submission's open-weights model implements it.

- `Task` — a benchmark task family with N variant cells. Each cell has a
  unique `CellId` (axes mapping) and produces a fresh env via `build_env`.

- `Adapter` — the bridge to a gym-shaped simulator backend. v0 ships
  `GymAdapter` (drives both SimplerEnv and LIBERO) and
  `MujocoPlaygroundAdapter` (MJX / Unitree Go1). The split exists only
  because legged control needs a different `Action` shape; the rollout
  loop is identical. `rollout` is one (policy, task, cell, seed) →
  RolloutResult.

- `Manifest` — the content-addressed reproducibility record. See
  `manifest.py` for the dataclass implementation; this Protocol exists so
  alternate manifest formats (e.g. a future signed-attestation variant)
  can plug in without changing call sites.

Runtime checking is enabled so tests can assert isinstance without the
overhead of full structural verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ._types import ActionLike, CellId, Observation, RolloutResult

__all__ = ["Policy", "Task", "Adapter", "Manifest", "CompatibilityReport", "CONTRACT_VERSION"]


# Declared stable for v0. Plugin packages target this version and the
# manifest sidecar (plugin_provenance.json) records both the version
# they targeted at registration time and the version eval-suite was
# running at sweep time.
#
# Stability commitments (semver):
#   - MAJOR: breaking Protocol changes — renaming or removing a
#     REQUIRED method on Policy / Task / Adapter / Manifest, or
#     changing a required return type. Triggers a deprecation cycle.
#   - MINOR: purely additive optional methods or class attributes
#     (e.g. adding `Task.canonical_axis_map` → 1.1.0).
#   - PATCH: docstring / behavior clarifications that don't change
#     the wire format.
#
# Optional hooks discovered via `getattr` (instruction_for,
# extract_image, canonical_axis_map, can_drive, ACTION_SPACE_HINT,
# OUTPUT_TYPE) are NOT in the stability surface — they can be removed
# in a future MAJOR via a one-MINOR deprecation cycle. The required
# Protocol methods listed below are what's actually stable.
CONTRACT_VERSION = "1.0.0"


@dataclass(frozen=True)
class CompatibilityReport:
    """Result of an Adapter's pre-flight check that a Policy can drive a Task.

    **Conservative semantics:** `ok=True` means *"no known reason to
    refuse"* — it is **not** a guarantee that the rollout will succeed.
    Action-space taxonomies are messier than a small enum can capture
    (quadrupeds, bimanual, mobile bases, multi-fingered hands all have
    nuance). The check rejects mismatches the Adapter is *sure* about;
    it cannot certify compatibility. The runtime `TypeError` raised by
    `GymAdapter._flatten_action` and friends stays as the genuine
    safety net for the cases this pre-flight couldn't rule out.

    `reason` is a human-readable explanation when `ok=False`. It's
    surfaced verbatim in the CLI error and in the portal's rejection
    metadata, so it should be a sentence a deployer can act on.
    """

    ok: bool
    reason: str | None = None


@runtime_checkable
class Policy(Protocol):
    """A robot policy: stateful per-episode, maps observations to actions.

    `reset` is called once per episode with the language instruction. Some
    policies (Octo) need the instruction at reset; others (RT-1) use it on
    every step. Implementations should cache as needed.

    `step` returns an `ActionLike` (Action for 7-DoF EEF manipulation, or
    JointAction for joint-space embodiments like Go1). The Adapter
    dispatches on the concrete type.
    """

    def reset(self, instruction: str) -> None: ...

    def step(self, observation: Observation) -> ActionLike: ...

    @property
    def name(self) -> str: ...

    @property
    def checkpoint_id(self) -> str:
        """A stable identifier for the loaded weights — used in the manifest."""
        ...


@runtime_checkable
class Task(Protocol):
    """A benchmark task family with N variant cells.

    `n_cells` is fixed at construction. `cell_id(i)` returns the variant
    axis mapping for cell `i`. `build_env(i)` constructs a fresh gymnasium
    env for that cell — fresh because most sim envs hold internal state
    that doesn't reset cleanly between cells.

    **Optional hooks** that the `GymAdapter` will call if present (so
    sim-specific quirks live in the Task, not in the Adapter):

    - `instruction_for(env) -> str` — extract the language instruction
      for this episode. SimplerEnv envs use `env.get_language_instruction()`;
      LIBERO envs use `env.language_instruction`; a Task can return a
      static string or pull from `bddl_file` metadata. If absent, the
      Adapter falls back to calling `env.get_language_instruction()`.
    - `extract_image(env, obs) -> NDArrayU8` — extract a single RGB frame
      from this env's observation dict. If absent, the Adapter walks
      `obs` recursively looking for a (H, W, 3|4) uint8 array.

    Both hooks are intentionally not required by the Protocol — they're
    optional overrides for sims whose API doesn't match the defaults.

    **Optional class attributes for the compatibility-check path**
    (read by Adapter.can_drive when present):

    - `ACTION_SPACE_HINT: Literal["eef_7dof", "joint_target", "custom"]` —
      a coarse, conservative tag for what the env expects. `"custom"` means
      "don't try to auto-check; the Task is shipping with a sibling Adapter
      that knows the wire format." Bimanual / mobile-base / multi-finger
      tasks declare `"custom"` and supply their own Adapter.
    - `canonical_axis_map: dict[str, CanonicalDim]` — see the
      canonical-generalization-axis taxonomy in `analysis.py`.
    """

    @property
    def name(self) -> str: ...

    @property
    def embodiment(self) -> str: ...

    @property
    def n_cells(self) -> int: ...

    def cell_id(self, cell: int) -> CellId: ...

    def build_env(self, cell: int) -> Any:  # gymnasium.Env, but typing-loose to avoid hard dep
        ...

    @property
    def max_episode_steps(self) -> int: ...


@runtime_checkable
class Adapter(Protocol):
    """Bridges a Policy + Task + (cell, seed) to a RolloutResult.

    Adapters know how to translate between the Policy's typed Action and
    the specific simulator's expected action format. `GymAdapter`
    flattens `Action` → 7-d numpy vector and handles SimplerEnv's
    multi-subtask advancement; the same Adapter drives LIBERO without
    modification because LIBERO uses the same 7-DoF EEF action
    convention. A different action space (e.g. 12-DoF joint targets for
    a quadruped like Unitree Go1) requires a sibling Adapter implementation
    — see `MujocoPlaygroundAdapter` and `eval_suite/tasks/unitree_go1.py`.

    **Optional method discovered via getattr:**

    - `can_drive(policy: Policy, task: Task) -> CompatibilityReport` —
      a pre-flight check the sweep driver calls once at trial-zero. The
      Adapter inspects `Task.ACTION_SPACE_HINT` and the Policy's
      output type to decide whether it has any reason to refuse. Return
      `CompatibilityReport(ok=True)` to proceed. Conservative semantics
      (`ok=True` is "no known reason to refuse," not a guarantee — see
      `CompatibilityReport` docstring). If the method is absent, the
      sweep proceeds without a pre-flight check; the runtime
      `_flatten_action` TypeError is the safety net.
    """

    @property
    def name(self) -> str: ...

    def rollout(
        self,
        policy: Policy,
        task: Task,
        cell: int,
        seed: int,
    ) -> RolloutResult: ...


@runtime_checkable
class Manifest(Protocol):
    """A content-addressed reproducibility record for a single sweep.

    The manifest carries every input that could change the result: code
    SHA, container digest, model checkpoint hash, sim commits, seed list,
    hardware. `run_id` is computed by canonicalizing the manifest and
    SHA256-ing the bytes, so any input change produces a new ID.
    """

    @property
    def run_id(self) -> str: ...

    def to_json(self) -> str: ...

    @classmethod
    def from_json(cls, payload: str) -> Manifest: ...

    def verify(self) -> bool:
        """Recompute run_id from contents and check it matches."""
        ...
