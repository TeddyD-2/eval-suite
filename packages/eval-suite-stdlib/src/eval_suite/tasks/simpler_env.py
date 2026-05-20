"""SimplerEnv-backed Task implementations.

**In plain words.** The two reference tasks every robotics paper
cites: `GoogleRobotPickCokeCan` (29 condition cells covering
orientation, lighting, background, distractors, table textures, and
paraphrased instructions) and `WidowXSpoonOnTowel` (one cell, used
to prove the suite handles a different embodiment without code
changes). These are how the suite's headline numbers are produced.


`GoogleRobotPickCokeCan` exposes the full Variant Aggregation cell grid
for the Google Robot pick-coke-can task family. Cells are defined
declaratively (see `_CELL_SPECS` below) and translated to SimplerEnv
gym envs at `build_env` time.

`WidowXSpoonOnTowel` is the platform-validation Task: one clean-conditions
cell only. The point isn't a sweep here — it's that the same Adapter +
Policy pipeline drives a different embodiment with no code changes.

Both classes import SimplerEnv lazily so `eval_suite.tasks.mock` can be
imported on a CI runner without GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .._types import CanonicalDim, CellId


@dataclass(frozen=True)
class _CellSpec:
    """Declarative spec for one cell in the variant grid.

    `env_name` is the underlying ManiSkill2 env id; `scene_name` is the
    SimplerEnv scene id; `env_kwargs` are passed through `simpler_env.make`
    to `gym.make`. `axes` is the variant axis-level map recorded in the
    manifest.

    `instruction_override` (v0) — when non-None, the Task's
    `instruction_for(env)` hook returns this string instead of letting
    SimplerEnv pick the env's default instruction. Used by the paraphrase
    axis to test language robustness on otherwise-identical cells.
    """

    env_name: str
    scene_name: str
    env_kwargs: dict[str, Any]
    axes: dict[str, str]
    instruction_override: str | None = field(default=None)


# ---------- Google Robot pick coke can cell catalog -----------------------
#
# Five axes (orientation, lighting, background, distractor, table_texture).
# The "vary one axis from baseline" pattern of the SimplerEnv reference
# scripts gives us a manageable cell count. Baseline cells exist for all
# three orientations; each non-orientation axis adds variants on top.

_BASELINE_SCENE = "google_pick_coke_can_1_v4"
_ENV_CORE = "GraspSingleOpenedCokeCanInScene-v0"
_ENV_DISTRACTOR = "GraspSingleOpenedCokeCanDistractorInScene-v0"

_ORIENTATIONS: dict[str, dict[str, Any]] = {
    "horizontal": {"lr_switch": True},
    "vertical": {"laid_vertically": True},
    "upright": {"upright": True},
}


def _build_google_robot_cell_specs() -> list[_CellSpec]:
    cells: list[_CellSpec] = []
    for orient_name, orient_kwargs in _ORIENTATIONS.items():
        # baseline: orientation only
        cells.append(_CellSpec(
            env_name=_ENV_CORE,
            scene_name=_BASELINE_SCENE,
            env_kwargs={**orient_kwargs},
            axes={"orientation": orient_name, "lighting": "base", "background": "base",
                  "distractor": "base", "table_texture": "base"},
        ))
        # lighting axis: darker, brighter
        for lighting_level, lighting_kwarg in (("darker", "slightly_darker_lighting"),
                                                ("brighter", "slightly_brighter_lighting")):
            cells.append(_CellSpec(
                env_name=_ENV_CORE,
                scene_name=_BASELINE_SCENE,
                env_kwargs={**orient_kwargs, lighting_kwarg: True},
                axes={"orientation": orient_name, "lighting": lighting_level, "background": "base",
                      "distractor": "base", "table_texture": "base"},
            ))
        # background axis: alt_1, alt_2
        for bg_level, bg_scene in (("alt_1", "google_pick_coke_can_1_v4_alt_background"),
                                    ("alt_2", "google_pick_coke_can_1_v4_alt_background_2")):
            cells.append(_CellSpec(
                env_name=_ENV_CORE,
                scene_name=bg_scene,
                env_kwargs={**orient_kwargs},
                axes={"orientation": orient_name, "lighting": "base", "background": bg_level,
                      "distractor": "base", "table_texture": "base"},
            ))
        # distractor axis: more
        cells.append(_CellSpec(
            env_name=_ENV_DISTRACTOR,
            scene_name=_BASELINE_SCENE,
            env_kwargs={**orient_kwargs, "distractor_config": "more"},
            axes={"orientation": orient_name, "lighting": "base", "background": "base",
                  "distractor": "more", "table_texture": "base"},
        ))
        # table_texture axis: cabinet1, cabinet2
        for table_level, table_scene in (
            ("cabinet1", "Baked_sc1_staging_objaverse_cabinet1_h870"),
            ("cabinet2", "Baked_sc1_staging_objaverse_cabinet2_h870"),
        ):
            cells.append(_CellSpec(
                env_name=_ENV_CORE,
                scene_name=table_scene,
                env_kwargs={**orient_kwargs},
                axes={"orientation": orient_name, "lighting": "base", "background": "base",
                      "distractor": "base", "table_texture": table_level},
            ))
    return cells


# v0 paraphrase axis: tests language robustness on the upright-baseline
# cell. Five levels — three in-distribution paraphrases + two deliberately
# out-of-distribution "boundary" instructions that ask the arm to do
# something it physically can't (locomotion-style verbs). The success
# criterion is unchanged (was-the-can-picked-up); the boundary cells are
# expected to score ~0 across all models. They serve as the rhetorical
# "these models aren't actually general" demonstration in the v0
# canonical-profile section.
#
# The cells share the otherwise-baseline grid position so they isolate
# the language shift (orientation=upright, all-else=base). They add a
# new `paraphrase` axis whose canonical mapping is `language`.

_PARAPHRASES: list[tuple[str, str]] = [
    ("base",         "pick up the coke can"),                   # baseline language
    ("synonym",      "grab the soda"),                          # synonym swap
    ("descriptive",  "lift the red can on the table"),          # descriptive ref
    ("ood_walk",     "walk forward to the can"),                # boundary: locomotion verb
    ("ood_stand",    "stand up and grab the can"),              # boundary: legged verb
]


def _build_paraphrase_cells() -> list[_CellSpec]:
    """Five paraphrase cells on the upright baseline.

    Three in-distribution paraphrases plus two cross-embodiment boundary
    cells. The boundary cells let v0 show the failure mode (videos of
    arms confusedly trying to grasp when told to "walk forward") alongside
    the per-cell success rates.
    """
    upright_kwargs = {"upright": True}
    cells: list[_CellSpec] = []
    for paraphrase_name, instruction in _PARAPHRASES:
        cells.append(_CellSpec(
            env_name=_ENV_CORE,
            scene_name=_BASELINE_SCENE,
            env_kwargs={**upright_kwargs},
            axes={"orientation": "upright", "lighting": "base", "background": "base",
                  "distractor": "base", "table_texture": "base", "paraphrase": paraphrase_name},
            instruction_override=instruction,
        ))
    return cells


_GOOGLE_ROBOT_CELL_SPECS = _build_google_robot_cell_specs() + _build_paraphrase_cells()


class GoogleRobotPickCokeCan:
    """Google Robot pick-coke-can task, full Variant Aggregation grid + v0 paraphrase axis.

    24 base cells (3 orientations × 8 vary-one-axis variants) plus 5
    paraphrase cells on the upright baseline (3 in-distribution
    paraphrases + 2 cross-embodiment boundary instructions) = 29 cells.
    """

    # v0 canonical-axis taxonomy. Each cell-level axis maps to one of
    # {language, visuals, physics, embodiment}. Read by
    # `analysis.canonical_profile_for_sweep`. Decentralized — each Task
    # declares its own mapping; the closed-enum keeps the dim names from
    # drifting.
    canonical_axis_map: dict[str, CanonicalDim] = {
        "orientation": "physics",      # pose-on-table is physical state
        "lighting": "visuals",
        "background": "visuals",
        "distractor": "visuals",       # adds objects in the scene
        "table_texture": "visuals",
        "paraphrase": "language",
    }

    @property
    def name(self) -> str:
        return "pick_coke_can"

    @property
    def embodiment(self) -> str:
        return "google_robot"

    @property
    def n_cells(self) -> int:
        return len(_GOOGLE_ROBOT_CELL_SPECS)

    def cell_id(self, cell: int) -> CellId:
        spec = _GOOGLE_ROBOT_CELL_SPECS[cell]
        return CellId(embodiment=self.embodiment, task=self.name, axes=spec.axes)

    def cell_spec(self, cell: int) -> _CellSpec:
        return _GOOGLE_ROBOT_CELL_SPECS[cell]

    def build_env(self, cell: int) -> Any:
        # Import locally so the CI/contract path doesn't pull SimplerEnv.
        import gymnasium as gym
        import mani_skill2_real2sim.envs  # noqa: F401
        spec = _GOOGLE_ROBOT_CELL_SPECS[cell]
        kwargs = {
            "obs_mode": "rgbd",
            "prepackaged_config": True,
            "robot": "google_robot_static",
            "scene_name": spec.scene_name,
            "control_freq": 3,
            "sim_freq": 513,
            "max_episode_steps": self.max_episode_steps,
            **spec.env_kwargs,
        }
        env = gym.make(spec.env_name, **kwargs)
        if spec.instruction_override is not None:
            # The GymAdapter calls Task.instruction_for(env) on every
            # step; we stash the override on the env object so the hook
            # can read it back without re-plumbing through the Adapter.
            env._eval_suite_instruction_override = spec.instruction_override
        return env

    def instruction_for(self, env: Any) -> str:
        """Return the paraphrase override if set; else SimplerEnv's default."""
        override = getattr(env, "_eval_suite_instruction_override", None)
        if override is not None:
            return str(override)
        getter = getattr(env, "get_language_instruction", None)
        if callable(getter):
            try:
                return str(getter()) or ""
            except Exception:
                return ""
        return ""

    @property
    def max_episode_steps(self) -> int:
        return 80


# ---------- WidowX Bridge — platform validation, single cell --------------


class WidowXSpoonOnTowel:
    """WidowX put-spoon-on-towel, single clean-conditions cell.

    WidowX/Bridge envs don't ship the full Variant Aggregation grid; the
    suite uses this Task as platform-validation only. Real value: same
    Adapter, same Policy interface, different embodiment.
    """

    # v0: single "condition" axis is structural (clean / would-be-noisy);
    # it doesn't vary in v0 since only the clean cell ships. The
    # canonical mapping is still declared so the analysis layer sees the
    # task contributes nothing to any canonical dim at v0 — except for
    # cross-task aggregation (v0), where the per-task overall mean
    # becomes one data point under "embodiment" by way of being a
    # different embodiment from Google Robot.
    canonical_axis_map: dict[str, CanonicalDim] = {
        # "condition" is the only axis; degenerate at one level. Map to
        # visuals as a placeholder — it doesn't actually vary, so it
        # contributes zero per-axis-variance to any dim.
        "condition": "visuals",
    }

    @property
    def name(self) -> str:
        return "spoon_on_towel"

    @property
    def embodiment(self) -> str:
        return "widowx"

    @property
    def n_cells(self) -> int:
        return 1

    def cell_id(self, cell: int) -> CellId:
        if cell != 0:
            raise IndexError(cell)
        return CellId(embodiment=self.embodiment, task=self.name, axes={"condition": "clean"})

    def build_env(self, cell: int) -> Any:
        if cell != 0:
            raise IndexError(cell)
        import simpler_env
        return simpler_env.make("widowx_spoon_on_towel")

    @property
    def max_episode_steps(self) -> int:
        return 60
