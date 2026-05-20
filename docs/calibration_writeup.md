# Sim-to-real calibration writeup

This page argues about what the suite's calibration numbers tell us
and where they fail. It does not claim the sim score *is* the real
score; the tier label is the honest claim.

## What `examples/calibration_pickcoke_rt1.py` reports

Running the demo against the v0 RT-1-Converged sweep manifest at
`manifests/rt1_google_robot_pick_coke_can/manifest.json` and the
per-condition real numbers from SimplerEnv paper Table 3, seeded into
`real_perf.json`:

```
Paired cells found: 16
Pearson r (n=16): +0.875  95% bootstrap CI [+0.733, +0.970]
  → tier A reportable
MMRV (sim vs real trajectory, synthesized stand-in): 0.100
```

## What it tells us

The Pearson r of +0.875 over 16 paired cells says **the suite's sim
ordering of RT-1-Converged conditions on Google-Robot-pick-coke-can
agrees with the real-robot ordering at a tight enough level to be
useful as a leading indicator**. A model that does well on the suite's
vertical-with-darker-lighting cell does well in the real-world
vertical-with-darker-lighting condition; a model that's weak on
horizontal-with-more-distractors is weak in the real horizontal-with-
more-distractors. The bootstrap CI doesn't include 0, which is the
minimum bar a "calibration claim" needs to clear.

What it doesn't tell us:

- **It doesn't tell us the absolute number is right.** All 16 paired
  cells show `delta = sim − real ≥ 0` (sim consistently scores higher
  than real, mostly by 3-10pp). That's the "sim-real gap" everyone
  expects — sim is easier than reality. Pearson r measures monotone
  agreement, not calibration. For a deployer, this means: "rank the
  three candidate policies on the suite, the ordering will agree with
  what you'd see in the real lab, but don't bet your warehouse rollout
  on the absolute number."
- **It doesn't tell us the model fails the same way.** A sim-fail
  mode that the real world doesn't share (a SAPIEN-specific physics
  glitch) can produce a sim score the real number doesn't reproduce.
  Pearson r doesn't catch this; per-cell delta analysis does (look at
  the table — uniform positive deltas suggest a global sim-vs-real
  offset, not a discriminative failure).
- **It doesn't generalize across (task, model).** This r is for
  RT-1-Converged on Google Robot pick-coke-can. The same registry
  contains Octo-base on the same task — but Octo's smaller real-sim
  gap (sim 0.298, real 0.293 on the aggregate) doesn't mean Octo's
  per-cell r is also +0.875; it has to be measured.

## Why MMRV is *also* in the substrate

Outcome-level Pearson r tells you the suite's per-cell *rates* track
real per-cell *rates*. It says nothing about whether the policy's
trajectories in sim look anything like its trajectories in real life
— a model that succeeds in sim by exploiting a contact-model glitch
gets a perfect outcome match while doing something nothing like the
real-world rollout. **MMRV** (maximum mean relative velocity error
between paired sim and real trajectories) catches that — it's the
SIMPLER-style metric that says "the *motion* matches, not just the
end-state."

The substrate provides both because they fail differently:

| Failure mode                                       | Outcome Pearson r | MMRV    |
|----------------------------------------------------|-------------------|---------|
| Sim physics matches; policy ranks differently      | catches it        | misses  |
| Sim physics wrong but outcome distribution matches | misses            | catches |
| Per-cell registry has only outcome data            | usable            | n/a     |
| Per-cell registry has paired trajectory data       | usable            | usable  |

A v2 deployment story should use both: outcome r to gate
candidate-policy selection; trajectory MMRV to drive sim-improvement
PRs against the underlying simulator when divergence is large enough
to matter.

## What the synthesized MMRV in the demo means

The demo synthesizes a 12-DoF joint trajectory pair where the sim
follows the real trajectory at 1.10× the velocity. MMRV = 0.10 is
exactly what that should produce — the metric reports a per-step
relative-velocity error of 10%, which is the test for "is the
implementation correct."

The real wire-up that *isn't* synthesized: replace `real_traj` with the
joint-trajectory column from an OXE / LeRobot episode loaded through
`eval_suite.policies.oxe_replay.OXEReplayPolicy`, and replace
`sim_traj` with the per-step `qpos` (or EEF xyz) from a sweep's
`trajectory.npz`. The `OXEReplayPolicy` is in stdlib as of Phase 1;
the calibration substrate (this commit) computes MMRV from any two
same-shape trajectories.

## What the tier label is doing

Per EXTENSION.md §4, a per-(task, model) report can be at tier C
(no real data), B (one published real number), A (≥10 paired cells,
report Pearson r), or A+ (≥100 paired real trials per cell across
≥80% of cells, profile-wide Pearson + flagged divergence cells).

The demo above produces **tier A** for `RT-1-Converged on
google_robot_pick_coke_can` because:

- 16 paired-cell entries exist in `real_perf.json` for this
  (task, model) pair after this commit.
- Pearson r is reported with bootstrap CI.

The next tier-upgrade levers, in order:

1. Seed Octo per-cell numbers from the SimplerEnv paper (we have the
   aggregate at tier B already; per-cell is mechanical).
2. WidowX per-cell numbers from SimplerEnv paper Table 4.
3. Trajectory-level paired data via `OXEReplayPolicy` + a sim sweep
   — pairs accrue into the sidecar; once one (task, model) has ≥20
   paired trajectories the MMRV becomes a real metric, not just a
   correctness check.
4. Partner-lab telemetry ingestion (`POST /calibration`) — listed in
   EXTENSION.md §4 as v2 work; the substrate (this commit + the
   sidecar) is what makes the v2 endpoint a five-line write rather
   than a design exercise.

The architecture choice in this PR: putting the statistics + sidecar
in `eval-suite-core`, not stdlib. Calibration is contract-shaped, not
plugin-shaped. Third-party plugins call the same `pearson_r_with_
bootstrap_ci` everyone else uses.
