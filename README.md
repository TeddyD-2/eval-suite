# eval-suite

[![CI](https://github.com/TeddyD-2/eval-suite/actions/workflows/ci.yml/badge.svg)](https://github.com/TeddyD-2/eval-suite/actions/workflows/ci.yml)
[![docker-build](https://github.com/TeddyD-2/eval-suite/actions/workflows/docker.yml/badge.svg)](https://github.com/TeddyD-2/eval-suite/actions/workflows/docker.yml)

The point of an evaluation suite is to tell you whether a robot model will help a robot complete real tasks. The way the field reports results right now does not do that. Every paper picks its own benchmark, runs ten or twenty trials, prints a single average success rate with no confidence interval, and asks the reader to believe it. Two models that score the same on average can fail in completely different ways — one collapses under dim lighting, the other on cluttered tables — and the single number hides the difference. Customers cannot tell which model will work in their specific conditions; researchers cannot tell which paper's numbers are comparable to which other paper's; developers cannot tell which axis of their model to improve.

This project is an attempt at the reporting layer everyone could agree on. It starts as the reporting contract: a profile of success rates broken down by condition, with confidence intervals on each; a worst-condition headline that ranks balanced models above spiky ones; a manifest that records every input — code version, simulator version, model checkpoint, seeds, hardware — so the same run reproduces byte-for-byte on a different machine; and a calibration tier that says, honestly, how much real-world data backs the simulator number. The same shape of output regardless of which robot, which simulator, which model.

It's also growing into a benchmark in its own right. The contract sits on top of simulators that already exist — SimplerEnv, LIBERO, MuJoCo Playground, eventually RoboCasa and ManiSkill3 and Isaac Lab — and that's the v0 substrate. The next step, prototyped now, is generating new tasks directly from 3D Gaussian-splat scans of real environments: a partner scans a factory floor or a kitchen, the suite turns the splat into a parametric cell grid, and that becomes a new task the same way SimplerEnv's pick-coke-can is a task. Once that pipeline is real, the suite isn't just a reporting layer on top of other people's benchmarks — it's the benchmark whose tasks come from real deployment environments and grow with the user base.

Where this sits in the robotics-simulation stack: at the layer above. SimplerEnv, LIBERO, RoboCasa, ManiSkill, Isaac Lab, MuJoCo Playground all sit at the simulator-and-tasks layer. They give you environments to roll out policies in. This project sits on top of any of them. You write a small adapter for a simulator once, and from then on every result that simulator produces comes out in the same reporting shape as every other simulator's results. The codebase ships working adapters for three simulators (SimplerEnv, LIBERO, MuJoCo Playground) to demonstrate the shape works, but the contract is what matters; the adapters are the proof.

## How somebody uses it

The simplest path: clone the repo, build the Docker image, run a sweep, look at the results.

```bash
git clone https://github.com/TeddyD-2/eval-suite.git
cd eval-suite
docker build -t evalsuite .
```

To run a real sweep you need a robot model checkpoint. The repo evaluates two reference models out of the box. RT-1 lives in a public Google Cloud bucket; Octo downloads from Hugging Face the first time it runs. Fetch RT-1 once:

```bash
mkdir -p checkpoints results
docker run --rm -v $PWD/checkpoints:/work/checkpoints evalsuite bash -c '\
  cd /work/checkpoints && \
  pip install --quiet gsutil && \
  PATH=$HOME/.local/bin:$PATH gsutil -m cp -r \
    gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_tf_trained_for_000400120 .'
```

Then run a sweep. This evaluates RT-1 on a pick-the-coke-can task across twenty-nine condition cells (different orientations, lighting, backgrounds, distractors, table textures, and language phrasings), twenty trials per cell. Takes about half an hour on an RTX 3090:

```bash
docker run --gpus all --rm \
  -v $PWD/checkpoints:/work/checkpoints \
  -v $PWD/results:/work/results \
  evalsuite \
  python -m eval_suite.cli sweep \
    --policy simpler_env \
    --policy-arg family=rt1 \
    --policy-arg policy_setup=google_robot \
    --policy-arg ckpt_path=/work/checkpoints/rt_1_tf_trained_for_000400120 \
    --task google_robot_pick_coke_can \
    --trials 20 \
    --output-dir /work/results/run \
    --videos-dir /work/results/run/videos
```

When the sweep finishes, `/work/results/run/` contains a sealed manifest with a content-addressed run identifier, a CSV with one row per trial, and one directory per rollout containing a video of the rollout, a numpy array of every step's action and reward, and a metadata file. To browse it, start Jupyter against the results directory:

```bash
docker run --gpus all --rm -p 8888:8888 \
  -v $PWD/results:/work/results \
  -v $PWD/takehome:/work/takehome \
  -e EVAL_SWEEP_DIR=/work/results/run \
  evalsuite \
  jupyter lab --no-browser --ip=0.0.0.0 --allow-root --notebook-dir=/work/takehome
```

Open `profile.ipynb`, run all cells. You get a bar chart of success rate with confidence intervals for every condition cell, a chart of per-axis averages with the worst axis flagged, a comparison against a published real-robot number, and a few sample success and failure rollouts inline.

![Four-bucket summary across five sweeps: each panel is one (model, robot) pair, bars are mean success rate per canonical dimension (wording / visuals / physics / robot type) with 95% Wilson CIs, the weakest measured dimension is red, gray bars are "not measured" for honesty. The RT-1 panel is the headline: 0.82 on visuals, 0.27 on wording — same scene, only the instruction string changed.](takehome/media/canonical_summary.png)

Other people can use the same image to submit their own results. The Dockerfile bakes in a submission portal — a small web service that accepts signed manifests, lists them, and renders each one as a browsable page. Start it:

```bash
docker run --rm -p 8000:8000 evalsuite \
  uvicorn eval_suite.portal.app:app --host 0.0.0.0 --port 8000
```

Browse to `http://localhost:8000/`. The submission flow signs a manifest with an Ed25519 key, posts it to the portal, and the portal stores it under the manifest's run ID. Two submitters posting the same run ID with different keys is the cross-corroboration signal: neither could have forged the number without the other lying identically.

## How somebody else extends it

Plugging in a new task, a new robot, or a new simulator does not require editing the project's source. Anyone can publish a pip package that ships a `Task`, a `Policy`, or an `Adapter`, register it through the standard Python entry-points mechanism, and the in-tree CLI picks it up automatically. A reference walkthrough lives in `docs/plugin_authoring.md`. A complete worked example lives in `examples/external_plugin_demo/`.

The four contracts are small. A `Task` describes a list of condition cells and knows how to build a simulator environment for each. A `Policy` takes an observation and returns an action. An `Adapter` runs a policy through a task and produces a rollout result. A `Manifest` records what produced the run. They are defined as Python protocols, meaning any class with the right method signatures satisfies them — no inheritance, no base classes to extend. The full surface is in `eval_suite/contracts.py`.

A conformance kit ships in `eval_suite/conformance/`. Plugin authors call it from their own pytest suites to check that their `Task` and `Policy` and `Adapter` actually satisfy the protocol and round-trip cleanly through a sweep. It's the floor for what a published plugin should pass.

![Plugin registry page from the bundled portal: third-party tasks/policies/adapters registered via entry-points show up alongside the in-tree reference ones, indistinguishable by mechanism. `eval-suite-plugin-demo` is the external example; everything else is in-tree.](takehome/media/registry.png)

## What's actually in the repo right now

Everything below is what "v0" means in `takehome/EXTENSION.md` — the substrate at submission time. The line that matters is the one between "everything shipped so far" (v0) and "everything next" (v1).

Twenty-nine condition cells of Google Robot pick-coke-can plus one platform-validation cell on WidowX, evaluated for two reference models (Octo and RT-1) at twenty trials per cell, with sealed manifests and videos. A twelve-cell legged-locomotion sweep on Unitree Go1 through MuJoCo Playground, demonstrating that the contract absorbs a different action space (joint targets instead of end-effector deltas). A LIBERO task demonstrating that the contract absorbs a different simulator. A submission portal with signed-manifest enforcement and a browsable HTML UI. A calibration tier system with one published real-robot reference comparison. Continuous-integration tests that verify the manifest format, hash determinism, protocol satisfaction, and end-to-end sweep shape on every push.

## What's still left to implement

Real-robot calibration is the largest piece. The current contract supports four tiers — no real data, one published reference, paired trials in matching cells, profile-wide Pearson correlation — but the only tier above "no data" that ships is the single published-reference comparison. The remaining tiers wait on deployment telemetry from somebody running these robots in production and an HTTP endpoint that ingests it; both are scoped out in `takehome/EXTENSION.md`.

Enforced sandboxing for submitted policies is documented but not yet enforced. Right now a submitter could in principle ship a policy that consults a remote service during evaluation. The intermediate step that ships now is an Ed25519 submitter attestation, which costs the submitter their reputation if the attestation turns out to be false. Hard sandboxing — gVisor or Firecracker, no network egress, audit log — is in the roadmap.

Trial counts are sized for shape discrimination, not for per-cell significance. Twenty trials per cell is what fits on the single RTX 3090 this was prototyped on while leaving room to iterate on the rest of the system. It's enough to tell that one model's worst axis is lighting and another's is distractors; it is not enough to claim that a five-point gap on one cell between two models is real. Scaling to a hundred trials per cell requires multi-GPU sweep parallelism and a GPU-parallel simulator backend (ManiSkill3); both are named in the roadmap.

Generating tasks from real-world 3D Gaussian-splat scans is the bigger missing piece. v0 runs on existing benchmark tasks; v1 ships the splat ingest pipeline as a stdlib plugin behind the `[splat]` extra — `python -m eval_suite.ingest.splat ingest <ply> --scene-metadata <json> --scene-extractions <json>` turns a splat checkpoint into a composed MJCF with declarative `NamedRegion` + `SpawnPoint` + `ExtractedBody` annotations and a `scene_transform` for mesh-to-MuJoCo alignment. The accompanying `ParametricSplatTask` couples that scene with a declarative success predicate (`RobotReachedRegion`, `MaintainedClearance`, `Survived`) bound into the manifest at schema 0.3.0, so a factory engineer captures their warehouse, picks a predicate from a library, and runs a sweep — no new Task subclass, no manifest schema editing. The v1 demo asset under `assets/tnt_truck_splat/` is the Tanks-and-Temples "Truck" scene (CC-BY 4.0, commercial-friendly per EXTENSION.md §6's partner-pilot path); the 9-cell sweep evaluates "did Go1 reach behind the truck under variant lighting/camera." v2 turns this into a contributor-driven library of real-deployment tasks.

Cross-task aggregation under the canonical generalization dimensions (combining a "language" score across multiple tasks into one cross-task language number) is deferred. The current per-task profile is fine; pooling cells across different tasks requires either a deployment-relevance weighting or a stratified bootstrap that respects the per-task structure, and either is a meaningful design choice that should be made with care rather than retrofitted.

A trained Unitree Go1 locomotion policy is deferred. The current Go1 sweep runs a random policy as the substrate proof. Wiring up one of the published trained checkpoints (DeepMimic, RoboPianist, mujoco_playground's reference) is mechanical work, not architectural.

More tasks across more embodiments — move-near, open-drawer, the full WidowX variant grid, paraphrase axes on LIBERO and Go1 — are mechanical and additive. Each is one new `Task` against the existing adapter. The substrate already absorbs them.

The full roadmap with rationale, including the philosophy and version-by-version detail of what's shipped, lives in `takehome/EXTENSION.md`.

## Repository layout

The repo ships two distributions that share the `eval_suite.*` import namespace via PEP 420. The split makes the substrate-plugin boundary a real package boundary: a third-party plugin depends on `eval-suite-core` only — never on the bundled reference plugins.

```
eval-suite/
├── README.md                                     # this file
├── CLAUDE.md                                     # operating guide: venvs, commands, editing rules, pitfalls
├── Dockerfile                                    # reproducer image; the supported way to run a real sweep
├── pyproject.toml                                # repo-level tooling config (mypy, ruff, pytest)
│
├── packages/
│   ├── eval-suite-core/                          # SUBSTRATE — what plugins depend on
│   │   ├── pyproject.toml
│   │   └── src/eval_suite/
│   │       ├── contracts.py                      # Policy / Task / Adapter / Manifest protocols
│   │       ├── manifest.py                       # content-addressed manifest with sign / verify
│   │       ├── statistics.py                     # Wilson CIs, worst-axis ranking
│   │       ├── sweep.py                          # the driver: model loads once, iterate (cell × seed)
│   │       ├── analysis.py                       # sweep dir → profile + canonical dims
│   │       ├── cli.py                            # python -m eval_suite.cli ...
│   │       ├── signing.py                        # Ed25519 submitter signatures
│   │       ├── registry.py                       # entry-points-based plugin discovery
│   │       ├── conformance/                      # library plugin authors call from their pytest
│   │       ├── portal/                           # FastAPI submission portal + browsable UI
│   │       └── calibration/real_perf.json        # published real-robot reference data (package data)
│   │
│   └── eval-suite-stdlib/                        # IN-TREE PLUGINS — structurally identical to a 3rd-party plugin
│       ├── pyproject.toml                        #   declares the `eval_suite.{tasks,policies,adapters}`
│       └── src/eval_suite/                       #   entry-points the in-tree code is registered under
│           ├── tasks/                            # GoogleRobotPickCokeCan, WidowXSpoonOnTowel, LIBEROSpatial,
│           │                                     #   UnitreeGo1Joystick, NamaqualandScanTask, ParametricSplatTask,
│           │                                     #   MockTask; declarative success-predicate registry
│           ├── policies/                         # SimplerEnvPolicy (Octo + RT-1), RandomLocomotionPolicy, MockPolicy
│           ├── adapters/                         # GymAdapter, MujocoPlaygroundAdapter
│           └── ingest/                           # asset-ingest plugins behind optional extras
│               └── splat/                        #   Gaussian-splat → MJCF pipeline (`[splat]` extra: trimesh, manifold3d, ...)
│
├── takehome/                                     # EXTENSION.md (design doc) + profile.ipynb + reviewer media
├── tests/                                        # contract tests; ruff + mypy --strict + pytest in CI
├── examples/external_plugin_demo/                # sibling pip package — depends on eval-suite-core ONLY
├── scripts/                                      # run_full_sweep.sh, run_go1_sweep.py, bench_amortization.py
├── assets/                                       # source + converted assets: namaqualand_scan (USD), tnt_truck_splat (GS)
├── docs/                                         # plugin_authoring.md, curated rollout videos, portal HTML snapshot
├── manifests/                                    # archived manifests from prior sweeps for verify() tests
├── results/                                      # sweep output dirs (gitignored; shape documented in CLAUDE.md)
├── submissions/                                  # portal submission store (gitignored; written by the portal at runtime)
└── archive/                                      # compressed sweep tarballs for long-term storage
```

**Install for development:** `pip install -e packages/eval-suite-core[dev] -e packages/eval-suite-stdlib`.

## Hardware

The Dockerfile targets Linux with an NVIDIA GPU (RTX-class or better) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). The image was built and tested on RunPod with an RTX 3090. The CI smoke tests run on hosted runners without a GPU; the real evaluation runs in the container on your own GPU box.

The MuJoCo Playground / Unitree Go1 sweep runs in a separate Python environment, not the Docker image. Its dependencies (JAX 0.10, numpy ≥ 2, mujoco-playground) conflict with the SimplerEnv stack (numpy < 2, SAPIEN). See `CLAUDE.md` for the two-environment setup; a unified Go1 image is named as a follow-up in the roadmap.

## License

MIT.
