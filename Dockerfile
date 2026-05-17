# Reproducer image for the v0 SimplerEnv stack (RT-1 + Octo on Google Robot
# and WidowX). Pinned to the exact commits the published manifests were
# produced from, so rebuilding the image on any Linux + RTX-class GPU box
# reproduces the headline numbers byte-equivalent.
#
# Quick start (assumes Linux + NVIDIA GPU + nvidia-container-toolkit):
#
#   docker build -t evalsuite .
#
#   # Fetch RT-1 checkpoint (Octo auto-downloads from HF at runtime):
#   mkdir -p checkpoints results
#   docker run --rm -v $PWD/checkpoints:/work/checkpoints evalsuite bash -c '\
#     cd /work/checkpoints && \
#     pip install --quiet gsutil && \
#     PATH=$HOME/.local/bin:$PATH gsutil -m cp -r \
#       gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_tf_trained_for_000400120 .'
#
#   # Run a sweep:
#   docker run --gpus all --rm \
#     -v $PWD/checkpoints:/work/checkpoints \
#     -v $PWD/results:/work/results \
#     evalsuite \
#     python -m eval_suite.cli sweep \
#       --model-family rt1 \
#       --rt1-ckpt-path /work/checkpoints/rt_1_tf_trained_for_000400120 \
#       --task google_robot_pick_coke_can \
#       --trials 20 \
#       --output-dir /work/results/run \
#       --videos-dir /work/results/run/videos
#
#   # Browse results in Jupyter:
#   docker run --gpus all --rm -p 8888:8888 \
#     -v $PWD/results:/work/results \
#     -v $PWD/takehome:/work/takehome \
#     -e EVAL_SWEEP_DIR=/work/results/run \
#     evalsuite \
#     jupyter lab --no-browser --ip=0.0.0.0 --allow-root --notebook-dir=/work/takehome
#
#   # Or run the submission portal:
#   docker run --rm -p 8000:8000 evalsuite \
#     uvicorn eval_suite.portal.app:app --host 0.0.0.0 --port 8000
#
# Checkpoints are not baked into the image (large + license-restricted on
# redistribution); the runtime mount path above is the supported pattern.
# MuJoCo Playground / Unitree Go1 (v0) lives in a separate venv because
# numpy<2 + sapien conflicts with JAX 0.10's numpy>=2 — see CLAUDE.md.

ARG CUDA_TAG=12.4.1-cudnn-devel-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=UTC \
    LANG=C.UTF-8

# System deps. python3.11 + venv, ffmpeg for mediapy mp4 encoding, vulkan
# loader so SAPIEN can find a driver (NVIDIA's proprietary ICD is provided
# by the container toolkit at run time via --gpus all), and a C++17 toolchain
# + cmake + python headers because ManiSkill2's ruckig motion-control
# dependency compiles from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        build-essential cmake \
        git ffmpeg unzip wget curl ca-certificates \
        libvulkan1 vulkan-tools libegl1 libglvnd0 libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3

ENV VENV=/opt/venv
RUN python3.11 -m venv $VENV
ENV PATH=$VENV/bin:$PATH

WORKDIR /work

# Step 1: numpy pin BEFORE anything else (sapien/pinocchio IK). `setuptools<81`
# MUST be quoted — unquoted, the shell parses `<` as input redirection.
RUN pip install --upgrade pip 'setuptools<81' wheel \
    && pip install numpy==1.24.4

# Step 2: SimplerEnv + ManiSkill2_real2sim at the published commits.
ARG SIMPLER_ENV_COMMIT=06accaca93535902d408da4855f21cece12bceb7
ARG MANISKILL2_REAL2SIM_COMMIT=ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682
ARG OCTO_COMMIT=653c54acde686fde619855f2eac0dd6edad7116b

RUN git clone https://github.com/simpler-env/SimplerEnv.git /opt/simpler-env \
    && cd /opt/simpler-env \
    && git checkout ${SIMPLER_ENV_COMMIT} \
    && git submodule update --init --recursive \
    && cd ManiSkill2_real2sim \
    && git checkout ${MANISKILL2_REAL2SIM_COMMIT} \
    && cd .. \
    && pip install -e ./ManiSkill2_real2sim \
    && pip install -e .

# Step 3: TF + JAX + Octo + dependency pin set.
RUN pip install tensorflow==2.15.0 \
    && pip install -r /opt/simpler-env/requirements_full_install.txt \
    && pip install "tensorflow[and-cuda]==2.15.1" \
    && pip install git+https://github.com/nathanrooy/simulated-annealing \
    && pip install "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    && git clone https://github.com/octo-models/octo /opt/octo \
    && cd /opt/octo && git checkout ${OCTO_COMMIT} && pip install -e . \
    && pip install \
        "transformers==4.34.1" \
        "chex==0.1.85" \
        "optax==0.1.5" \
        "opencv-python<4.10" \
        "setuptools<81" \
        "mediapy==1.2.0" \
        "jupyterlab"

# Step 4: eval-suite split into two distributions sharing the `eval_suite.*`
# PEP 420 namespace.
#   - eval-suite-core    — substrate (contracts, manifest, sweep, portal,
#                          conformance, calibration registry as package data)
#   - eval-suite-stdlib  — in-tree reference plugins (tasks/policies/adapters)
# Third-party plugins depend on -core only; -stdlib is the bundled reference.
COPY pyproject.toml README.md /work/eval-suite/
COPY packages /work/eval-suite/packages
COPY tests /work/eval-suite/tests
COPY scripts /work/eval-suite/scripts
COPY takehome /work/takehome
RUN pip install -e "/work/eval-suite/packages/eval-suite-core[portal]" \
                -e "/work/eval-suite/packages/eval-suite-stdlib"

# Step 5: sanity import on build — verify the substrate imports, the canonical
# GymAdapter is loadable, and signing + the v0 canonical taxonomy are wired.
RUN python -c "import eval_suite, simpler_env; \
    from eval_suite.policies.simpler_env import SimplerEnvPolicy; \
    from eval_suite.tasks.simpler_env import GoogleRobotPickCokeCan, WidowXSpoonOnTowel; \
    from eval_suite.adapters import GymAdapter; \
    from eval_suite.signing import generate_keypair; \
    from eval_suite._types import CanonicalDim; \
    print('eval-suite import OK; google_robot cells=', GoogleRobotPickCokeCan().n_cells)"

WORKDIR /work
# Default: print sweep CLI help. Override with any of the example invocations
# at the top of this file.
CMD ["python", "-m", "eval_suite.cli", "sweep", "--help"]
