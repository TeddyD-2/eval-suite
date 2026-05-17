#!/usr/bin/env bash
# v0 full sweep — N=20 per cell on Google Robot's 24 base cells
# plus 5 v0 paraphrase cells (3 in-distribution paraphrases + 2 cross-
# embodiment boundary cells), plus 1 WidowX cell × 2 models.
#
# Total per-Google-Robot-model: 29 cells × 20 = 580 trials. Total
# WidowX: 20 trials. Two models. Plus Go1 (separate venv, separate
# script). Estimated wall: ~3.5h.
#
# Structured as one process per (model, task) — TF and JAX don't cohabit
# cleanly, and a crash in one combo shouldn't kill the others. Each
# process amortizes the model load + compile across its trials.
#
# Spot-check progress: `tail -f results/sweep_<...>/trials.csv`
#                  or  `tail -f results/sweep_<...>/sweep.log`

set -uo pipefail  # no -e: keep going if one sweep crashes
shopt -s lastpipe

ROOT=/home/teddy/eval-suite
VENV=/home/teddy/simpler-env/.venv
RT1_CONVERGED=/home/teddy/simpler-env/checkpoints/rt_1_tf_trained_for_000400120
RT1_X=/home/teddy/simpler-env/checkpoints/rt_1_x_tf_trained_for_002272480_step
RESULTS_BASE="$ROOT/results"
STAMP=$(date -u +"%Y%m%dT%H%M%SZ")
RUN_DIR="$RESULTS_BASE/sweep_$STAMP"
mkdir -p "$RUN_DIR"

# shellcheck source=/dev/null
source "$VENV/bin/activate"
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

cd "$ROOT"

# Re-install in case the editable install lost track between sessions.
pip install --quiet -e packages/eval-suite-core -e packages/eval-suite-stdlib >/dev/null 2>&1 || true

echo "[$(date -u +%FT%TZ)] full sweep starting → $RUN_DIR" | tee -a "$RUN_DIR/sweep.log"
START_S=$(date +%s)

run_sweep() {
    local label="$1"; shift
    local cell_out="$RUN_DIR/$label"
    local videos_out="$cell_out/videos"
    mkdir -p "$cell_out" "$videos_out"
    echo "[$(date -u +%FT%TZ)] >>> $label" | tee -a "$RUN_DIR/sweep.log"
    # Pipe stdout+stderr through tee so we get real-time progress in the
    # log and on the tmux pane. The python -u keeps line-buffering.
    python -u -m eval_suite.cli sweep \
        --output-dir "$cell_out" \
        --videos-dir "$videos_out" \
        "$@" 2>&1 | tee -a "$cell_out/sweep.log" "$RUN_DIR/sweep.log"
    local rc=${PIPESTATUS[0]}
    echo "[$(date -u +%FT%TZ)] <<< $label (rc=$rc)" | tee -a "$RUN_DIR/sweep.log"
    return $rc
}

# 1. RT-1-Converged on Google Robot pick coke can (29 cells × N=20 — incl. v0 paraphrase axis)
run_sweep "rt1_google_robot_pick_coke_can" \
    --model-family rt1 \
    --rt1-ckpt-path "$RT1_CONVERGED" \
    --task google_robot_pick_coke_can \
    --trials 20 \
    --calibration-tier C \
    --notes "v0 sweep: RT-1-Converged on Google Robot, 24-cell VA grid + 5 paraphrase cells (3 in-distribution + 2 cross-embodiment boundary)"

# 2. Octo-base on Google Robot pick coke can (29 cells × N=20 — incl. v0 paraphrase axis)
run_sweep "octo_google_robot_pick_coke_can" \
    --model-family octo \
    --octo-model-type octo-base \
    --task google_robot_pick_coke_can \
    --trials 20 \
    --calibration-tier B \
    --calibration-source "SimplerEnv paper Table 3 (Octo on Google Robot pick coke can)" \
    --notes "v0 sweep: Octo-base on Google Robot, 24-cell VA grid + 5 paraphrase cells (3 in-distribution + 2 cross-embodiment boundary); calibration tier-B on baseline subset"

# 3. RT-1-X on WidowX put spoon on towel (1 cell × N=20)
run_sweep "rt1x_widowx_spoon_on_towel" \
    --model-family rt1 \
    --rt1-ckpt-path "$RT1_X" \
    --task widowx_spoon_on_towel \
    --trials 20 \
    --calibration-tier C \
    --notes "v0 platform validation: RT-1-X on WidowX, clean conditions only"

# 4. Octo-base on WidowX put spoon on towel (1 cell × N=20)
run_sweep "octo_widowx_spoon_on_towel" \
    --model-family octo \
    --octo-model-type octo-base \
    --task widowx_spoon_on_towel \
    --trials 20 \
    --calibration-tier C \
    --notes "v0 platform validation: Octo-base on WidowX, clean conditions only"

END_S=$(date +%s)
ELAPSED=$((END_S - START_S))
echo "[$(date -u +%FT%TZ)] all sweeps done in ${ELAPSED}s ($(printf '%dh%02dm' $((ELAPSED/3600)) $(((ELAPSED%3600)/60))))" | tee -a "$RUN_DIR/sweep.log"
echo "Per-sweep manifests:" | tee -a "$RUN_DIR/sweep.log"
ls -la "$RUN_DIR"/*/manifest.json 2>&1 | tee -a "$RUN_DIR/sweep.log"
