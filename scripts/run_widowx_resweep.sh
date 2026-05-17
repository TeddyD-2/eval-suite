#!/usr/bin/env bash
# Re-run the two WidowX combos into the existing v0 sweep dir, after
# the np.bool_ → JSON serialization fix in rollout_data.py.

set -uo pipefail

ROOT=/home/teddy/eval-suite
VENV=/home/teddy/simpler-env/.venv
RT1_X=/home/teddy/simpler-env/checkpoints/rt_1_x_tf_trained_for_002272480_step
RUN_DIR="$ROOT/results/sweep_20260516T072703Z"

source "$VENV/bin/activate"
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

cd "$ROOT"
pip install --quiet -e packages/eval-suite-core -e packages/eval-suite-stdlib >/dev/null 2>&1 || true

echo "[$(date -u +%FT%TZ)] WIDOWX re-sweep starting" | tee -a "$RUN_DIR/sweep.log"

# --- RT-1-X on WidowX ----------------------------------------------------
label=rt1x_widowx_spoon_on_towel
cell_out="$RUN_DIR/$label"
mkdir -p "$cell_out" "$cell_out/videos"
echo "[$(date -u +%FT%TZ)] >>> $label" | tee -a "$RUN_DIR/sweep.log"
python -u -m eval_suite.cli sweep \
    --output-dir "$cell_out" \
    --videos-dir "$cell_out/videos" \
    --model-family rt1 \
    --rt1-ckpt-path "$RT1_X" \
    --task widowx_spoon_on_towel \
    --trials 20 \
    --calibration-tier C \
    --notes "v0 platform validation: RT-1-X on WidowX, clean conditions only" \
    2>&1 | tee -a "$cell_out/sweep.log" "$RUN_DIR/sweep.log"
rc=${PIPESTATUS[0]}
echo "[$(date -u +%FT%TZ)] <<< $label (rc=$rc)" | tee -a "$RUN_DIR/sweep.log"

# --- Octo-base on WidowX -------------------------------------------------
label=octo_widowx_spoon_on_towel
cell_out="$RUN_DIR/$label"
mkdir -p "$cell_out" "$cell_out/videos"
echo "[$(date -u +%FT%TZ)] >>> $label" | tee -a "$RUN_DIR/sweep.log"
python -u -m eval_suite.cli sweep \
    --output-dir "$cell_out" \
    --videos-dir "$cell_out/videos" \
    --model-family octo \
    --octo-model-type octo-base \
    --task widowx_spoon_on_towel \
    --trials 20 \
    --calibration-tier C \
    --notes "v0 platform validation: Octo-base on WidowX, clean conditions only" \
    2>&1 | tee -a "$cell_out/sweep.log" "$RUN_DIR/sweep.log"
rc=${PIPESTATUS[0]}
echo "[$(date -u +%FT%TZ)] <<< $label (rc=$rc)" | tee -a "$RUN_DIR/sweep.log"

echo "[$(date -u +%FT%TZ)] WIDOWX re-sweep done" | tee -a "$RUN_DIR/sweep.log"
