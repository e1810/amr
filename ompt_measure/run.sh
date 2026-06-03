#!/usr/bin/env bash
set -eo pipefail

if [ -f /opt/intel/oneapi/setvars.sh ]; then
    source /opt/intel/oneapi/setvars.sh >/dev/null
fi

set -u

# Increase coarse_n to increase fine_n = coarse_n * 2^max_level.
# The default here is intentionally heavier than the quick examples so the
# per-thread loop time is larger than OpenMP/OMPT overhead.
# The default fine_n is 512; 16 stable diffusion substeps per visible step
# give about the same physical diffusion progress as the original fine_n=128 run.
COARSE_N=${COARSE_N:-64}
PARTS=${PARTS:-4}
MAX_LEVEL=${MAX_LEVEL:-3}
STEPS=${STEPS:-2000}
DIFFUSION=${DIFFUSION:-0.01}
STEP_INTERVAL=${STEP_INTERVAL:-100}
INITIAL_PATTERN=${INITIAL_PATTERN:-checker}

if [ "$INITIAL_PATTERN" = "hotspot" ]; then
    DEFAULT_INITIAL_SCALE=0.12
else
    DEFAULT_INITIAL_SCALE=0.25
fi
INITIAL_SCALE=${INITIAL_SCALE:-$DEFAULT_INITIAL_SCALE}
DIFFUSION_SUBSTEPS=${DIFFUSION_SUBSTEPS:-16}

OUTPUT_ROOT=${OUTPUT_ROOT:-outputs}
OMPT_MODE=measure
CASE_NAME=${CASE_NAME:-$INITIAL_PATTERN}
CASE_DIR=${CASE_DIR:-$OUTPUT_ROOT/$CASE_NAME/$OMPT_MODE}
SVG_DIR=${SVG_DIR:-$CASE_DIR/svg}
TXT_DIR=${TXT_DIR:-$CASE_DIR/txt}

mkdir -p "$CASE_DIR" "$SVG_DIR" "$TXT_DIR"

TIMING_CSV=${TIMING_CSV:-$CASE_DIR/ompt_regions.csv}
TIMING_SVG=${TIMING_SVG:-$CASE_DIR/ompt_timing.svg}
SNAPSHOT_INTERVAL=${SNAPSHOT_INTERVAL:-100}
SNAPSHOT_PREFIX=${SNAPSHOT_PREFIX:-$TXT_DIR/snapshot}

export OMP_TOOL=enabled
export OMP_TOOL_LIBRARIES=${OMP_TOOL_LIBRARIES:-$PWD/ompt_measure/libompt_measure.so}
export AMR_OMPT_OUT=$TIMING_CSV
export AMR_OMPT_EXEC_INTERVAL=$STEP_INTERVAL
export OMP_PROC_BIND=${OMP_PROC_BIND:-true}
export OMP_PLACES=${OMP_PLACES:-cores}

./amr "$COARSE_N" "$PARTS" "$MAX_LEVEL" "$INITIAL_SCALE" "$STEPS" "$DIFFUSION" \
    "$SNAPSHOT_INTERVAL" "$SNAPSHOT_PREFIX" "$INITIAL_PATTERN" \
    "$DIFFUSION_SUBSTEPS"

ompt_measure/render_ompt_timing "$TIMING_CSV" "$TIMING_SVG" "$STEP_INTERVAL"

if [ "$SNAPSHOT_INTERVAL" -gt 0 ]; then
    for ((step = 0; step <= STEPS; step += SNAPSHOT_INTERVAL)); do
        printf -v snapshot_txt '%s_%06d.txt' "$SNAPSHOT_PREFIX" "$step"
        [ -e "$snapshot_txt" ] || continue
        snapshot_name=$(basename "$snapshot_txt" .txt)
        ./render_amr_snapshot "$snapshot_txt" "$SVG_DIR/${snapshot_name}.svg"
    done
fi

fine_n=$((COARSE_N * (1 << MAX_LEVEL)))
printf 'fine_n=%d, case_dir=%s, wrote %s and %s\n' "$fine_n" "$CASE_DIR" "$TIMING_CSV" "$TIMING_SVG"
