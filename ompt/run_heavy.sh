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
INITIAL_SCALE=${INITIAL_SCALE:-0.25}
STEPS=${STEPS:-2000}
DIFFUSION=${DIFFUSION:-0.01}
STEP_INTERVAL=${STEP_INTERVAL:-100}
INITIAL_PATTERN=${INITIAL_PATTERN:-checker}
DIFFUSION_SUBSTEPS=${DIFFUSION_SUBSTEPS:-16}

OUTPUT=${OUTPUT:-ompt/amr_ompt.svg}
TIMING_CSV=${TIMING_CSV:-ompt/ompt_regions.csv}
TIMING_SVG=${TIMING_SVG:-ompt/ompt_timing.svg}
SNAPSHOT_INTERVAL=${SNAPSHOT_INTERVAL:-100}
SNAPSHOT_PREFIX=${SNAPSHOT_PREFIX:-snapshots/checker_heavy}

export OMP_TOOL=enabled
export OMP_TOOL_LIBRARIES=${OMP_TOOL_LIBRARIES:-$PWD/ompt/libamr_ompt_tool.so}
export AMR_OMPT_OUT=$TIMING_CSV
export AMR_OMPT_STEP_INTERVAL=$STEP_INTERVAL
export OMP_PROC_BIND=${OMP_PROC_BIND:-true}
export OMP_PLACES=${OMP_PLACES:-cores}

./amr "$COARSE_N" "$PARTS" "$MAX_LEVEL" "$INITIAL_SCALE" "$STEPS" "$DIFFUSION" \
    "$OUTPUT" "$SNAPSHOT_INTERVAL" "$SNAPSHOT_PREFIX" "$INITIAL_PATTERN" \
    "$DIFFUSION_SUBSTEPS"

ompt/render_ompt_timing "$TIMING_CSV" "$TIMING_SVG" "$STEP_INTERVAL"

if [ "$SNAPSHOT_INTERVAL" -gt 0 ]; then
    for snapshot_txt in "${SNAPSHOT_PREFIX}"_*.txt; do
        [ -e "$snapshot_txt" ] || continue
        ./render_amr_snapshot "$snapshot_txt"
    done
fi

fine_n=$((COARSE_N * (1 << MAX_LEVEL)))
printf 'fine_n=%d, wrote %s and %s\n' "$fine_n" "$TIMING_CSV" "$TIMING_SVG"
