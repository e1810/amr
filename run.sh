#!/usr/bin/env bash
set -eo pipefail

if [ -f /opt/intel/oneapi/setvars.sh ]; then
    source /opt/intel/oneapi/setvars.sh >/dev/null
fi

set -u

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

amr_metrics_read_energy_uj() {
    local total=0
    local found=0
    local path base name energy

    for path in /sys/class/powercap/intel-rapl:*; do
        [ -d "$path" ] || continue
        base=$(basename "$path")
        [[ "$base" =~ ^intel-rapl:[0-9]+$ ]] || continue
        [ -r "$path/name" ] || continue
        name=$(cat "$path/name") || continue
        [[ "$name" == package-* ]] || continue
        [ -r "$path/energy_uj" ] || continue
        energy=$(cat "$path/energy_uj") || continue

        total=$((total + energy))
        found=1
    done

    if [ "$found" -eq 0 ]; then
        printf 'nan\n'
    else
        printf '%s\n' "$total"
    fi
}

amr_metrics_write_csv() {
    local elapsed_ns=$1
    local energy_before_uj=$2
    local energy_after_uj=$3
    local csv=$4

    awk -v elapsed_ns="$elapsed_ns" \
        -v before="$energy_before_uj" \
        -v after="$energy_after_uj" '
        BEGIN {
            elapsed_s = elapsed_ns / 1000000000.0
            energy_j = "nan"
            avg_power_w = "nan"

            if (before != "nan" && after != "nan") {
                delta_uj = after - before
                if (delta_uj >= 0 && elapsed_s > 0.0) {
                    energy_j = delta_uj / 1000000.0
                    avg_power_w = energy_j / elapsed_s
                }
            }

            print "elapsed_s,avg_power_w,energy_j"
            printf "%.9f,%s,%s\n", elapsed_s, avg_power_w, energy_j
        }
    ' >"$csv"
}

amr_metrics_print_summary() {
    local csv=$1
    awk -F, '
        NR == 2 {
            printf "amr_elapsed_s=%s, avg_power_w=%s, energy_j=%s, wrote %s\n",
                   $1, $2, $3, csv
        }
    ' csv="$csv" "$csv"
}

# Increase coarse_n to increase fine_n = coarse_n * 2^max_level.
# The default here is intentionally heavier than the quick examples so the
# per-thread loop time is larger than OpenMP/OMPT overhead.
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

if [ -n "${OMPT_TOOL_LIBRARY:-}" ]; then
    USE_OMPT=1
    : "${OMPT_MODE:?OMPT_MODE is required when OMPT_TOOL_LIBRARY is set}"
    : "${OMPT_RENDERER:?OMPT_RENDERER is required when OMPT_TOOL_LIBRARY is set}"
else
    USE_OMPT=0
fi

RUN_MODE=${OMPT_MODE:-plain}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs}
CASE_NAME=${CASE_NAME:-$INITIAL_PATTERN}
CASE_DIR=${CASE_DIR:-$OUTPUT_ROOT/$CASE_NAME/$RUN_MODE}
SVG_DIR=${SVG_DIR:-$CASE_DIR/svg}
TXT_DIR=${TXT_DIR:-$CASE_DIR/txt}

mkdir -p "$CASE_DIR" "$SVG_DIR" "$TXT_DIR"

METRICS_CSV=${METRICS_CSV:-$CASE_DIR/metrics.csv}
SNAPSHOT_INTERVAL=${SNAPSHOT_INTERVAL:-100}
SNAPSHOT_PREFIX=${SNAPSHOT_PREFIX:-$TXT_DIR/snapshot}

if [ "$USE_OMPT" -eq 1 ]; then
    TIMING_CSV=${TIMING_CSV:-$CASE_DIR/ompt_regions.csv}
    TIMING_SVG=${TIMING_SVG:-$CASE_DIR/ompt_timing.svg}
    export OMP_TOOL=enabled
    export OMP_TOOL_LIBRARIES=${OMP_TOOL_LIBRARIES:-$REPO_ROOT/$OMPT_TOOL_LIBRARY}
    export AMR_OMPT_OUT=$TIMING_CSV
    export AMR_OMPT_EXEC_INTERVAL=$STEP_INTERVAL
else
    export OMP_TOOL=disabled
fi
export OMP_PROC_BIND=${OMP_PROC_BIND:-true}
export OMP_PLACES=${OMP_PLACES:-cores}

ENERGY_BEFORE_UJ=$(amr_metrics_read_energy_uj)
AMR_START_NS=$(date +%s%N)
set +e
"$REPO_ROOT/amr" "$COARSE_N" "$PARTS" "$MAX_LEVEL" "$INITIAL_SCALE" "$STEPS" "$DIFFUSION" \
    "$SNAPSHOT_INTERVAL" "$SNAPSHOT_PREFIX" "$INITIAL_PATTERN" \
    "$DIFFUSION_SUBSTEPS"
AMR_STATUS=$?
set -e
AMR_END_NS=$(date +%s%N)
ENERGY_AFTER_UJ=$(amr_metrics_read_energy_uj)
AMR_ELAPSED_NS=$((AMR_END_NS - AMR_START_NS))
amr_metrics_write_csv "$AMR_ELAPSED_NS" "$ENERGY_BEFORE_UJ" "$ENERGY_AFTER_UJ" "$METRICS_CSV"
amr_metrics_print_summary "$METRICS_CSV"
if [ "$AMR_STATUS" -ne 0 ]; then
    exit "$AMR_STATUS"
fi

if [ "$USE_OMPT" -eq 1 ]; then
    "$REPO_ROOT/$OMPT_RENDERER" "$TIMING_CSV" "$TIMING_SVG" "$STEP_INTERVAL"
fi

if [ "$SNAPSHOT_INTERVAL" -gt 0 ]; then
    for ((step = 0; step <= STEPS; step += SNAPSHOT_INTERVAL)); do
        printf -v snapshot_txt '%s_%06d.txt' "$SNAPSHOT_PREFIX" "$step"
        [ -e "$snapshot_txt" ] || continue
        snapshot_name=$(basename "$snapshot_txt" .txt)
        "$REPO_ROOT/render_amr_snapshot" "$snapshot_txt" "$SVG_DIR/${snapshot_name}.svg"
    done
fi

fine_n=$((COARSE_N * (1 << MAX_LEVEL)))
if [ "$USE_OMPT" -eq 1 ]; then
    printf 'fine_n=%d, case_dir=%s, wrote %s, %s and %s\n' "$fine_n" "$CASE_DIR" "$TIMING_CSV" "$TIMING_SVG" "$METRICS_CSV"
else
    printf 'fine_n=%d, case_dir=%s, wrote %s\n' "$fine_n" "$CASE_DIR" "$METRICS_CSV"
fi
