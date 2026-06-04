#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export OMPT_MODE=${OMPT_MODE:-dvfs}
export OMPT_TOOL_LIBRARY=ompt_dvfs/libompt_dvfs.so
export OMPT_RENDERER=ompt_dvfs/render_ompt_timing
exec "$REPO_ROOT/run.sh" "$@"
