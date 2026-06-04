#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export OMPT_MODE=measure
export OMPT_TOOL_LIBRARY=ompt_measure/libompt_measure.so
export OMPT_RENDERER=ompt_measure/render_ompt_timing
exec "$REPO_ROOT/run.sh" "$@"
