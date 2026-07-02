#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bash ompt_resctrl/resctrl_cleanup.sh

export OMPT_MODE=${OMPT_MODE:-resctrl}
export OMPT_TOOL_LIBRARY=ompt_resctrl/libompt_resctrl.so
export OMPT_RENDERER=ompt_resctrl/render_ompt_timing
export AMR_RESCTRL_RESOURCE=${AMR_RESCTRL_RESOURCE:-mba}
export AMR_RESCTRL_MONITOR=${AMR_RESCTRL_MONITOR:-1}
export AMR_OMPT_EXEC_INTERVAL=${AMR_OMPT_EXEC_INTERVAL:-100}
exec "$REPO_ROOT/run.sh" "$@"
