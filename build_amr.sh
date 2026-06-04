#!/usr/bin/env bash
set -eo pipefail

if [ -f /opt/intel/oneapi/setvars.sh ]; then
    # oneAPI prints a banner by default; keep the build output quiet.
    source /opt/intel/oneapi/setvars.sh >/dev/null
fi

set -u

CXX=${CXX:-icpx}
if ! command -v "$CXX" >/dev/null 2>&1; then
    CXX=g++
fi

case "$(basename "$CXX")" in
    icpx|icpx-*) OMP_FLAG=-qopenmp ;;
    *) OMP_FLAG=-fopenmp ;;
esac

"$CXX" -std=c++17 -O2 -Wall -Wextra -pedantic "$OMP_FLAG" \
    amr.cpp -o amr

"$CXX" -std=c++17 -O2 -Wall -Wextra -pedantic \
    render_amr_snapshot.cpp -o render_amr_snapshot

"$CXX" -std=c++17 -O2 -Wall -Wextra -pedantic -fPIC -shared "$OMP_FLAG" \
    ompt_measure/ompt_measure.cpp -o ompt_measure/libompt_measure.so

"$CXX" -std=c++17 -O2 -Wall -Wextra -pedantic \
    ompt_measure/render_ompt_timing.cpp -o ompt_measure/render_ompt_timing

"$CXX" -std=c++17 -O2 -Wall -Wextra -pedantic -fPIC -shared "$OMP_FLAG" \
    ompt_dvfs/ompt_dvfs.cpp ompt_dvfs/msr_freq.cpp -o ompt_dvfs/libompt_dvfs.so

"$CXX" -std=c++17 -O2 -Wall -Wextra -pedantic \
    ompt_dvfs/render_ompt_timing.cpp -o ompt_dvfs/render_ompt_timing
