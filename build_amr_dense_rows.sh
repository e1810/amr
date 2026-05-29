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
    amr_dense_rows.cpp -o amr_dense_rows
