#!/usr/bin/env bash
# Release build of the unified forward-pass solver.
#
# Environment overrides:
#   MARCH=...   -march value (default native), e.g. MARCH=x86-64-v3 for cross/emulated builds
#   OUT=...     output binary path (default build/attention_solver, or build/attention_solver_papi)
#   PAPI=1      build with PAPI hardware counters (-DATTENTION_USE_PAPI). Include/library paths
#               come from PAPI_INC/PAPI_LIB, falling back to TACC's TACC_PAPI_INC/TACC_PAPI_LIB.
set -euo pipefail
cd "$(dirname "$0")"

extra_flags=()
default_out=build/attention_solver
if [[ "${PAPI:-0}" == "1" ]]; then
  default_out=build/attention_solver_papi
  papi_inc="${PAPI_INC:-${TACC_PAPI_INC:-}}"
  papi_lib="${PAPI_LIB:-${TACC_PAPI_LIB:-}}"
  extra_flags+=(-DATTENTION_USE_PAPI)
  [[ -n "$papi_inc" ]] && extra_flags+=(-I"$papi_inc")
  [[ -n "$papi_lib" ]] && extra_flags+=(-L"$papi_lib" -Wl,-rpath,"$papi_lib")
  extra_flags+=(-lpapi)
fi

out="${OUT:-$default_out}"
mkdir -p "$(dirname "$out")"

clang++ \
  -fopencilk \
  -std=c++17 \
  -march="${MARCH:-native}" \
  -O3 \
  -DNDEBUG \
  -Wall \
  -Ithird_party \
  src/attention_solver.cpp \
  "${extra_flags[@]}" \
  -o "$out"
