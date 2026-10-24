#!/usr/bin/env bash
# Debug build of the unified solver. Set CILKSAN=1 to instrument with the Cilksan race detector.
set -euo pipefail
mkdir -p build

extra_flags=()
if [[ "${CILKSAN:-0}" == "1" ]]; then
  extra_flags+=(-fsanitize=cilk -O1)
fi

clang++ \
  -fopencilk \
  -std=c++17 \
  -g \
  -Wall \
  -Wno-deprecated-declarations \
  -Ithird_party \
  "${extra_flags[@]}" \
  src/attention_solver.cpp \
  -o build/attention_solver_debug
