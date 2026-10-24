#!/usr/bin/env bash
# Release build of the unified forward-pass solver. Override MARCH for cross/emulated builds,
# e.g. MARCH=x86-64-v3 ./build_solvers_release.sh
set -euo pipefail
mkdir -p build

clang++ \
  -fopencilk \
  -std=c++17 \
  -march="${MARCH:-native}" \
  -O3 \
  -DNDEBUG \
  -Wall \
  -Ithird_party \
  src/attention_solver.cpp \
  -o build/attention_solver
