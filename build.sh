clang++ \
  -fopencilk \
  -std=c++17 \
  -O3 \
  -Wall \
  -Ithird_party \
  src/solver.cpp \
  -o build/solver
