clang++ \
  -fopencilk \
  -std=c++17 \
  -march=native \
  -O3 \
  -Wall \
  -Ithird_party \
  src/fa2_solver.cpp \
  -o build/fa2_solver

clang++ \
  -fopencilk \
  -std=c++17 \
  -march=native \
  -O3 \
  -Wall \
  -Ithird_party \
  src/standard_solver.cpp \
  -o build/standard_solver

clang++ \
  -fopencilk \
  -std=c++17 \
  -march=native \
  -O3 \
  -Wall \
  -Ithird_party \
  src/rec_solver.cpp \
  -o build/rec_solver

