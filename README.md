# CPU Attention Suite

Cache-oblivious self-attention algorithms on CPU (C++17, OpenCilk, Eigen).

## Building

Builds require the OpenCilk compiler; locally use the `opencilk:v3.0` docker image:

```bash
docker run --rm -v "$PWD":/work -w /work opencilk:v3.0 ./build_solvers_release.sh
```

This produces `build/attention_solver`. Set `MARCH=...` to override `-march=native`.
`./debug_build.sh` builds `build/attention_solver_debug`; with `CILKSAN=1` it is instrumented
with the Cilksan race detector.

## Running

```bash
python python_scripts/generate_problem.py -N 65536 -d 64 -o data/problem_65536_64.npy --seed 424
CILK_NWORKERS=16 build/attention_solver fa2 data/problem_65536_64.npy -b 64 -w 1 -r 5 --json
```

Algorithms: `reference`, `standard`, `streamed`, `fa2`, `recursive`, `summary` (alias `rec`;
requires `-p` to be a power of 4, defaulting to `CILK_NWORKERS`), `query`.
Run `build/attention_solver` with no arguments for the full option list.

## Testing

```bash
python python_scripts/run_tests.py --docker opencilk:v3.0   # or omit --docker on Linux
```

Validates the C++ `reference` solver against a float64 NumPy reference, then checks every
solver configuration (block sizes, `p`, worker counts) against the C++ reference. Use
`--quick` for a reduced set, `-k <algorithm>` to filter, and `-v` to print passing checks.

## Licenses

We include Eigen, which is distributed under Mozilla Public License Version 2.0.
