#!/usr/bin/env python3
"""Correctness tests for the dense attention solvers in build/attention_solver.

Each test problem is generated deterministically (and cached under data/tests). For every
problem the C++ `reference` solver is first validated against a float64 NumPy reference, and
then every other forward solver configuration is checked against the C++ reference output.
Backward solvers (*_bwd) are checked gradient by gradient (dQ, dK, dV) against a float64
NumPy backward pass, on the same problems extended with a random upstream gradient dO.

Run natively (Linux with OpenCilk installed):
    python python_scripts/run_tests.py

Run with the solver inside the OpenCilk docker image (one container is reused for all runs):
    python python_scripts/run_tests.py --docker opencilk:v3.0
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_problem import generate_problem  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
D = 64

# Tolerances on the output O relative to the C++ reference (float32 accumulation in both).
MAX_ABS_TOL = 1e-5
REL_FRO_TOL = 1e-5
# Looser bound for the C++ reference against the float64 NumPy ground truth.
REF_MAX_ABS_TOL = 1e-5
REF_REL_FRO_TOL = 1e-5
# Backward gradients vs the float64 NumPy backward. Looser: the solver's forward setup (O, L)
# is float32, and dS = P * (dP - D) subtracts nearly equal terms. The absolute tolerance is
# scaled by max(1, max |grad|): with Q scaled up, dK = dS^T Q grows to ~10.
GRAD_MAX_ABS_TOL = 1e-5
GRAD_REL_FRO_TOL = 1e-4
GRADIENT_NAMES = ("dQ", "dK", "dV")


@dataclass(frozen=True)
class Problem:
    n: int
    seed: int
    q_scale: float = 1.0  # Scaling Q widens the score range, stressing the running-max rescaling

    @property
    def name(self) -> str:
        suffix = "" if self.q_scale == 1.0 else f"_qs{self.q_scale:g}"
        return f"N{self.n}_seed{self.seed}{suffix}"


@dataclass(frozen=True)
class SolverRun:
    algorithm: str
    args: tuple[str, ...] = ()
    workers: int = 1

    @property
    def label(self) -> str:
        return " ".join([self.algorithm, *self.args, f"[w={self.workers}]"])


@dataclass
class Results:
    passed: int = 0
    failures: list[str] = field(default_factory=list)


FULL_PROBLEMS = [
    Problem(1, 11),
    Problem(7, 12),
    Problem(63, 13),
    Problem(64, 14),
    Problem(65, 15),
    Problem(100, 16),
    Problem(257, 17),
    Problem(1000, 18),
    Problem(1024, 19),
    Problem(2048, 20),
    Problem(1000, 21, q_scale=20.0),
]
QUICK_PROBLEMS = [Problem(7, 12), Problem(65, 15), Problem(257, 17), Problem(1000, 21, q_scale=20.0)]


def solver_runs(workers: list[int]) -> list[SolverRun]:
    runs = [SolverRun("recursive", ("-b", str(b))) for b in (16, 64)]
    for w in workers:
        runs += [SolverRun("standard", workers=w), SolverRun("streamed", workers=w)]
        runs += [SolverRun("fa2", ("-b", str(b)), w) for b in (16, 64, 100)]
        runs += [
            SolverRun("summary", ("-b", str(b), "-p", str(p)), w)
            for b, p in itertools.product((16, 64), (1, 4, 16))
        ]
        runs += [
            SolverRun("query", ("-b", str(b), "-g", str(g)), w)
            for b, g in itertools.product((16, 64), (16, 64, 256))
        ]
        runs += [
            SolverRun("hybrid", ("-b", str(b), "-g", str(g), "-p", str(p)), w)
            for b, g, p in itertools.product((16, 64), (16, 64), (1, 3, 4, 12, 16, 48))
        ]
    return runs


def backward_runs(workers: list[int]) -> list[SolverRun]:
    runs = []
    for w in workers:
        runs.append(SolverRun("standard_bwd", workers=w))
        runs += [SolverRun("fa2_bwd", ("-b", str(b)), w) for b in (16, 64, 100)]
    return runs


def is_backward(run: SolverRun) -> bool:
    return run.algorithm.endswith("_bwd")


def numpy_attention_backward(qkvdo: np.ndarray) -> np.ndarray:
    """Float64 (dQ, dK, dV) of O = softmax(Q K^T / sqrt(d)) V given upstream gradient dO."""
    q, k, v, do = (x.astype(np.float64) for x in qkvdo)
    scale = 1.0 / np.sqrt(q.shape[1])
    s = (q @ k.T) * scale
    s -= s.max(axis=1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=1, keepdims=True)
    o = p @ v
    dv = p.T @ do
    dp = do @ v.T
    ds = p * (dp - (do * o).sum(axis=1, keepdims=True))
    return np.stack([ds @ k * scale, ds.T @ q * scale, dv])


def numpy_attention(qkv: np.ndarray) -> np.ndarray:
    q, k, v = (x.astype(np.float64) for x in qkv)
    s = (q @ k.T) / np.sqrt(q.shape[1])
    s -= s.max(axis=1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=1, keepdims=True)
    return p @ v


def error_metrics(out: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    diff = out.astype(np.float64) - ref.astype(np.float64)
    max_abs = float(np.abs(diff).max())
    ref_norm = float(np.linalg.norm(ref))
    # A (near-)zero reference, e.g. dQ and dK at N = 1, has no meaningful relative error
    rel_fro = float(np.linalg.norm(diff) / ref_norm) if ref_norm > 1e-6 else 0.0
    return max_abs, rel_fro


class SolverRunner:
    """Runs the solver natively or via `docker exec` into one long-lived container."""

    def __init__(self, solver: str, docker_image: str | None):
        self.solver = solver
        self.container: str | None = None
        if docker_image:
            self.container = subprocess.check_output(
                [
                    "docker", "run", "-d", "--rm",
                    "-v", f"{REPO_ROOT}:/work", "-w", "/work",
                    docker_image, "sleep", "infinity",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()

    def close(self) -> None:
        if self.container:
            subprocess.run(["docker", "kill", self.container], capture_output=True)
            self.container = None

    def run(self, run: SolverRun, problem_path: Path, output_path: Path) -> subprocess.CompletedProcess:
        # Paths are passed relative to the repo root so they resolve both natively and in /work.
        cmd = [
            self.solver, run.algorithm,
            str(problem_path.relative_to(REPO_ROOT)), *run.args,
            "-o", str(output_path.relative_to(REPO_ROOT)),
        ]
        env_var = f"CILK_NWORKERS={run.workers}"
        if self.container:
            cmd = ["docker", "exec", "-e", env_var, self.container, *cmd]
            env = None
        else:
            env = {**os.environ, "CILK_NWORKERS": str(run.workers)}
        return subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)


def load_problem(problem: Problem, data_dir: Path, with_dO: bool = False) -> tuple[Path, np.ndarray]:
    """(3, N, d) [Q, K, V] problem, or (4, N, d) [Q, K, V, dO] for backward tests."""
    path = data_dir / f"problem_{problem.name}{'_bwd' if with_dO else ''}.npy"
    if path.exists():
        return path, np.load(path)
    qkv = generate_problem(problem.n, D, problem.seed)
    qkv[0] *= np.float32(problem.q_scale)
    if with_dO:
        d_o = np.random.default_rng(problem.seed + 10_000).uniform(-1.0, 1.0, size=(1, problem.n, D))
        qkv = np.concatenate([qkv, d_o.astype(np.float32)])
    np.save(path, qkv)
    return path, qkv


def check(results: Results, label: str, metrics: tuple[float, float], tols: tuple[float, float],
          verbose: bool) -> None:
    max_abs, rel_fro = metrics
    ok = max_abs <= tols[0] and rel_fro <= tols[1]
    line = f"max_abs={max_abs:.2e} rel_fro={rel_fro:.2e}  {label}"
    if ok:
        results.passed += 1
        if verbose:
            print(f"  ok    {line}")
    else:
        results.failures.append(line)
        print(f"  FAIL  {line}")


def run_problem(runner: SolverRunner, problem: Problem, runs: list[SolverRun], data_dir: Path,
                results: Results, verbose: bool) -> None:
    out_path = data_dir / f"out_{problem.name}.npy"
    print(f"{problem.name}")
    forward = [r for r in runs if not is_backward(r)]
    backward = [r for r in runs if is_backward(r)]

    def execute(run: SolverRun, problem_path: Path) -> np.ndarray | None:
        proc = runner.run(run, problem_path, out_path)
        if proc.returncode != 0:
            msg = f"exit code {proc.returncode}  {run.label}\n{proc.stderr.strip()}"
            results.failures.append(msg)
            print(f"  FAIL  {msg}")
            return None
        return np.load(out_path)

    if forward:
        run_forward(runner, problem, forward, data_dir, out_path, execute, results, verbose)
    if backward:
        problem_path, qkvdo = load_problem(problem, data_dir, with_dO=True)
        grads_ref = numpy_attention_backward(qkvdo)
        for run in backward:
            out = execute(run, problem_path)
            if out is None:
                continue
            for name, grad, ref in zip(GRADIENT_NAMES, out, grads_ref):
                abs_tol = GRAD_MAX_ABS_TOL * max(1.0, float(np.abs(ref).max()))
                check(results, f"{run.label} {name}", error_metrics(grad, ref),
                      (abs_tol, GRAD_REL_FRO_TOL), verbose)

    out_path.unlink(missing_ok=True)


def run_forward(runner: SolverRunner, problem: Problem, runs: list[SolverRun], data_dir: Path, out_path: Path,
                execute, results: Results, verbose: bool) -> None:
    problem_path, qkv = load_problem(problem, data_dir)

    bad_p = SolverRun("summary", ("-p", "8"))
    if "summary" in {r.algorithm for r in runs}:
        proc = runner.run(bad_p, problem_path, out_path)
        if proc.returncode == 0:
            results.failures.append(f"expected non-zero exit  {bad_p.label}")
            print(f"  FAIL  expected non-zero exit  {bad_p.label}")
        else:
            results.passed += 1
            if verbose:
                print(f"  ok    rejected as expected  {bad_p.label}")

    ref_run = SolverRun("reference")
    o_ref = execute(ref_run, problem_path)
    if o_ref is None:
        return
    check(results, f"{ref_run.label} vs numpy float64", error_metrics(o_ref, numpy_attention(qkv)),
          (REF_MAX_ABS_TOL, REF_REL_FRO_TOL), verbose)

    for run in runs:
        out = execute(run, problem_path)
        if out is not None:
            check(results, run.label, error_metrics(out, o_ref), (MAX_ABS_TOL, REL_FRO_TOL), verbose)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solver", default="build/attention_solver",
                        help="Solver binary, relative to the repo root (default: %(default)s)")
    parser.add_argument("--docker", metavar="IMAGE", help="Run the solver inside this docker image")
    parser.add_argument("--workers", default="1,4",
                        help="Comma-separated CILK_NWORKERS values for parallel solvers (default: %(default)s)")
    parser.add_argument("--data-dir", default="data/tests", help="Cache for generated problems")
    parser.add_argument("--quick", action="store_true", help="Run a reduced problem set")
    parser.add_argument("-k", "--algorithm", action="append",
                        help="Only test these algorithms (repeatable)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print passing checks too")
    args = parser.parse_args()

    data_dir = (REPO_ROOT / args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    if not args.docker and not (REPO_ROOT / args.solver).exists():
        print(f"Solver binary not found: {args.solver} (build it first)", file=sys.stderr)
        return 2

    workers = [int(w) for w in args.workers.split(",")]
    runs = solver_runs(workers) + backward_runs(workers)
    if args.algorithm:
        runs = [r for r in runs if r.algorithm in args.algorithm]

    results = Results()
    runner = SolverRunner(args.solver, args.docker)
    try:
        for problem in QUICK_PROBLEMS if args.quick else FULL_PROBLEMS:
            run_problem(runner, problem, runs, data_dir, results, args.verbose)
    finally:
        runner.close()

    total = results.passed + len(results.failures)
    print(f"\n{results.passed}/{total} checks passed")
    for failure in results.failures:
        print(f"  FAIL  {failure}")
    return 0 if not results.failures else 1


if __name__ == "__main__":
    sys.exit(main())
