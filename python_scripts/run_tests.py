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
import json
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
        runs += [SolverRun("fa2_lockfree_bwd", ("-b", str(b)), w) for b in (16, 64, 100)]
        runs += [SolverRun("fa2_locking_bwd", ("-b", str(b)), w) for b in (16, 64, 100)]
        runs += [
            SolverRun("summary_bwd", ("-b", str(b), "-p", str(p)), w)
            for b, p in itertools.product((16, 64), (1, 4, 16))
        ]
        runs += [
            SolverRun("owned_bwd", ("-b", str(b), "-g", str(g)), w)
            for b, g in itertools.product((16, 64), (16, 64, 256))
        ]
        # p = 3, 12, 48 exercise the switch to owned traversals after 0, 1 and 2 summary levels
        runs += [
            SolverRun("hybrid_bwd", ("-b", str(b), "-g", "64", "-p", str(p)), w)
            for b, p in itertools.product((16, 64), (1, 3, 4, 12, 16, 48))
        ]
    return runs


def sparse_runs(workers: list[int]) -> list[SolverRun]:
    return [
        SolverRun(algorithm, ("-b", str(b), "-g", str(g)), w)
        for algorithm in ("sparse_fwd", "sparse_bwd")
        for w in workers for b, g in itertools.product((16, 64), (16, 256))
    ]


def is_sparse(run: SolverRun) -> bool:
    return run.algorithm.startswith("sparse_")


# Sparse cases: (label, mask kind, fractions of zero query, jointly zero K/V, and zero dO rows).
# Forward runs the cases without zero dO rows; backward runs them all.
SPARSE_CASES = [
    ("full", "full", 0.0, 0.0, 0.0),
    ("block", "block", 0.0, 0.0, 0.0),
    ("band", "band", 0.0, 0.0, 0.0),
    ("element", "element", 0.0, 0.0, 0.0),
    ("diagonal", "diagonal", 0.0, 0.0, 0.0),
    ("zeroQ", "full", 0.5, 0.0, 0.0),
    ("zeroKV", "full", 0.0, 0.5, 0.0),
    ("allZeroQ", "full", 1.0, 0.0, 0.0),
    ("allZeroKV", "full", 0.0, 1.0, 0.0),
    ("block+zeros", "block", 0.4, 0.4, 0.0),
    ("element+zeros", "element", 0.4, 0.4, 0.0),
    ("band+zeros", "band", 0.5, 0.25, 0.0),
    ("zeroDO", "full", 0.0, 0.0, 0.5),
    ("allZeroDO", "full", 0.0, 0.0, 1.0),
    ("block+zeroDO", "block", 0.0, 0.0, 0.4),
    ("element+allKinds", "element", 0.3, 0.3, 0.3),
    ("band+allKinds", "band", 0.4, 0.25, 0.4),
]


def make_mask(kind: str, n: int, seed: int) -> np.ndarray:
    """Dense 0/1 mask with every row nonempty (the diagonal is always allowed)."""
    rng = np.random.default_rng(seed)
    i, j = np.indices((n, n))
    if kind == "full":
        mask = np.ones((n, n), dtype=bool)
    elif kind == "diagonal":
        mask = np.zeros((n, n), dtype=bool)
    elif kind == "band":
        mask = np.abs(i - j) <= max(1, n // 8)
    elif kind == "block":
        tile = max(4, n // 6)
        tiles = rng.random(((n + tile - 1) // tile,) * 2) < 0.4
        mask = tiles[i // tile, j // tile]
    elif kind == "element":
        mask = rng.random((n, n)) < 0.15
    else:
        raise ValueError(kind)
    return (mask | (i == j)).astype(np.uint8)


def zero_row_runs(n: int, fraction: float, seed: int) -> np.ndarray:
    """Row indices to zero, in aligned runs of about n/16 rows (as the solver's generator places them)."""
    run = max(1, n // 16)
    runs = (n + run - 1) // run
    chosen = np.random.default_rng(seed).permutation(runs)[: int(round(fraction * runs))]
    rows = [r for c in chosen for r in range(c * run, min(n, (c + 1) * run))]
    return np.array(rows, dtype=int)


def numpy_masked_attention(qkv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    q, k, v = (x.astype(np.float64) for x in qkv)
    s = (q @ k.T) / np.sqrt(q.shape[1])
    s = np.where(mask.astype(bool), s, -np.inf)
    s -= s.max(axis=1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=1, keepdims=True)
    return p @ v


def numpy_masked_attention_backward(qkvdo: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Float64 (dQ, dK, dV) of masked attention given the upstream gradient dO."""
    q, k, v, do = (x.astype(np.float64) for x in qkvdo)
    scale = 1.0 / np.sqrt(q.shape[1])
    s = np.where(mask.astype(bool), (q @ k.T) * scale, -np.inf)
    s -= s.max(axis=1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=1, keepdims=True)
    o = p @ v
    dp = do @ v.T
    ds = p * (dp - (do * o).sum(axis=1, keepdims=True))
    return np.stack([ds @ k * scale, ds.T @ q * scale, p.T @ do])


def load_sparse_case(problem: Problem, case: tuple, data_dir: Path,
                     backward: bool = False) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    """Problem file with the case's zero rows (and dO for backward) and its dense mask file (cached)."""
    label, kind, zero_q, zero_kv, zero_do = case
    stem = f"sparse_{problem.name}_{label.replace('+', '_')}{'_bwd' if backward else ''}"
    problem_path, mask_path = data_dir / f"{stem}.npy", data_dir / f"{stem}_mask.npy"
    if problem_path.exists() and mask_path.exists():
        return problem_path, mask_path, np.load(problem_path), np.load(mask_path)
    _, qkv = load_problem(problem, data_dir, with_dO=backward)
    qkv = qkv.copy()
    if backward:
        qkv[3, zero_row_runs(problem.n, zero_do, problem.seed + 4)] = 0.0
    qkv[0, zero_row_runs(problem.n, zero_q, problem.seed + 1)] = 0.0
    kv_rows = zero_row_runs(problem.n, zero_kv, problem.seed + 2)
    qkv[1, kv_rows] = 0.0
    qkv[2, kv_rows] = 0.0
    mask = make_mask(kind, problem.n, problem.seed + 3)
    np.save(problem_path, qkv)
    np.save(mask_path, mask)
    return problem_path, mask_path, qkv, mask


def run_sparse(runner: "SolverRunner", problem: Problem, runs: list[SolverRun], data_dir: Path,
               out_path: Path, results: "Results", verbose: bool) -> None:
    forward = [r for r in runs if not is_backward(r)]
    backward = [r for r in runs if is_backward(r)]
    for case in SPARSE_CASES:
        if forward and case[4] == 0.0:
            run_sparse_case(runner, problem, case, forward, data_dir, out_path, results, verbose, backward=False)
        if backward:
            run_sparse_case(runner, problem, case, backward, data_dir, out_path, results, verbose, backward=True)


def run_sparse_case(runner: "SolverRunner", problem: Problem, case: tuple, runs: list[SolverRun], data_dir: Path,
                    out_path: Path, results: "Results", verbose: bool, backward: bool) -> None:
    problem_path, mask_path, data, mask = load_sparse_case(problem, case, data_dir, backward)
    ref = numpy_masked_attention_backward(data, mask) if backward else numpy_masked_attention(data, mask)
    for run in runs:
        masked = SolverRun(run.algorithm, (*run.args, "--mask-file", str(mask_path.relative_to(REPO_ROOT))),
                           run.workers)
        proc = runner.run(masked, problem_path, out_path)
        label = f"{run.label} [{case[0]}]"
        if proc.returncode != 0:
            results.failures.append(f"exit code {proc.returncode}  {label}\n{proc.stderr.strip()}")
            print(f"  FAIL  exit code {proc.returncode}  {label}\n{proc.stderr.strip()}")
            continue
        out = np.load(out_path)
        if not backward:
            check(results, label, error_metrics(out, ref), (REF_MAX_ABS_TOL, REF_REL_FRO_TOL), verbose)
            continue
        for name, grad, grad_ref in zip(GRADIENT_NAMES, out, ref):
            abs_tol = GRAD_MAX_ABS_TOL * max(1.0, float(np.abs(grad_ref).max()))
            check(results, f"{label} {name}", error_metrics(grad, grad_ref), (abs_tol, GRAD_REL_FRO_TOL), verbose)


def check_mask_generators(runner: "SolverRunner", data_dir: Path, results: "Results", verbose: bool) -> None:
    """The quadtree must reproduce a dense mask exactly, and each built-in pattern must have its shape."""
    n, b = 300, 16
    out = data_dir / "dumped_mask.npy"
    rel_out = str(out.relative_to(REPO_ROOT))
    print(f"mask generators (n = {n}, b = {b})")

    def dump(*args: str) -> np.ndarray | None:
        proc = runner.run_args(["sparse_fwd", "--random", str(n), "-b", str(b), *args, "--dump-mask", rel_out], 1)
        if proc.returncode != 0:
            results.failures.append(f"exit code {proc.returncode}  dump {' '.join(args)}\n{proc.stderr.strip()}")
            return None
        return np.load(out)

    def verdict(ok: bool, label: str) -> None:
        if ok:
            results.passed += 1
            if verbose:
                print(f"  ok    {label}")
        else:
            results.failures.append(label)
            print(f"  FAIL  {label}")

    i, j = np.indices((n, n))
    dense = (np.random.default_rng(7).random((n, n)) < 0.3).astype(np.uint8)
    dense_path = data_dir / "roundtrip_mask.npy"
    np.save(dense_path, dense)
    got = dump("--mask-file", str(dense_path.relative_to(REPO_ROOT)))
    verdict(got is not None and np.array_equal(got, dense), "dense mask -> quadtree -> dense is exact")

    got = dump("--mask", "band", "--mask-density", "0.2")
    width = int((0.2 * n - 1) / 2)
    verdict(got is not None and np.array_equal(got, (np.abs(i - j) <= width).astype(np.uint8)),
            f"band pattern is |i - j| <= {width}")

    tile = 32
    got = dump("--mask", "block", "--mask-density", "0.4", "--mask-tile", str(tile))
    ok = got is not None and bool(np.all(np.diag(got) == 1))
    if ok:
        off = got.astype(bool) & (i != j)
        for ti in range(0, n, tile):
            for tj in range(0, n, tile):
                block = off[ti:ti + tile, tj:tj + tile]
                interior = ~np.eye(*block.shape, k=tj - ti, dtype=bool) if ti == tj else np.ones_like(block)
                vals = block[interior]
                ok = ok and (vals.all() or not vals.any())
    verdict(ok, f"block pattern: tiles of {tile} all-or-nothing, diagonal allowed")

    got = dump("--mask", "element", "--mask-density", "0.3")
    ok = got is not None and bool(np.all(np.diag(got) == 1)) and abs(got[i != j].mean() - 0.3) < 0.02
    verdict(ok, "element pattern: density 0.3 +- 0.02, diagonal allowed")
    out.unlink(missing_ok=True)


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

    def run_args(self, args: list[str], workers: int) -> subprocess.CompletedProcess:
        """Runs the solver with explicit arguments (no problem file or output)."""
        cmd = [self.solver, *args]
        if self.container:
            cmd = ["docker", "exec", "-e", f"CILK_NWORKERS={workers}", self.container, *cmd]
            env = None
        else:
            env = {**os.environ, "CILK_NWORKERS": str(workers)}
        return subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)

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


def expect_rejected(runner: SolverRunner, run: SolverRun, problem_path: Path, out_path: Path,
                    results: Results, verbose: bool) -> None:
    """Checks that the solver refuses an invalid configuration (e.g. a non-power-of-4 p)."""
    proc = runner.run(run, problem_path, out_path)
    if proc.returncode == 0:
        results.failures.append(f"expected non-zero exit  {run.label}")
        print(f"  FAIL  expected non-zero exit  {run.label}")
    else:
        results.passed += 1
        if verbose:
            print(f"  ok    rejected as expected  {run.label}")


def check_memory(runner: SolverRunner, results: Results, verbose: bool) -> None:
    """--memory must isolate each algorithm's own footprint from its inputs: the standard
    solvers' N x N matrices must show up in full, while FA2's tiles stay far below them."""
    n = 2048
    matrix = n * n * 4
    print(f"memory (N = {n}, N x N float32 = {matrix / 2**20:.0f} MiB)")
    cases = [
        ("standard", 1, True),            # S
        ("fa2", 1, False),
        ("standard_bwd", 2, True),        # P and dS
        ("fa2_locking_bwd", 1, False),
    ]
    for algorithm, matrices, expect_quadratic in cases:
        proc = runner.run_args([algorithm, "--random", str(n), "-w", "0", "-r", "1", "--jsonl", "--memory"], 4)
        label = f"{algorithm} --memory"
        if proc.returncode != 0 or not proc.stdout.strip():
            results.failures.append(f"exit code {proc.returncode}  {label}\n{proc.stderr.strip()}")
            print(f"  FAIL  exit code {proc.returncode}  {label}")
            continue
        mem = json.loads(proc.stdout.splitlines()[0]).get("memory", {})
        delta = mem.get("peak_delta_bytes", -1)
        if expect_quadratic:
            ok = delta >= 0.9 * matrices * matrix
            expectation = f">= 0.9 x {matrices} x N^2 floats"
        else:
            ok = 0 <= delta < 0.25 * matrix
            expectation = "< 0.25 x N^2 floats"
        ok = ok and mem.get("peak_reset", False)
        line = f"peak_delta={delta / 2**20:.1f} MiB ({expectation})  {label}"
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
    forward = [r for r in runs if not is_backward(r) and not is_sparse(r)]
    backward = [r for r in runs if is_backward(r) and not is_sparse(r)]
    sparse = [r for r in runs if is_sparse(r)]

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
    if sparse:
        run_sparse(runner, problem, sparse, data_dir, out_path, results, verbose)
    if backward:
        problem_path, qkvdo = load_problem(problem, data_dir, with_dO=True)
        grads_ref = numpy_attention_backward(qkvdo)
        if "summary_bwd" in {r.algorithm for r in backward}:
            expect_rejected(runner, SolverRun("summary_bwd", ("-p", "8")), problem_path, out_path, results, verbose)
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

    if "summary" in {r.algorithm for r in runs}:
        expect_rejected(runner, SolverRun("summary", ("-p", "8")), problem_path, out_path, results, verbose)

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
    runs = solver_runs(workers) + backward_runs(workers) + sparse_runs(workers)
    if args.algorithm:
        runs = [r for r in runs if r.algorithm in args.algorithm]

    results = Results()
    runner = SolverRunner(args.solver, args.docker)
    try:
        for problem in QUICK_PROBLEMS if args.quick else FULL_PROBLEMS:
            run_problem(runner, problem, runs, data_dir, results, args.verbose)
        if not args.algorithm:
            check_memory(runner, results, args.verbose)
        if not args.algorithm or "sparse_fwd" in args.algorithm:
            check_mask_generators(runner, data_dir, results, args.verbose)
    finally:
        runner.close()

    total = results.passed + len(results.failures)
    print(f"\n{results.passed}/{total} checks passed")
    for failure in results.failures:
        print(f"  FAIL  {failure}")
    return 0 if not results.failures else 1


if __name__ == "__main__":
    sys.exit(main())
