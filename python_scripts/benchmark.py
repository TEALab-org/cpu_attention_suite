#!/usr/bin/env python3
"""Restartable benchmark driver for the dense forward-pass solvers.

Runs on one machine (one architecture) per invocation, in these phases:

  build  Compile build/attention_solver for this machine into the results directory.
  tune   At a demo size (--tune-n), sweep each tunable solver's parameters and pick the
         configuration with the lowest median time (written to tune/best.json).
  sweep  Time each solver with its tuned parameters over the sequence lengths (--min-n..--max-n),
         also recording the peak resident memory each algorithm adds above its inputs.
  papi   With --papi: build a PAPI-enabled solver and repeat the sweep collecting hardware
         cache-miss counters instead of (well, as well as) timings.

--pass selects the forward solvers (default) or the backward (*_bwd) solvers. Backward results
go to tune_backward/, sweep_backward/ and papi_backward/, so the two passes never mix.

Every experiment writes one JSON file, updated after each repetition, so an interrupted run
loses at most the repetition in flight; rerunning the same command resumes where it stopped.
Results live under <results-dir>/<label>/ and are never written into the repository.

Uses only the Python standard library, so it runs in the opencilk docker image and on
cluster login/compute nodes without extra packages.

Example:
    python3 python_scripts/benchmark.py --results-dir $SCRATCH/attn_results --label skx --papi
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import re
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
D = 64
BYTES_F32 = 4

# Exit codes
EXIT_DONE = 0
EXIT_ERROR = 1
EXIT_INCOMPLETE = 3  # Stopped by --time-limit or a signal; rerun to resume

# Sweep order per pass: earlier algorithms run first at each N
DEFAULT_ALGORITHMS = {
    "forward": ["fa2", "summary", "query", "hybrid", "streamed", "standard"],
    "backward": ["fa2_lockfree_bwd", "fa2_locking_bwd", "summary_bwd", "owned_bwd", "hybrid_bwd", "standard_bwd"],
}

# Statuses that are final for an experiment (complete or deliberately not run)
FINAL_STATUSES = {"complete", "skipped"}
# Statuses that are only retried with --retry-failed
FAILED_STATUSES = {"failed", "timeout"}


# ----------------------------------------------------------------------------
# Machine description
# ----------------------------------------------------------------------------


def physical_core_count() -> int:
    """Physical cores available to this process (respects Slurm/taskset affinity, skips SMT)."""
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except AttributeError:  # Not Linux
        return os.cpu_count() or 1
    cores = set()
    for cpu in cpus:
        topo = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = (topo / "physical_package_id").read_text().strip()
            core = (topo / "core_id").read_text().strip()
        except OSError:
            return len(cpus)
        cores.add((package, core))
    return max(1, len(cores))


def mem_total_bytes() -> Optional[int]:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def default_label() -> str:
    partition = os.environ.get("SLURM_JOB_PARTITION")
    if partition:
        return partition
    return re.sub(r"[^a-z0-9]+", "_", cpu_model().lower()).strip("_") or "unknown"


def command_output(cmd: List[str]) -> Optional[str]:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_commit() -> Optional[str]:
    commit = command_output(["git", "rev-parse", "HEAD"])
    if commit:
        if command_output(["git", "status", "--porcelain", "--untracked-files=no"]):
            commit += "-dirty"
        return commit
    # No usable git (e.g. not installed in the container): read the checkout's HEAD directly
    git_dir = REPO_ROOT / ".git"
    try:
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[len("ref: "):]
        ref_path = git_dir / ref
        if ref_path.exists():
            return ref_path.read_text().strip()
        for line in (git_dir / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        pass
    return None


# ----------------------------------------------------------------------------
# Experiment specification
# ----------------------------------------------------------------------------

# Parameters each algorithm takes, in file-name order
ALGORITHM_PARAMS = {
    "standard": [],
    "streamed": [],
    "fa2": ["b"],
    "summary": ["b", "p"],
    "query": ["b", "g"],
    "hybrid": ["b", "g", "p"],
    "standard_bwd": [],
    "fa2_lockfree_bwd": ["b"],
    "fa2_locking_bwd": ["b"],
    "summary_bwd": ["b", "p"],
    "owned_bwd": ["b", "g"],
    "hybrid_bwd": ["b", "g", "p"],
}
PARAM_FLAGS = {"b": "-b", "g": "-g", "p": "-p"}


def powers_of_4_near(threads: int) -> List[int]:
    """Powers of 4 bracketing the thread count, plus one more level of over-decomposition."""
    lower = 1
    while lower * 4 <= threads:
        lower *= 4
    upper = lower if lower == threads else lower * 4
    return sorted({lower, upper, upper * 4})


def tuning_grid(algorithm: str, threads: int) -> List[Dict[str, int]]:
    blocks = [16, 32, 64, 128, 256]
    grains = [64, 128, 256, 512, 1024, 2048]
    if algorithm in ("fa2", "fa2_lockfree_bwd", "fa2_locking_bwd"):
        return [{"b": b} for b in blocks + [512]]
    if algorithm in ("summary", "summary_bwd"):
        return [{"b": b, "p": p} for b, p in itertools.product(blocks, powers_of_4_near(threads))]
    if algorithm in ("query", "owned_bwd"):
        return [{"b": b, "g": g} for b, g in itertools.product(blocks[:4], grains) if g >= b]
    if algorithm in ("hybrid", "hybrid_bwd"):
        return [{"b": b, "g": g, "p": threads} for b, g in itertools.product(blocks[:4], grains) if g >= b]
    return [{}]


def summary_levels(algorithm: str, p: int) -> int:
    """Number of ParallelSummary levels (each doubles the auxiliary summary memory)."""
    levels = 0
    if algorithm in ("summary", "summary_bwd"):
        while p > 1:
            p //= 4
            levels += 1
    elif algorithm in ("hybrid", "hybrid_bwd"):
        while p > 0 and p % 4 == 0:
            p //= 4
            levels += 1
    return levels


def is_backward(algorithm: str) -> bool:
    return algorithm.endswith("_bwd")


def result_dir_name(kind: str, pass_name: str) -> str:
    """tune/sweep/papi for the forward pass (unchanged layout), tune_backward/... for backward."""
    return kind if pass_name == "forward" else f"{kind}_{pass_name}"


def estimated_bytes(algorithm: str, n: int, params: Dict[str, int], threads: int) -> int:
    if is_backward(algorithm):
        # Q, K, V, dO, O, L, dQ, dK, dV, plus the gradients copied between repetitions
        base = 12 * n * D * BYTES_F32
        if algorithm == "standard_bwd":
            return base + 2 * n * n * BYTES_F32  # P and dS
        if algorithm in ("summary_bwd", "hybrid_bwd"):
            # Each summary level doubles the private (dQ, dK, dV) gradient buffers
            levels = summary_levels(algorithm, params.get("p", 1))
            return base + 3 * (2 ** levels) * n * D * BYTES_F32
        return base
    base = 5 * n * D * BYTES_F32  # Q, K, V, O, plus the copy assigned between repetitions
    if algorithm == "standard":
        return base + n * n * BYTES_F32
    if algorithm == "streamed":
        return base + threads * n * BYTES_F32
    if algorithm in ("summary", "hybrid", "query"):
        levels = summary_levels(algorithm, params.get("p", 1))
        return base + (2 ** levels) * n * D * BYTES_F32
    return base


@dataclass
class Experiment:
    kind: str  # "tune", "sweep" or "papi"
    algorithm: str
    n: int
    params: Dict[str, int]
    reps: int
    warmup: int
    seed: int

    def file_name(self) -> str:
        parts = [self.algorithm, f"N{self.n}"]
        parts += [f"{k}{self.params[k]}" for k in ALGORITHM_PARAMS[self.algorithm] if k in self.params]
        return "_".join(parts) + ".json"

    def solver_args(self) -> List[str]:
        args = [self.algorithm, "--random", str(self.n), "--seed", str(self.seed)]
        for key in ALGORITHM_PARAMS[self.algorithm]:
            if key in self.params:
                args += [PARAM_FLAGS[key], str(self.params[key])]
        return args


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------


def atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_json(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


class Interrupted(Exception):
    """Raised when the time limit is reached or a termination signal arrives."""


@dataclass
class Runner:
    args: argparse.Namespace
    machine_dir: Path
    threads: int
    commit: Optional[str]
    deadline: Optional[float]
    current: Optional[subprocess.Popen] = None
    stop_requested: bool = False
    counts: Dict[str, int] = field(default_factory=dict)

    def request_stop(self, *_):
        self.stop_requested = True
        proc = self.current
        if proc and proc.poll() is None:
            proc.terminate()

    def check_stop(self) -> None:
        if self.stop_requested or (self.deadline is not None and time.time() >= self.deadline):
            raise Interrupted()

    def tally(self, status: str) -> None:
        self.counts[status] = self.counts.get(status, 0) + 1

    def record_base(self, exp: Experiment, solver: Path) -> dict:
        record = {
            "experiment": exp.kind,
            "pass": self.args.pass_name,
            "name": exp.algorithm,
            "N": exp.n,
            "n": exp.n,
            "d": D,
            "threads": self.threads,
            "params": exp.params,
            "seed": exp.seed,
            "warmup": exp.warmup,
            "reps_target": exp.reps,
            "label": self.args.label,
            "git_commit": self.commit,
            "solver": str(solver),
            "times": [],
            "status": "running",
        }
        # Flat aliases matching the original result files used by the plotting notebooks
        if "b" in exp.params:
            record["block_size"] = record["base_size"] = exp.params["b"]
        if "g" in exp.params:
            record["grain"] = exp.params["g"]
        if "p" in exp.params:
            record["p"] = exp.params["p"]
        return record

    def run(self, exp: Experiment, out_dir: Path, solver: Path, extra_args: List[str] = ()) -> Optional[dict]:
        """Runs (or resumes) one experiment; returns its record, or None if not run."""
        self.check_stop()
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / exp.file_name()
        record = load_json(path)

        if record is not None:
            status = record.get("status")
            if status in FINAL_STATUSES or (status in FAILED_STATUSES and not self.args.retry_failed):
                self.tally(status)
                return record
        if record is None or record.get("status") in FAILED_STATUSES:
            record = self.record_base(exp, solver)

        mem = mem_total_bytes()
        need = estimated_bytes(exp.algorithm, exp.n, exp.params, self.threads)
        if mem is not None and need > self.args.mem_fraction * mem:
            record.update(status="skipped", reason=f"needs ~{need / 2**30:.1f} GiB of {mem / 2**30:.1f} GiB")
            atomic_write_json(path, record)
            print(f"  skip  {path.name}: {record['reason']}")
            self.tally("skipped")
            return record

        projected = self.projected_rep_seconds(exp, out_dir)
        if projected is not None and self.args.max_rep_seconds and projected > self.args.max_rep_seconds:
            record.update(status="skipped", reason=f"projected {projected:.0f} s per repetition")
            atomic_write_json(path, record)
            print(f"  skip  {path.name}: {record['reason']}")
            self.tally("skipped")
            return record

        remaining = exp.reps - len(record["times"])
        if remaining <= 0:  # Killed after the last repetition was saved but before completion
            record["status"] = "complete"
            atomic_write_json(path, record)
            self.tally("complete")
            return record
        cmd =[str(solver), *exp.solver_args(), *extra_args,
               "-r", str(remaining), "-w", str(exp.warmup), "--jsonl"]
        record["command"] = cmd
        print(f"  run   {path.name}  ({len(record['times'])}/{exp.reps} done)", flush=True)

        env = {**os.environ, "CILK_NWORKERS": str(self.threads)}
        stderr_path = path.with_suffix(".stderr")
        timed_out = threading.Event()
        with open(stderr_path, "w") as stderr_file:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr_file, text=True, env=env)
            self.current = proc
            watchdog = threading.Thread(target=self.watch, args=(proc, timed_out), daemon=True)
            watchdog.start()
            try:
                for line in proc.stdout:
                    rep = json.loads(line)
                    record["times"].append(rep["time_ms"])
                    if "counters" in rep:
                        record.setdefault("counters", []).append(rep["counters"])
                    # Peak memory of this experiment's first solve; a resumed run keeps the original
                    if "memory" in rep and "memory" not in record:
                        record["memory"] = rep["memory"]
                    atomic_write_json(path, record)
                    print(f"        rep {len(record['times'])}/{exp.reps}: {rep['time_ms'] / 1000:.3f} s", flush=True)
            finally:
                proc.wait()
                self.current = None

        stderr_text = stderr_path.read_text()
        if proc.returncode == 0 and len(record["times"]) >= exp.reps:
            record["status"] = "complete"
            stderr_path.unlink(missing_ok=True)
        elif self.stop_requested or (self.deadline is not None and time.time() >= self.deadline):
            record["status"] = "partial"
            atomic_write_json(path, record)
            raise Interrupted()
        elif timed_out.is_set():
            record.update(status="timeout", reason=f"exceeded {self.args.experiment_timeout} s")
        else:
            record.update(status="failed", reason=f"exit code {proc.returncode}",
                          stderr_tail=stderr_text[-2000:])
            print(f"  FAIL  {path.name}: exit code {proc.returncode}\n{stderr_text[-2000:]}")
        atomic_write_json(path, record)
        self.tally(record["status"])
        return record

    def watch(self, proc: subprocess.Popen, timed_out: threading.Event) -> None:
        start = time.time()
        limit = self.args.experiment_timeout
        while proc.poll() is None:
            now = time.time()
            if (self.deadline is not None and now >= self.deadline) or (limit and now - start >= limit):
                if limit and now - start >= limit:
                    timed_out.set()
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return
            time.sleep(1.0)

    def projected_rep_seconds(self, exp: Experiment, out_dir: Path) -> Optional[float]:
        """Extrapolates the time per repetition from the largest smaller N already measured."""
        best = None
        for path in out_dir.glob(f"{exp.algorithm}_N*.json"):
            rec = load_json(path)
            if not rec or rec.get("params") != exp.params or not rec.get("times") or rec["N"] >= exp.n:
                continue
            if best is None or rec["N"] > best["N"]:
                best = rec
        if best is None:
            return None
        return statistics.median(best["times"]) / 1000.0 * (exp.n / best["N"]) ** 2


# ----------------------------------------------------------------------------
# Phases
# ----------------------------------------------------------------------------


def build_solver(out: Path, march: str, papi: bool) -> bool:
    # Build beside the target and rename into place, so another job (e.g. the other pass on
    # the same machine label) that is running the old binary keeps its copy untouched
    tmp = out.with_name(f"{out.name}.tmp{os.getpid()}")
    env = {**os.environ, "OUT": str(tmp), "MARCH": march, "PAPI": "1" if papi else "0"}
    print(f"Building {out} (march={march}{', PAPI' if papi else ''})", flush=True)
    proc = subprocess.run([str(REPO_ROOT / "build_solvers_release.sh")], env=env, cwd=REPO_ROOT,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        print(f"Build failed:\n{proc.stdout}{proc.stderr}", file=sys.stderr)
        return False
    os.replace(tmp, out)
    return True


def tune_phase(runner: Runner, solver: Path, algorithms: List[str]) -> Dict[str, Dict[str, int]]:
    args = runner.args
    tune_dir = runner.machine_dir / result_dir_name("tune", args.pass_name)
    best_path = tune_dir / "best.json"
    best = load_json(best_path) or {}

    print(f"\n== Tuning ({args.pass_name}) at N = {args.tune_n}")
    for algorithm in algorithms:
        if algorithm in best:
            continue
        grid = tuning_grid(algorithm, runner.threads)
        if grid == [{}]:
            best[algorithm] = {}
            continue
        results = []
        for params in grid:
            exp = Experiment("tune", algorithm, args.tune_n, params, args.reps,
                             warmup_for(args, args.tune_n), args.seed)
            rec = runner.run(exp, tune_dir, solver)
            if rec and rec.get("status") == "complete":
                results.append((statistics.median(rec["times"]), params))
        if results:
            median_ms, params = min(results, key=lambda r: r[0])
            best[algorithm] = params
            print(f"  best  {algorithm}: {params} ({median_ms / 1000:.3f} s median)")
            atomic_write_json(best_path, best)
        else:
            print(f"  WARN  no successful tuning runs for {algorithm}; leaving it out of the sweep")
    return best


def warmup_for(args: argparse.Namespace, n: int) -> int:
    # Warmup matters for short runs (page faults, runtime start-up) but doubles the cost of long ones
    return 1 if n <= args.warmup_max_n else 0


def sweep_phase(runner: Runner, solver: Path, kind: str, best: Dict[str, Dict[str, int]],
                algorithms: List[str], reps: int, extra_args: List[str] = ()) -> None:
    args = runner.args
    out_dir = runner.machine_dir / result_dir_name(kind, args.pass_name)
    print(f"\n== {kind} ({args.pass_name}) over N = {', '.join(str(n) for n in args.sizes)}")
    for n in args.sizes:
        for algorithm in algorithms:
            if algorithm not in best:
                continue
            exp = Experiment(kind, algorithm, n, best[algorithm], reps, warmup_for(args, n), args.seed)
            runner.run(exp, out_dir, solver, extra_args)


def write_run_info(machine_dir: Path, args: argparse.Namespace, threads: int, commit: Optional[str]) -> None:
    info = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "label": args.label,
        "pass": args.pass_name,
        "hostname": socket.gethostname(),
        "cpu_model": cpu_model(),
        "threads": threads,
        "logical_cpus": os.cpu_count(),
        "mem_total_bytes": mem_total_bytes(),
        "git_commit": commit,
        "compiler": command_output(["clang++", "--version"]),
        "lscpu": command_output(["lscpu"]),
        "python": sys.version,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "argv": sys.argv,
    }
    runs_dir = machine_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    job = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    atomic_write_json(runs_dir / f"{stamp}_{job}.json", info)


def parse_sizes(args: argparse.Namespace) -> List[int]:
    if args.sizes:
        return sorted({int(s) for s in args.sizes.split(",")})
    sizes = []
    n = args.min_n
    while n <= args.max_n:
        sizes.append(n)
        n *= 2
    return sizes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", required=True, type=Path,
                        help="Root directory for results (kept outside the repository)")
    parser.add_argument("--label", default=None,
                        help="Machine label; results go to <results-dir>/<label> (default: Slurm partition or CPU model)")
    parser.add_argument("--threads", type=int, default=None,
                        help="CILK_NWORKERS for every run (default: physical cores available to this job)")
    parser.add_argument("--reps", type=int, default=10, help="Timed repetitions per experiment (default: %(default)s)")
    parser.add_argument("--warmup-max-n", type=int, default=2**18,
                        help="Run one untimed warmup only for N <= this (default: %(default)s)")
    parser.add_argument("--pass", dest="pass_name", choices=["forward", "backward"], default="forward",
                        help="Benchmark the forward or the backward (*_bwd) solvers (default: %(default)s)")
    parser.add_argument("--algorithms", default=None,
                        help="Comma-separated solvers to benchmark (default: forward "
                             f"{','.join(DEFAULT_ALGORITHMS['forward'])}; backward "
                             f"{','.join(DEFAULT_ALGORITHMS['backward'])})")
    parser.add_argument("--phases", default="build,tune,sweep,papi",
                        help="Comma-separated phases to run (default: %(default)s; papi also needs --papi)")
    parser.add_argument("--tune-n", type=int, default=2**16, help="Problem size for tuning (default: %(default)s)")
    parser.add_argument("--min-n", type=int, default=2**12, help="Smallest sweep size (default: %(default)s)")
    parser.add_argument("--max-n", type=int, default=2**22, help="Largest sweep size (default: %(default)s)")
    parser.add_argument("--sizes", default=None, help="Explicit comma-separated sweep sizes (overrides min/max)")
    parser.add_argument("--seed", type=int, default=424, help="Problem generator seed (default: %(default)s)")
    parser.add_argument("--march", default="native", help="-march for the build (default: %(default)s)")
    parser.add_argument("--papi", action="store_true", help="Also run the PAPI cache-miss sweep")
    parser.add_argument("--papi-reps", type=int, default=3,
                        help="Repetitions per PAPI experiment (default: %(default)s)")
    parser.add_argument("--papi-events", default="PAPI_L1_DCM,PAPI_L2_DCM,PAPI_L3_TCM",
                        help="PAPI events to count (default: %(default)s)")
    parser.add_argument("--mem-fraction", type=float, default=0.8,
                        help="Skip experiments estimated to need more than this fraction of RAM (default: %(default)s)")
    parser.add_argument("--max-rep-seconds", type=float, default=None,
                        help="Skip experiments whose extrapolated time per repetition exceeds this")
    parser.add_argument("--experiment-timeout", type=float, default=None,
                        help="Kill and mark as timed out any single experiment running longer than this (s)")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="Stop cleanly after this many seconds (set a bit below the job's wall time)")
    parser.add_argument("--retry-failed", action="store_true", help="Rerun experiments that failed or timed out")
    args = parser.parse_args()

    args.label = args.label or default_label()
    args.sizes = parse_sizes(args)
    phases = set(args.phases.split(","))
    algorithms = ([a for a in args.algorithms.split(",") if a] if args.algorithms
                  else DEFAULT_ALGORITHMS[args.pass_name])
    unknown = [a for a in algorithms if a not in ALGORITHM_PARAMS]
    if unknown:
        parser.error(f"unknown algorithms: {', '.join(unknown)}")
    wrong_pass = [a for a in algorithms if is_backward(a) != (args.pass_name == "backward")]
    if wrong_pass:
        parser.error(f"not {args.pass_name} algorithms: {', '.join(wrong_pass)} (see --pass)")

    machine_dir = (args.results_dir / args.label).resolve()
    machine_dir.mkdir(parents=True, exist_ok=True)
    threads = args.threads or physical_core_count()
    commit = git_commit()
    deadline = time.time() + args.time_limit if args.time_limit else None

    print(f"Results: {machine_dir}\nPass: {args.pass_name}  Threads: {threads}  Reps: {args.reps}  Commit: {commit}")
    write_run_info(machine_dir, args, threads, commit)

    runner = Runner(args, machine_dir, threads, commit, deadline)
    signal.signal(signal.SIGTERM, runner.request_stop)
    signal.signal(signal.SIGINT, runner.request_stop)

    build_dir = machine_dir / "build"
    solver = build_dir / "attention_solver"
    papi_solver = build_dir / "attention_solver_papi"
    exit_code = EXIT_DONE

    try:
        if "build" in phases and not build_solver(solver, args.march, papi=False):
            return EXIT_ERROR
        if not solver.exists():
            print(f"Solver not found: {solver} (run the build phase)", file=sys.stderr)
            return EXIT_ERROR

        best = load_json(machine_dir / result_dir_name("tune", args.pass_name) / "best.json") or {}
        if "tune" in phases:
            best = tune_phase(runner, solver, algorithms)
        best = {a: p for a, p in best.items() if a in algorithms}
        for a in algorithms:
            if a not in best and not tuning_grid(a, threads)[0]:
                best[a] = {}  # Untunable solvers need no tuning results

        if "sweep" in phases:
            # Timing runs also record each experiment's peak memory (cheap: two /proc reads)
            sweep_phase(runner, solver, "sweep", best, algorithms, args.reps, ["--memory"])

        if args.papi and "papi" in phases:
            if "build" in phases and not build_solver(papi_solver, args.march, papi=True):
                print("Skipping PAPI phase: PAPI build failed (is the papi module loaded?)", file=sys.stderr)
                exit_code = EXIT_ERROR
            elif not papi_solver.exists():
                print(f"Skipping PAPI phase: {papi_solver} not found", file=sys.stderr)
                exit_code = EXIT_ERROR
            else:
                extra = ["--papi-events", args.papi_events]
                # Probe once so a node without counter access fails fast with PAPI's error message
                probe = subprocess.run(
                    [str(papi_solver), "fa2", "--random", "256", "-r", "1", "--jsonl", *extra],
                    capture_output=True, text=True, env={**os.environ, "CILK_NWORKERS": str(threads)})
                if probe.returncode != 0:
                    print(f"Skipping PAPI phase: counters unavailable:\n{probe.stderr.strip()}", file=sys.stderr)
                    exit_code = EXIT_ERROR
                else:
                    sweep_phase(runner, papi_solver, "papi", best, algorithms, args.papi_reps, extra)
    except Interrupted:
        print("\nStopped before finishing (time limit or signal); rerun the same command to resume.")
        exit_code = EXIT_INCOMPLETE

    print("\nExperiment status counts this run: " +
          ", ".join(f"{k}={v}" for k, v in sorted(runner.counts.items())))
    if exit_code == EXIT_DONE and runner.counts.get("failed", 0) + runner.counts.get("timeout", 0):
        exit_code = EXIT_ERROR
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
