#!/bin/bash
# Diagnostic: why is owned_bwd's first repetition ~5x slower than the rest on Ice Lake (N >= 2^19)?
# Submit from the repository root:
#   sbatch slurm_jobs/diag_owned_first_run.slurm
#
# Runs owned_bwd at N = 2^19 (Ice Lake's tuned b = 128, g = 256) under conditions that each change
# one thing, recording per-repetition CPU time and page faults (--rusage) and /proc/vmstat deltas.
# Results go to $RESULTS_DIR/diagnostics/owned_first_run_<jobid>/, with a summary at the end of
# the job log. Takes ~40 minutes if the slow first repetition reproduces.
#SBATCH -J diag_owned
#SBATCH -p icx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 80
#SBATCH -t 01:30:00
#SBATCH -o diag_owned.%j.out
#SBATCH -e diag_owned.%j.out
##SBATCH -A <your-allocation>

set -uo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
RESULTS_DIR="${RESULTS_DIR:-${SCRATCH:-$HOME}/cpu_attention_results}"
MODULES="${MODULES:-opencilk/3.0.0}"
N="${N:-524288}"
SMALL_N="${SMALL_N:-262144}" # Case H
THREADS="${THREADS:-80}"
OUT="$RESULTS_DIR/diagnostics/owned_first_run_${SLURM_JOB_ID:-$$}"
mkdir -p "$OUT"

if command -v module >/dev/null 2>&1 && [[ -n "$MODULES" ]]; then
  # shellcheck disable=SC2086
  module load $MODULES || echo "Warning: failed to load some of: $MODULES"
fi

echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-none}  Output: $OUT"

# ---------------------------------------------------------------------------
# Environment that could explain a first-run penalty
# ---------------------------------------------------------------------------
{
  echo "== uname";              uname -a
  echo "== glibc";              ldd --version 2>&1 | head -1
  echo "== THP enabled";        cat /sys/kernel/mm/transparent_hugepage/enabled 2>&1
  echo "== THP defrag";         cat /sys/kernel/mm/transparent_hugepage/defrag 2>&1
  echo "== THP khugepaged";     cat /sys/kernel/mm/transparent_hugepage/khugepaged/defrag 2>&1
  echo "== numa_balancing";     cat /proc/sys/kernel/numa_balancing 2>&1
  echo "== zone_reclaim_mode";  cat /proc/sys/vm/zone_reclaim_mode 2>&1
  echo "== overcommit_memory";  cat /proc/sys/vm/overcommit_memory 2>&1
  echo "== numactl";            numactl --hardware 2>&1 || echo "numactl not available"
  echo "== lscpu";              lscpu
  echo "== meminfo";            head -20 /proc/meminfo
} > "$OUT/environment.txt" 2>&1
grep -A1 -E "== (THP|numa_balancing|zone_reclaim)" "$OUT/environment.txt"

# ---------------------------------------------------------------------------
# Build the solver for this node into the diagnostic directory
# ---------------------------------------------------------------------------
SOLVER="$OUT/attention_solver"
if ! OUT="$SOLVER" "$REPO/build_solvers_release.sh" > "$OUT/build.log" 2>&1; then
  echo "Build failed; see $OUT/build.log"
  exit 1
fi

# vmstat counters that reveal page-fault, huge-page, compaction and NUMA-migration work
VMSTAT_KEYS="pgfault pgmajfault thp_fault_alloc thp_fault_fallback thp_collapse_alloc compact_stall \
compact_fail compact_success numa_hint_faults numa_pte_updates numa_pages_migrated pgmigrate_success \
pgscan_direct pgsteal_direct allocstall_normal allocstall_movable"

vmstat_snapshot() {
  awk -v keys="$VMSTAT_KEYS" 'BEGIN { n = split(keys, k, " "); for (i = 1; i <= n; i++) want[k[i]] = 1 }
                             ($1 in want) { print $1, $2 }' /proc/vmstat
}

# run_case <name> <algorithm> <solver args...>; set ENV_PREFIX / WRAP for environment or wrappers
run_case() {
  local name="$1"; shift
  echo "-- $name: $*  ${ENV_PREFIX:-} ${WRAP:-}"
  vmstat_snapshot > "$OUT/$name.vmstat_before"
  # shellcheck disable=SC2086
  env CILK_NWORKERS="${CASE_THREADS:-$THREADS}" ${ENV_PREFIX:-} ${WRAP:-} \
    "$SOLVER" "$@" --random "$N" --seed 424 --jsonl --rusage \
    > "$OUT/$name.jsonl" 2> "$OUT/$name.stderr"
  local status=$?
  vmstat_snapshot > "$OUT/$name.vmstat_after"
  [[ $status -ne 0 ]] && echo "   exit $status; see $OUT/$name.stderr"
  cat "$OUT/$name.jsonl"
}

OWNED=(owned_bwd -b 128 -g 256)

# A. Reproduce: two repetitions, no warmup (as in the sweep at N >= 2^19)
run_case A_owned_baseline       "${OWNED[@]}" -w 0 -r 2
# B. Control: summary_bwd showed no first-run penalty
run_case B_summary_control      summary_bwd -b 256 -p 1024 -w 0 -r 2
# C. Does one untimed warmup remove it?
run_case C_owned_warmup         "${OWNED[@]}" -w 1 -r 2
# D. glibc malloc: a single arena (serializes allocation; tests per-thread arena growth)
ENV_PREFIX="GLIBC_TUNABLES=glibc.malloc.arena_max=1" \
run_case D_owned_one_arena      "${OWNED[@]}" -w 0 -r 2
# E. glibc malloc: never return freed memory to the OS (tests repeated page faulting of heaps)
ENV_PREFIX="GLIBC_TUNABLES=glibc.malloc.trim_threshold=17179869184:glibc.malloc.top_pad=1073741824" \
run_case E_owned_no_trim        "${OWNED[@]}" -w 0 -r 2
if command -v numactl >/dev/null 2>&1 && numactl --interleave=all true 2>/dev/null; then
  # F. NUMA: interleave all pages across both sockets
  WRAP="numactl --interleave=all" \
  run_case F_owned_interleave   "${OWNED[@]}" -w 0 -r 2
  # G. NUMA: one socket only (half the workers, memory bound locally)
  CASE_THREADS=$(( THREADS / 2 )) WRAP="numactl --cpunodebind=0 --membind=0" \
  run_case G_owned_one_socket   "${OWNED[@]}" -w 0 -r 2
else
  echo "-- F/G skipped: numactl unavailable or not permitted to set memory policy here"
fi
# H. Size threshold: the sweep only ran 2^18 with a warmup, so this was never observed there
N="$SMALL_N" run_case H_owned_small_no_warmup "${OWNED[@]}" -w 0 -r 2

# ---------------------------------------------------------------------------
# Summary: per repetition time, CPU split, faults; vmstat deltas per case
# ---------------------------------------------------------------------------
python3 - "$OUT" <<'EOF'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
print("\n== Summary (per repetition)")
print(f"{'case':28} {'rep':>3} {'wall s':>8} {'user s':>9} {'sys s':>8} {'sys %':>6} {'minflt':>11} {'invol csw':>9}")
for jsonl in sorted(out.glob("*.jsonl")):
    for line in jsonl.read_text().splitlines():
        rep = json.loads(line)
        u = rep.get("rusage", {})
        cpu = u.get("user_s", 0) + u.get("sys_s", 0)
        sys_pct = 100 * u.get("sys_s", 0) / cpu if cpu else 0
        print(f"{jsonl.stem:28} {rep['rep']:>3} {rep['time_ms'] / 1000:8.1f} {u.get('user_s', 0):9.1f} "
              f"{u.get('sys_s', 0):8.1f} {sys_pct:6.1f} {u.get('minflt', 0):11d} {u.get('nivcsw', 0):9d}")

print("\n== /proc/vmstat deltas per case (nonzero only)")
for before in sorted(out.glob("*.vmstat_before")):
    name = before.name[: -len(".vmstat_before")]
    after = out / f"{name}.vmstat_after"
    if not after.exists():
        continue
    b = dict(line.split() for line in before.read_text().splitlines())
    a = dict(line.split() for line in after.read_text().splitlines())
    deltas = {k: int(a[k]) - int(b.get(k, 0)) for k in a if int(a[k]) - int(b.get(k, 0))}
    print(f"{name:28} " + ", ".join(f"{k}={v}" for k, v in deltas.items()))
EOF
echo "Done: $OUT"
