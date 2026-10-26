#!/usr/bin/env bash
# Shared body of the per-architecture Slurm jobs: loads modules and runs the benchmark driver.
#
#   run_benchmark.sh <label>
#
# Settings (override with `sbatch --export=ALL,REPS=5,... slurm_jobs/skx.slurm`):
#   REPO             Repository checkout (default: the directory sbatch was run from)
#   RESULTS_DIR      Results root, outside the repo (default: $SCRATCH/cpu_attention_results)
#   MODULES          Modules to load (default: "opencilk/3.0.0 papi/7.1.0")
#   REPS             Timed repetitions per experiment (default: 10)
#   PAPI_REPS        Repetitions per PAPI experiment (default: 3)
#   PHASES           Driver phases (default: build,tune,sweep,papi)
#   ALGORITHMS       Solvers to tune and sweep (default: fa2,summary,query,hybrid)
#   MAX_N            Largest sweep size (default: 1048576 = 2^20)
#   MAX_REP_SECONDS  Skip experiments projected to exceed this per repetition (default: unset)
#   EXTRA_ARGS       Any further benchmark.py arguments
#   RESUBMIT         1 = resubmit this job when it stops at its time limit (default: 0)
set -uo pipefail

label="$1"
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
RESULTS_DIR="${RESULTS_DIR:-${SCRATCH:-$HOME}/cpu_attention_results}"
MODULES="${MODULES:-opencilk/3.0.0 papi/7.1.0}"

if command -v module >/dev/null 2>&1 && [[ -n "$MODULES" ]]; then
  # shellcheck disable=SC2086
  module load $MODULES || echo "Warning: failed to load some of: $MODULES"
fi

# Stop cleanly ~10 minutes before the Slurm wall time so partial results are saved
time_limit_args=()
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  end=$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null || true)
  if [[ -n "$end" ]] && end_s=$(date -d "$end" +%s 2>/dev/null); then
    time_limit_args=(--time-limit $(( end_s - $(date +%s) - 600 )))
  fi
fi

args=(
  --results-dir "$RESULTS_DIR"
  --label "$label"
  --reps "${REPS:-10}"
  --papi-reps "${PAPI_REPS:-3}"
  --phases "${PHASES:-build,tune,sweep,papi}"
  --algorithms "${ALGORITHMS:-fa2,summary,query,hybrid}"
  --max-n "${MAX_N:-1048576}"
  --papi
  "${time_limit_args[@]}"
)
[[ -n "${MAX_REP_SECONDS:-}" ]] && args+=(--max-rep-seconds "$MAX_REP_SECONDS")
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS:-}" ]] && args+=($EXTRA_ARGS)

echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-none}  Repo: $REPO"
python3 "$REPO/python_scripts/benchmark.py" "${args[@]}"
status=$?

# Exit code 3 means "stopped at the time limit, rerun to resume"
if [[ $status -eq 3 && "${RESUBMIT:-0}" == "1" && -n "${SLURM_JOB_ID:-}" ]]; then
  echo "Resubmitting to continue..."
  (cd "$REPO" && sbatch --export=ALL "slurm_jobs/${label}.slurm")
fi
exit $status
