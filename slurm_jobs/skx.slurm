#!/bin/bash
# Stampede3 Skylake (Xeon Platinum 8160) node (48 cores). Submit from the repository root:
#   sbatch slurm_jobs/skx.slurm
#   sbatch --export=ALL,RESUBMIT=1 slurm_jobs/skx.slurm   # keep resubmitting until done
#   sbatch --export=ALL,PASS=backward,RESUBMIT=1 slurm_jobs/skx.slurm   # backward pass
# See slurm_jobs/run_benchmark.sh for the settings that can be overridden.
#SBATCH -J attn_skx
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH -t 02:00:00
#SBATCH -o attn_skx.%j.out
#SBATCH -e attn_skx.%j.out
##SBATCH -A <your-allocation>

bash "${SLURM_SUBMIT_DIR}/slurm_jobs/run_benchmark.sh" skx
