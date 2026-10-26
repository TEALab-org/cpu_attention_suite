#!/bin/bash
# Stampede3 Ice Lake (Xeon Platinum 8380) node (80 cores). Submit from the repository root:
#   sbatch slurm_jobs/icx.slurm
#   sbatch --export=ALL,RESUBMIT=1 slurm_jobs/icx.slurm   # keep resubmitting until done
#   sbatch --export=ALL,PASS=backward,RESUBMIT=1 slurm_jobs/icx.slurm   # backward pass
# See slurm_jobs/run_benchmark.sh for the settings that can be overridden.
#SBATCH -J attn_icx
#SBATCH -p icx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 80
#SBATCH -t 04:00:00
#SBATCH -o attn_icx.%j.out
#SBATCH -e attn_icx.%j.out
##SBATCH -A <your-allocation>

bash "${SLURM_SUBMIT_DIR}/slurm_jobs/run_benchmark.sh" icx
