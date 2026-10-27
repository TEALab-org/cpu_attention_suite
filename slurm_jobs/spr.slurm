#!/bin/bash
# Stampede3 Sapphire Rapids (Xeon CPU Max 9480) node (112 cores). Submit from the repository root:
#   sbatch slurm_jobs/spr.slurm
#   sbatch --export=ALL,RESUBMIT=1 slurm_jobs/spr.slurm   # keep resubmitting until done
#   sbatch --export=ALL,PASS=backward,RESUBMIT=1 slurm_jobs/spr.slurm   # backward pass
#   sbatch --export=ALL,SPARSITY=1,RESUBMIT=1 slurm_jobs/spr.slurm   # sparsity experiment (add PASS=backward)
# See slurm_jobs/run_benchmark.sh for the settings that can be overridden.
#SBATCH -J attn_spr
#SBATCH -p spr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 112
#SBATCH -t 04:00:00
#SBATCH -o attn_spr.%j.out
#SBATCH -e attn_spr.%j.out
##SBATCH -A <your-allocation>

bash "${SLURM_SUBMIT_DIR}/slurm_jobs/run_benchmark.sh" spr
