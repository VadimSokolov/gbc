#!/bin/bash
#SBATCH --job-name=gbcevt-scale
#SBATCH --partition=normal
#SBATCH --constraint=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-scale-%j.out
#SBATCH --error=slurm-scale-%j.out

# Table 4 and Figure 1. Reads results/timing.json for the published wall-clock,
# so run experiments/run_timing.py on the benchmark node first.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "node: $(hostname)"; grep -m1 'model name' /proc/cpuinfo
python3 -u experiments/run_scale.py
