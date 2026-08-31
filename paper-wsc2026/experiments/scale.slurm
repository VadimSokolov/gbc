#!/bin/bash
#SBATCH --job-name=gbcevt-scale
#SBATCH --partition=normal
#SBATCH --constraint=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-scale-%j.out
#SBATCH --error=slurm-scale-%j.out

# Table 4 and Figure 1. Reads results/timing.json for the published wall-clock,
# so run experiments/run_timing.py on the benchmark node first.
# Exclusive node: the shared-node run was 43x slower per forward pass on a
# node other jobs had saturated (load 44/48).  8 threads help the two
# full-batch trainings; the leave-one-year-out loop is thread-insensitive.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "node: $(hostname)"; grep -m1 'model name' /proc/cpuinfo
python3 -u experiments/run_scale.py
