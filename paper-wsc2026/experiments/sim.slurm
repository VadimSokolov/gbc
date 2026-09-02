#!/bin/bash
#SBATCH --job-name=gbcevt-sim
#SBATCH --partition=normal
#SBATCH --constraint=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00      # measured 6 min; 10x headroom
#SBATCH --output=slurm-sim-%j.out
#SBATCH --error=slurm-sim-%j.out

# run_sim.py is one process, not an array: it trains a single amortised network
# and then loops 300 replicates over each of three tail regimes.  Its outputs are
# RMSE, bias and coverage, not wall-clock, so a shared node is fine and no
# exclusive allocation is needed.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "node: $(hostname)"; grep -m1 'model name' /proc/cpuinfo
# -u is not optional: block-buffered stdout makes a long job look hung.
python3 -u experiments/run_sim.py
