#!/bin/bash
#SBATCH --job-name=gbcevt-timing
#SBATCH --partition=normal
#SBATCH --constraint=intel
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-timing-%j.out
#SBATCH --error=slurm-timing-%j.out

# Single-threaded on one Intel hop node: per-station cost is a serial quantity
# and both arms of the comparison must run on the same core.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# submit from the directory holding experiments/ and data/
cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "node: $(hostname)"
grep -m1 'model name' /proc/cpuinfo
python3 -u experiments/run_timing.py
