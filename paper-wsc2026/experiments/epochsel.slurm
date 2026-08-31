#!/bin/bash
#SBATCH --job-name=gbcevt-epochsel
#SBATCH --partition=normal
#SBATCH --constraint=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-epochsel-%j.out
#SBATCH --error=slurm-epochsel-%j.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# submit from the directory holding experiments/ and data/
cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "node: $(hostname)"; grep -m1 'model name' /proc/cpuinfo
python3 -u experiments/run_epochsel.py
