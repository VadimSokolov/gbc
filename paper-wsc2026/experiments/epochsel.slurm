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

# One worker process per core, one BLAS thread each.  The cells are independent
# trainings, and full-batch steps on a single 45000-row tensor do not fill 16
# threads: the same sweep took 4h35m thread-parallel and 1h20m process-parallel.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export GBCEVT_JOBS=$SLURM_CPUS_PER_TASK

# submit from the directory holding experiments/ and data/
cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "node: $(hostname)"; grep -m1 'model name' /proc/cpuinfo
# -u is not optional here: stdout to a file is block-buffered, and a long job
# with no progress output is indistinguishable from a hung one.
python3 -u experiments/run_epochsel.py
