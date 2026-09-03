#!/bin/bash
#SBATCH --job-name=gbcevt-suite
#SBATCH --partition=normal
#SBATCH --constraint=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00      # all four drivers measured 50 min; 3.6x headroom
#SBATCH --output=slurm-suite-%j.out
#SBATCH --error=slurm-suite-%j.out

# The table drivers named by $DRIVERS (default: all four), run SEQUENTIALLY
# in one job on purpose.  Each begins by
# rewriting results/numbers.txt to drop its own previous rows (read-filter-write),
# so two of them running at once can silently clobber each other's ledger entries.
# One job is the cheapest way to make that impossible.
#
# run_scale.py is last because it reads results/timing.json, which the separate
# exclusive-node timing job produces; this job is submitted dependent on it.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "node: $(hostname)"; grep -m1 'model name' /proc/cpuinfo
for d in ${DRIVERS:-run_pnw run_attr run_heavytail run_scale}; do
  echo "=================== $d ==================="
  python3 -u experiments/$d.py || { echo "FAILED: $d"; exit 1; }
  echo "=================== $d DONE =============="
done

# Figures are part of the paper, so the suite has to redraw them: they read
# results/numbers.txt and results/scale_map.csv, and leaving this to a manual
# step is how a figure ends up quoting a superseded ratio.  Skipped only when
# a partial $DRIVERS run did not refresh what the figures read.
if [ -z "${DRIVERS:-}" ] || echo "${DRIVERS}" | grep -q run_scale; then
  echo "=================== make_figures ==================="
  python3 -u make_figures.py || { echo "FAILED: make_figures"; exit 1; }
  echo "=================== make_figures DONE =============="
fi
echo "ALL DRIVERS DONE"
