#!/bin/bash
set -euo pipefail

# Submit from any directory and wait until every artifact-producing job has
# completed.  Sequential submission protects results/numbers.txt from writers
# that replace their own ledger rows.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PAPER_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PAPER_ROOT"

sbatch --wait experiments/timing.slurm
sbatch --wait experiments/suite.slurm
sbatch --wait experiments/sim.slurm
sbatch --wait experiments/epochsel.slurm

echo "Reproduction complete: tables, Figure 1, timing diagnostics, simulation, and budget sensitivity."
