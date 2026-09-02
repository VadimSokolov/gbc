# Reproduction bundle: Polson and Sokolov, WSC 2026

Everything behind *Generative Bayesian Computation for Extreme Value Theory with
Climate Applications* (Winter Simulation Conference 2026): the drivers that write
each table and figure, the station data they read, and the lab notebook recording
what was run when.

The estimators themselves live in the `gbc` package one directory up. Nothing in
here re-implements a method; these are drivers.

## Install and run

```bash
git clone https://github.com/VadimSokolov/gbc.git
cd gbc
pip install -e .
cd paper-wsc2026
python experiments/run_pnw.py          # writes tab/rl.tex, tab/spatial.tex
```

The drivers locate the package by walking up from their own location, so they run
from this directory whether or not `gbc` is installed.

## What produces what

| Paper object | Driver | Writes |
|---|---|---|
| Table 1 (Seattle return levels) | `experiments/run_pnw.py` | `tab/rl.tex` |
| Table 2 (horseshoe shrinkage of ξ) | `experiments/run_pnw.py` | `tab/spatial.tex` |
| Table 3 (2021 heat-dome conditional risk) | `experiments/run_attr.py` | `tab/attr.tex` |
| Table 4 (continental amortization) | `experiments/run_scale.py` | `tab/scale.tex`, `results/scale_map.csv` |
| Table 5 (heavy-tailed precipitation) | `experiments/run_heavytail.py` | `tab/heavytail.tex` |
| Table 6 (simulation study) | `experiments/run_sim.py` | `tab/sim.tex` |
| Figure 1 (map + cost curves) | `make_figures.py` | `fig/scale_panels.pdf` |
| Wall-clock and MCMC diagnostics | `experiments/run_timing.py` | `results/timing.json`, `results/mcmc_diagnostics.csv` |
| Training-budget sensitivity | `experiments/run_epochsel.py` | `results/epoch_selection.{csv,json}` |

`make_figures.py` reads `results/` and must run after `run_scale.py`.

Two scripts in `experiments/` are diagnostics rather than table drivers, kept
because they are the evidence behind claims in the lab notebook:
`attr_probe.py` measures where a body-only predictive network saturates, and
`tail_val.py` checks that the GPD tail head recovers the far tail it could not.

Every value printed in the paper is appended to `results/numbers.txt` as a
tab-separated `value / driver / identifier / date` line, so a number in a table
can be traced back to the run that produced it.

## Data

`data/` holds the derived station series, so the study reproduces without
re-contacting NOAA:

| File | Content |
|---|---|
| `pnw_jja_maxima.csv` | JJA annual TMAX maxima, 5 Pacific Northwest GHCN-Daily stations |
| `conus_tmax_maxima.csv` | JJA annual TMAX maxima, 112 CONUS stations with ≥ 50 years |
| `conus_prcp_maxima.csv` | annual daily-precipitation maxima, 110 CONUS stations |
| `conus_prcp_meta.csv` | station metadata for the precipitation set |
| `conus_tmax_meta.csv` | latitude and longitude for the 112 temperature stations |
| `hadcrut5_annual.csv` | HadCRUT5 global mean temperature anomaly, 1850 to 2025 |

The `experiments/fetch_*.py` scripts rebuild these from the GHCN-Daily access
CSVs and the HadCRUT5 annual summary. A station-year is kept when it has at least
80 valid daily observations in the season.

## Compute

The published timing benchmark, simulation and budget-sensitivity run were
produced on Intel `hop` nodes of the GMU Hopper cluster. Tables 1 through 5 used
the local CPU runs recorded in `experiments/empirical.md`; Table 4 imports the
pinned Hopper timing JSON. The drivers run unchanged on either machine. To
regenerate the complete artifact set on Hopper with one command:

```bash
bash experiments/reproduce_all.sh
```

The wrapper waits for these four jobs in sequence:

```bash
sbatch experiments/timing.slurm      # wall-clock + MCMC diagnostics, 1 thread, exclusive node
sbatch experiments/suite.slurm       # Tables 1, 2, 3, 4, 5 and Figure 1, drivers run in sequence
sbatch experiments/sim.slurm         # Table 6
sbatch experiments/epochsel.slurm    # training-budget sensitivity, 8 worker processes
```

`suite.slurm` runs its drivers sequentially on purpose: they all append to the
same `results/numbers.txt`, and concurrent drivers race on it. It takes the
driver list from `$DRIVERS`, so a single table can be rebuilt without the rest.

Run `timing.slurm` before `suite.slurm`: `run_scale.py` publishes the recorded
benchmark timings when `results/timing.json` is present and warns loudly when it
is not, so a laptop rebuild cannot quietly substitute a laptop's wall-clock.

`timing.slurm` requests an exclusive node and pins every library to one thread.
Both arms of the amortized-versus-MCMC comparison are then timed in one process
on one core, which is what makes the marginal-cost ratio a like-for-like
measurement rather than a comparison of two machines.

## Reproducibility caveats

- **Torch training is not bit-reproducible across machines or thread counts.**
  Re-running moves the second decimal of the CRPS and coverage cells of Table 1.
  Return levels are stable. Anyone re-running should regenerate the tables and
  re-check the prose against `results/numbers.txt` rather than assuming the
  printed values still hold.
- **The 20 subsamples per station in `run_heavytail.py` overlap**, so the
  station-clustered bootstrap, not the naive one, is the interval to quote.
- `run_pnw.py` retrains the leave-one-year-out predictive network on every
  invocation and takes the longest of the laptop-scale drivers.

## Citing

Cite the paper for the method and this repository for the code; see the top-level
`README.md` for the software entry.
