"""Hardware-pinned wall-clock benchmark with MCMC convergence diagnostics.

Supersedes the timing block inside ``run_scale.py``.  Three things that block a
defensible amortization claim are supplied here and nowhere else:

  1. *Hardware and software provenance.*  The CPU model, node, thread pinning,
     and library versions are recorded with the numbers, so a 422x marginal-cost
     ratio is attached to a machine rather than floating free.
  2. *MCMC convergence diagnostics.*  A wall-clock cost per chain means nothing
     unless the chain converged.  We run four chains per station, report
     rank-normalized split-Rhat and bulk ESS for (mu, sigma, xi) and for the
     derived z_100, and report cost per effective draw, which is the unit in
     which an MCMC-vs-forward-pass comparison is actually fair.
  3. *Repeated trials.*  Every timed quantity is measured over several repeats
     and summarized by its median, with the spread reported.

Both arms are timed in one process on one node with one thread, so the ratio is
like-for-like.  Run under SLURM on an Intel ``hop`` node:

    sbatch experiments/timing.slurm
"""
import json
import os
import platform
import socket
import subprocess
import sys
import time

# Pin to a single thread BEFORE importing numpy/torch: per-station cost is a
# serial quantity, and thread-count drift is the usual way these ratios rot.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402
import torch                                                        # noqa: E402

torch.set_num_threads(1)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The gbc package sits at ROOT/gbc/gbc in the paper tree and one level higher
# when these drivers ship inside the gbc repository itself; add whichever parent
# actually holds it, so a driver runs unchanged from either checkout.
for _cand in (os.path.join(ROOT, "gbc"), ROOT, os.path.dirname(ROOT)):
    if os.path.isfile(os.path.join(_cand, "gbc", "__init__.py")):
        sys.path.insert(0, _cand)
        break
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from gbc.evt import gev_return_level                                # noqa: E402
from gbc.evt_inference import return_level_ci_delta, gev_mcmc       # noqa: E402
from gbc.diagnostics import rhat, ess_bulk                          # noqa: E402
from run_scale import train_amortised, gbc_rl                       # noqa: E402

STAMP = time.strftime("%Y-%m-%d")
N_RET = 100
MIN_YEARS = 50
N_MCMC_STATIONS = 25          # the subset run_scale.py chains
N_CHAINS = 4
MCMC_ITER, MCMC_BURN, MCMC_THIN = 12000, 4000, 4
REPEATS_FAST = 20             # forward pass / MLE
REPEATS_SLOW = 3              # MCMC
_LEDGER = os.path.join(ROOT, "results", "numbers.txt")


def ledger(value, ident, fmt="{:.4g}"):
    with open(_LEDGER, "a") as fh:
        fh.write(f"{fmt.format(value)}\trun_timing.py\t{ident}\t{STAMP}\n")
    return value


def _cpu_model():
    """Best-effort CPU model string on Linux, macOS, or anything else."""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return platform.processor() or "unknown"


def provenance():
    return {
        "stamp": STAMP,
        "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID", "none"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST", "none"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION", "none"),
        "cpu_model": _cpu_model(),
        "cpu_count_visible": os.cpu_count(),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", "unset"),
        "threads_torch": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
    }


def _median_spread(times, n_units):
    """Median and min-max per-unit cost in seconds, from repeated total times."""
    per = np.asarray(times, float) / n_units
    return float(np.median(per)), float(per.min()), float(per.max())


def main():
    prov = provenance()
    print("=" * 74)
    for k, v in prov.items():
        print(f"{k:>20}: {v}")
    print("=" * 74)

    df = pd.read_csv(os.path.join(ROOT, "data", "conus_tmax_maxima.csv"),
                     index_col="year")
    cols = [c for c in df.columns if df[c].notna().sum() >= MIN_YEARS]
    series = {c: df[c].dropna().values.astype(float) for c in cols}
    n_st = len(series)
    print(f"\nstations: {n_st}")

    # ── 1. one-off amortized training (the fixed cost) ───────────────────────
    t = time.perf_counter(); rl_net = train_amortised("rl", seed=1)
    t_train_rl = time.perf_counter() - t
    print(f"train amortized RL net: {t_train_rl:.1f}s")

    # ── 2. marginal cost of a station: GBC forward pass ──────────────────────
    fwd = []
    for _ in range(REPEATS_FAST):
        t = time.perf_counter()
        _ = {c: gbc_rl(rl_net, x) for c, x in series.items()}
        fwd.append(time.perf_counter() - t)
    gbc_s, gbc_lo, gbc_hi = _median_spread(fwd, n_st)
    print(f"GBC forward pass : {1000*gbc_s:.3f} ms/station "
          f"[{1000*gbc_lo:.3f}, {1000*gbc_hi:.3f}]")

    # ── 3. marginal cost of a station: per-station MLE ───────────────────────
    mle_t = []
    for _ in range(REPEATS_FAST):
        t = time.perf_counter()
        _ = {c: return_level_ci_delta(x, N_RET) for c, x in series.items()}
        mle_t.append(time.perf_counter() - t)
    mle_s, mle_lo, mle_hi = _median_spread(mle_t, n_st)
    print(f"per-station MLE  : {1000*mle_s:.3f} ms/station "
          f"[{1000*mle_lo:.3f}, {1000*mle_hi:.3f}]")

    # ── 4. marginal cost of a station: per-station MCMC, one chain ───────────
    sub = list(series)[:N_MCMC_STATIONS]
    mcmc_t = []
    for _ in range(REPEATS_SLOW):
        t = time.perf_counter()
        for c in sub:
            gev_mcmc(series[c], n_iter=MCMC_ITER, burn=MCMC_BURN,
                     thin=MCMC_THIN, seed=0)
        mcmc_t.append(time.perf_counter() - t)
    mcmc_s, mcmc_lo, mcmc_hi = _median_spread(mcmc_t, len(sub))
    print(f"per-station MCMC : {1000*mcmc_s:.1f} ms/station (1 chain) "
          f"[{1000*mcmc_lo:.1f}, {1000*mcmc_hi:.1f}]")

    # ── 5. MCMC convergence diagnostics: 4 chains per station ────────────────
    names = ["mu", "sigma", "xi", "z100"]
    per_station = []
    for c in sub:
        chains = []
        for k in range(N_CHAINS):
            s = gev_mcmc(series[c], n_iter=MCMC_ITER, burn=MCMC_BURN,
                         thin=MCMC_THIN, seed=100 + k)
            z = np.array([gev_return_level(N_RET, *p) for p in s])
            chains.append(np.column_stack([s, z]))
        arr = np.stack(chains)                      # (chains, draws, 4)
        row = {"station": c, "draws_per_chain": arr.shape[1]}
        for j, nm in enumerate(names):
            row[f"rhat_{nm}"] = rhat(arr[:, :, j])
            row[f"ess_{nm}"] = ess_bulk(arr[:, :, j])
        per_station.append(row)
    diag = pd.DataFrame(per_station)
    diag.to_csv(os.path.join(ROOT, "results", "mcmc_diagnostics.csv"),
                index=False)

    worst_rhat = float(max(diag[f"rhat_{n}"].max() for n in names))
    min_ess = float(min(diag[f"ess_{n}"].min() for n in names))
    med_ess_z = float(diag["ess_z100"].median())
    n_unconverged = int((diag[[f"rhat_{n}" for n in names]] > 1.01).any(axis=1).sum())
    print(f"\nMCMC diagnostics over {len(sub)} stations x {N_CHAINS} chains "
          f"({diag['draws_per_chain'].iloc[0]} draws/chain):")
    print(f"  worst rank-normalized split-Rhat : {worst_rhat:.4f}")
    print(f"  min bulk ESS (any parameter)     : {min_ess:.0f}")
    print(f"  median bulk ESS for z100         : {med_ess_z:.0f}")
    print(f"  stations with any Rhat > 1.01    : {n_unconverged} / {len(sub)}")

    # cost per effective draw: the fair unit for MCMC-vs-forward-pass
    s_per_eff = mcmc_s / med_ess_z
    print(f"  MCMC cost per effective z100 draw: {1000*s_per_eff:.4f} ms")

    # ── 6. the amortization arithmetic ───────────────────────────────────────
    ratio = mcmc_s / gbc_s
    breakeven = t_train_rl / (mcmc_s - gbc_s)
    print(f"\nmarginal-cost ratio MCMC/GBC : {ratio:.1f}x")
    print(f"break-even locations         : {breakeven:.0f}")

    out = {
        "provenance": prov,
        "n_stations": n_st,
        "n_mcmc_stations": len(sub),
        "n_chains": N_CHAINS,
        "mcmc_config": {"n_iter": MCMC_ITER, "burn": MCMC_BURN,
                        "thin": MCMC_THIN,
                        "draws_per_chain": int(diag["draws_per_chain"].iloc[0])},
        "repeats": {"fast": REPEATS_FAST, "slow": REPEATS_SLOW},
        "train_s_rl": t_train_rl,
        "gbc_ms_per_station": 1000 * gbc_s,
        "gbc_ms_range": [1000 * gbc_lo, 1000 * gbc_hi],
        "mle_ms_per_station": 1000 * mle_s,
        "mle_ms_range": [1000 * mle_lo, 1000 * mle_hi],
        "mcmc_ms_per_station": 1000 * mcmc_s,
        "mcmc_ms_range": [1000 * mcmc_lo, 1000 * mcmc_hi],
        "marginal_ratio": ratio,
        "breakeven_locations": breakeven,
        "mcmc_worst_rhat": worst_rhat,
        "mcmc_min_ess": min_ess,
        "mcmc_median_ess_z100": med_ess_z,
        "mcmc_stations_rhat_above_1p01": n_unconverged,
        "mcmc_ms_per_effective_z100_draw": 1000 * s_per_eff,
    }
    with open(os.path.join(ROOT, "results", "timing.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    ledger(t_train_rl, "timing:t_train", "{:.1f}")
    ledger(1000 * gbc_s, "timing:gbc_ms_per_station")
    ledger(1000 * mcmc_s, "timing:mcmc_ms_per_station", "{:.0f}")
    ledger(1000 * mle_s, "timing:mle_ms_per_station")
    ledger(ratio, "timing:marginal_ratio", "{:.0f}")
    ledger(breakeven, "timing:breakeven_locations", "{:.0f}")
    ledger(worst_rhat, "timing:mcmc_worst_rhat", "{:.3f}")
    ledger(med_ess_z, "timing:mcmc_median_ess_z100", "{:.0f}")
    ledger(min_ess, "timing:mcmc_min_ess", "{:.0f}")
    print("\nwrote results/timing.json, results/mcmc_diagnostics.csv, ledger lines")


if __name__ == "__main__":
    main()
