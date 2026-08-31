"""Amortization-at-scale demo (tab:scale, fig:scale, fig:map).

One GBC network is trained on a STANDARDISED prior-predictive regime and then
applied to every CONUS station by a forward pass (location-scale standardisation:
the GEV is location-scale for fixed xi, and xi is scale-invariant, so a single
network amortises across climates). We report, over all stations:
  - wall-clock: GBC (one training + N forward passes) vs per-station MCMC and MLE;
  - calibration at scale: leave-one-year-out predictive coverage and CRPS,
    aggregated over thousands of held-out station-years;
  - agreement of the amortised GBC z_100 with per-station MLE / MCMC.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The gbc package sits at ROOT/gbc/gbc in the paper tree and one level higher
# when these drivers ship inside the gbc repository itself; add whichever parent
# actually holds it, so a driver runs unchanged from either checkout.
for _cand in (os.path.join(ROOT, "gbc"), ROOT, os.path.dirname(ROOT)):
    if os.path.isfile(os.path.join(_cand, "gbc", "__init__.py")):
        sys.path.insert(0, _cand)
        break
from gbc.evt import gev_quantile, gev_return_level, _block_maxima_summary
from gbc.iqn import train_iqn, predict_iqn, sample_iqn
from gbc.evt_inference import return_level_ci_delta, gev_mcmc, fit_stationary_gev
from gbc.metrics import crps_samples

STAMP = "2026-06-17"
N_RET = 100
MIN_YEARS = 50
_LEDGER = os.path.join(ROOT, "results", "numbers.txt")


def ledger(value, ident, fmt="{:.4g}"):
    with open(_LEDGER, "a") as fh:
        fh.write(f"{fmt.format(value)}\trun_scale.py\t{ident}\t{STAMP}\n")
    return value


# ── amortised estimators: ONE network, applied to every station ───────────────

def _sim_standardised(n_sim, target, rng, N=N_RET):
    """Prior-predictive GEV datasets, each standardised by its own sample mean/sd.

    target='rl'   -> y = standardised z_N(theta)   (functional posterior)
    target='draw' -> y = a standardised fresh GEV draw  (predictive)
    """
    X, y = [], []
    for _ in range(n_sim):
        mu = rng.normal(0.0, 1.5)
        sigma = float(np.exp(rng.normal(0.0, 0.4)))
        xi = float(np.clip(rng.normal(0.0, 0.3), -0.8, 0.5))
        n = int(rng.integers(MIN_YEARS, 130))
        data = gev_quantile(rng.uniform(size=n), mu, sigma, xi)
        m, s = float(data.mean()), float(data.std() + 1e-9)
        X.append(_block_maxima_summary((data - m) / s))
        if target == "rl":
            y.append((gev_return_level(N, mu, sigma, xi) - m) / s)
        else:
            y.append((float(gev_quantile(rng.uniform(), mu, sigma, xi)) - m) / s)
    return np.array(X), np.array(y)


def train_amortised(target, n_sim=60000, epochs=2000, seed=1):
    X, y = _sim_standardised(n_sim, target, np.random.default_rng(seed))
    return train_iqn(X, y, epochs=epochs, seed=seed)


def gbc_rl(trained, x, alpha=0.10):
    m, s = float(np.mean(x)), float(np.std(x) + 1e-9)
    model, xm, xs, ym, ys = trained
    q = predict_iqn(model, _block_maxima_summary((np.asarray(x, float) - m) / s).reshape(1, -1),
                    xm, xs, ym, ys, np.array([alpha / 2, 0.5, 1 - alpha / 2])).ravel()
    return {"zN": m + s * q[1], "lo": m + s * q[0], "hi": m + s * q[2]}


def gbc_predict_samples(trained, x, B=500):
    m, s = float(np.mean(x)), float(np.std(x) + 1e-9)
    model, xm, xs, ym, ys = trained
    return m + s * sample_iqn(model, _block_maxima_summary((np.asarray(x, float) - m) / s).reshape(1, -1),
                              xm, xs, ym, ys, B=B).ravel()


def authoritative_timing():
    """Timings from ``run_timing.py`` if it has been run, else None.

    Wall-clock is a property of a machine, not of a method, so the published
    numbers come from a recorded single-threaded benchmark on one exclusive node
    (``experiments/timing.slurm``) rather than from whatever laptop happened to
    build the tables.  The statistical results below are machine-independent and
    are always measured here.
    """
    path = os.path.join(ROOT, "results", "timing.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        t = json.load(fh)
    return {"train_s": t["train_s_rl"],
            "gbc_s_per": t["gbc_ms_per_station"] / 1000.0,
            "mcmc_s_per": t["mcmc_ms_per_station"] / 1000.0,
            "mle_s_per": t["mle_ms_per_station"] / 1000.0,
            "source": (f"{t['provenance']['cpu_model']}, 1 thread, "
                       f"{t['provenance']['host']}, {t['provenance']['stamp']}")}


def main():
    df = pd.read_csv(os.path.join(ROOT, "data", "conus_tmax_maxima.csv"), index_col="year")
    cols = [c for c in df.columns if df[c].notna().sum() >= MIN_YEARS]
    series = {c: df[c].dropna().values.astype(float) for c in cols}
    Nst = len(series)
    print(f"CONUS TMAX stations with >= {MIN_YEARS} yr: {Nst}")

    # 1. train the two amortised networks ONCE (timed)
    t = time.time(); rl_net = train_amortised("rl", seed=1); t_train_rl = time.time() - t
    t = time.time(); pr_net = train_amortised("draw", seed=2); t_train_pr = time.time() - t
    print(f"trained amortised RL net in {t_train_rl:.1f}s, predictive net in {t_train_pr:.1f}s")

    # 2. amortised GBC z_100 for every station by forward pass (timed)
    t = time.time()
    gbc = {c: gbc_rl(rl_net, x) for c, x in series.items()}
    t_gbc_fwd = time.time() - t
    print(f"GBC z100 for all {Nst} stations by forward pass: {t_gbc_fwd:.2f}s "
          f"({1000 * t_gbc_fwd / Nst:.1f} ms/station)")

    # 3. per-station MLE (all) and MCMC (subset) for the timing + agreement comparison
    t = time.time(); mle = {c: return_level_ci_delta(x, N_RET) for c, x in series.items()}
    t_mle = time.time() - t
    sub = list(series)[:25]
    t = time.time()
    mcmc = {}
    for c in sub:
        s = gev_mcmc(series[c], n_iter=12000, burn=4000, thin=4, seed=0)
        mcmc[c] = float(np.median([gev_return_level(N_RET, *p) for p in s]))
    t_mcmc_sub = time.time() - t
    t_mcmc_per = t_mcmc_sub / len(sub)
    print(f"MLE all: {t_mle:.1f}s; MCMC {len(sub)} stations: {t_mcmc_sub:.1f}s "
          f"({t_mcmc_per:.1f}s/station)")

    # 4. calibration at scale: leave-one-year-out predictive coverage + CRPS (amortised)
    hits = tot = 0
    crps = []
    n_years = sum(len(x) for x in series.values())
    t_loyo = time.time()
    for si, (c, x) in enumerate(series.items()):
        if si % 10 == 0 and si:
            el = time.time() - t_loyo
            print(f"  LOYO {si}/{len(series)} stations, {tot}/{n_years} years, "
                  f"{el:.0f}s elapsed, ~{el * (n_years / max(tot, 1) - 1):.0f}s left",
                  flush=True)
        for i in range(len(x)):
            samp = gbc_predict_samples(pr_net, np.delete(x, i), B=400)
            lo, hi = np.quantile(samp, [0.05, 0.95])
            hits += int(lo <= x[i] <= hi); tot += 1
            crps.append(crps_samples(np.array([x[i]]), samp.reshape(-1, 1)))
    cov_scale = hits / tot
    crps_scale = float(np.mean(crps))
    print(f"LOYO calibration over {tot} station-years: coverage={cov_scale:.3f}, CRPS={crps_scale:.3f}")

    # 5. agreement amortised-GBC vs MLE / MCMC
    zg = np.array([gbc[c]["zN"] for c in series]); zm = np.array([mle[c]["zN"] for c in series])
    r_mle = float(np.corrcoef(zg, zm)[0, 1]); mad_mle = float(np.median(np.abs(zg - zm)))
    zg_s = np.array([gbc[c]["zN"] for c in sub]); zc_s = np.array([mcmc[c] for c in sub])
    r_mcmc = float(np.corrcoef(zg_s, zc_s)[0, 1]); mad_mcmc = float(np.median(np.abs(zg_s - zc_s)))
    width_gbc = float(np.median([gbc[c]["hi"] - gbc[c]["lo"] for c in series]))
    print(f"agreement: r(GBC,MLE)={r_mle:.3f} MAD={mad_mle:.2f}C; "
          f"r(GBC,MCMC)={r_mcmc:.3f} MAD={mad_mcmc:.2f}C")

    # Marginal cost is the honest amortisation metric. Total wall-clock favours GBC
    # only once the one-off training is amortised, i.e. beyond the break-even count
    # N* = t_train / t_mcmc_per; below it (e.g. 112 stations) MCMC's total is smaller.
    bench = authoritative_timing()
    if bench is not None:
        print(f"\nusing recorded benchmark timings ({bench['source']}); "
              f"the in-process figures above are this machine's and are not published")
        t_train_rl = bench["train_s"]
        t_gbc_fwd = bench["gbc_s_per"] * Nst
        t_mcmc_per = bench["mcmc_s_per"]
        t_mle = bench["mle_s_per"] * Nst
    else:
        print("\nWARNING: results/timing.json absent; publishing this machine's "
              "timings. Run experiments/run_timing.py on the benchmark node.")
    gbc_total = t_train_rl + t_gbc_fwd
    mcmc_total = t_mcmc_per * Nst
    ms_gbc = 1000 * t_gbc_fwd / Nst
    marg_ratio = t_mcmc_per / (t_gbc_fwd / Nst)
    n_star = t_train_rl / (t_mcmc_per - t_gbc_fwd / Nst)
    print(f"\nfor {Nst} stations: GBC one-off train {t_train_rl:.0f}s + {t_gbc_fwd:.2f}s forward; "
          f"MCMC {mcmc_total:.0f}s")
    print(f"marginal cost: GBC {ms_gbc:.2f} ms/station vs MCMC {1000*t_mcmc_per:.0f} ms/station "
          f"({marg_ratio:.0f}x lower); break-even at N*={n_star:.0f} stations")

    for v, k in [(Nst, "scale:n_stations"), (t_train_rl, "scale:t_train"),
                 (t_train_pr, "scale:t_train_pred"), (len(sub), "scale:n_mcmc"),
                 (ms_gbc, "scale:ms_per_station"), (t_mcmc_per, "scale:mcmc_s_per"),
                 (gbc_total, "scale:gbc_total_s"), (mcmc_total, "scale:mcmc_total_s"),
                 (marg_ratio, "scale:marginal_ratio"), (n_star, "scale:crossover"),
                 (cov_scale, "scale:coverage"),
                 (crps_scale, "scale:crps"), (tot, "scale:station_years"),
                 (r_mle, "scale:r_mle"), (mad_mle, "scale:mad_mle"),
                 (r_mcmc, "scale:r_mcmc"), (width_gbc, "scale:gbc_ci_width")]:
        ledger(v, k)

    # data for figures: map (lat, lon, z100) + timing.  Coordinates come from the
    # small per-station metadata file when it is present, so the study runs
    # without the 36 MB GHCN inventory; the inventory is the fallback for a
    # freshly fetched station set that the metadata file does not yet cover.
    meta = os.path.join(ROOT, "data", "conus_tmax_meta.csv")
    inv = os.path.join(ROOT, "data", "cache", "ghcnd-inventory.txt")
    if os.path.exists(meta):
        m = pd.read_csv(meta)
        loc = {r.station: (r.lat, r.lon) for r in m.itertuples()}
    elif os.path.exists(inv):
        loc = {}
        for line in open(inv):
            if line[31:35].strip() == "TMAX":
                loc[line[0:11]] = (float(line[12:20]), float(line[21:30]))
    else:
        raise FileNotFoundError(
            f"need station coordinates: expected {meta} or {inv}")
    rows = [(c, loc[c][0], loc[c][1], gbc[c]["zN"]) for c in series if c in loc]
    pd.DataFrame(rows, columns=["station", "lat", "lon", "z100"]).to_csv(
        os.path.join(ROOT, "results", "scale_map.csv"), index=False)
    pd.DataFrame({"method": ["GBC (amortised)", "MCMC (per-station)", "MLE (per-station)"],
                  "total_s": [gbc_total, mcmc_total, t_mle],
                  "per_station_s": [t_gbc_fwd / Nst, t_mcmc_per, t_mle / Nst]}).to_csv(
        os.path.join(ROOT, "results", "scale_timing.csv"), index=False)

    # tab:scale  (honest amortisation framing: marginal cost + calibration + agreement,
    # not a raw wall-clock speedup, which only favours GBC beyond N* stations)
    with open(os.path.join(ROOT, "tab", "scale.tex"), "w") as fh:
        fh.write("% generated by experiments/run_scale.py -- do not edit by hand\n")
        fh.write("\\begin{tabular}{lcc}\n\\toprule\n")
        fh.write(f"{Nst} CONUS stations & GBC-QNN (amortised) & per-station MCMC \\\\\n\\midrule\n")
        fh.write(f"Procedure & one network, {Nst} forward passes & "
                 f"{len(sub)} chains (timed subset) \\\\\n")
        fh.write(f"Marginal cost per station & {ms_gbc:.2f}\\,ms & {1000*t_mcmc_per:.0f}\\,ms "
                 f"(${marg_ratio:.0f}\\times$) \\\\\n")
        fh.write(f"One-off training & {t_train_rl:.0f}\\,s (return-level net) & n/a \\\\\n")
        fh.write(f"LOYO coverage, GBC (nom.\\ 0.90) & \\multicolumn{{2}}{{c}}{{{cov_scale:.2f} "
                 f"over {tot} station-years}} \\\\\n")
        fh.write(f"LOYO CRPS, GBC ($^\\circ$C) & \\multicolumn{{2}}{{c}}{{{crps_scale:.2f}}} \\\\\n")
        fh.write(f"$z_{{100}}$ agreement & \\multicolumn{{2}}{{c}}{{$r={r_mle:.2f}$ vs MLE "
                 f"({Nst}), $r={r_mcmc:.2f}$ vs MCMC ({len(sub)})}} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print("wrote tab/scale.tex, results/scale_map.csv, results/scale_timing.csv")


if __name__ == "__main__":
    main()
