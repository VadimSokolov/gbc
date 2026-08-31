# Empirical Lab Notebook — GBC-EVT (WSC 2026 con126s1)

All paper numbers must trace to a run recorded here and to `results/numbers.txt`.
Estimators live in the `gbc` library (`gbc.evt_inference`, v0.4.0, pushed 2026-06-17).
This replaces the fabricated results in the as-accepted draft (see `audit/`): the original
tables were hardcoded literals in `make_figures.py` with no experiment behind them.

## 0. Data

| File | Source | Content |
|------|--------|---------|
| `data/pnw_jja_maxima.csv` | NOAA GHCN-Daily (access CSVs), fetched 2026-06-17 | JJA annual TMAX maxima (degC), 5 PNW stations |
| `data/hadcrut5_annual.csv` | HadCRUT5.0.2.0 global annual summary | global mean anomaly (degC), 1850-2025 |

Script: `experiments/fetch_data.py`. TMAX is tenths-degC in source; station-year kept if >= 80 valid JJA days.
Real records (longer than the draft's fabricated spans): SEA 1948-2025 (n=78), PDX 1938-2025 (n=88),
GEG 1890-2025 (n=136), EUG 1938-2025 (n=88), PDT 1928-2025 (n=98).
**Validation:** SEA 2021 JJA max = 42.2 degC, reproducing the 2021 heat-dome record exactly.

## 1. Methods (gbc.evt_inference)

| Method | Function | Role |
|--------|----------|------|
| Stationary GEV MLE | `fit_stationary_gev`, `return_level_ci_delta` | baseline |
| Non-stationary GEV MLE | `fit_ns_gev`, `ns_params_at` | baseline (HadCRUT5 covariate) |
| Hill | `hill` | baseline tail index |
| Bayes GEV MCMC | `gev_mcmc` | RW-Metropolis, priors mu~N(ref,5^2), log sig~N(ref,.5^2), xi~N(0,.3^2)[-1,2] |
| GBC-QNN | `train_functional_iqn` + `gbc_return_level_posterior`; `gbc_crps_coverage_loyo` | ours |
| GBC + Horseshoe | `simulate_hierarchical_training_data` + `horseshoe_posterior` | ours (spatial) |

### 1.1 GBC-QNN hyperparameters (validated)
IQN: hdim/nh/lr per `gbc.iqn` defaults. Predictive IQN trains on `simulate_gev_training_data`
(target = fresh GEV draw); functional IQN targets `z_N(theta)`. n_sim=20000-40000, epochs=1500.
**Undertraining check:** epochs<=700 gives diffuse predictive (CRPS~16, coverage 1.0); epochs>=1500
calibrates (predictive q[.05,.5,.95] match empirical within 0.5 degC).

## 2. Experiment plan (paper table/figure -> experiment)

| Paper object | Experiment | Script | Compute |
|--------------|-----------|--------|---------|
| Table `tab:rl` + Fig survivor/comparison | SEA 6-method return levels, LOYO CRPS/coverage | `run_pnw.py` | local CPU |
| Table `tab:spatial` + Fig horseshoe | 5-station xi MLE vs horseshoe, kappa, trends | `run_pnw.py` | local CPU |
| Table `tab:attr` + Fig records | heat-dome record probabilities, 4 models | `run_pnw.py` | local CPU |
| Fig threshold, Fig meu | GBC threshold posterior; MEU activation threshold | `run_pnw.py` | local CPU |
| Table `tab:sim` | sim study: R reps, GEV(35,3,0.1), n=70; NS and threshold sims | `run_sim.py --rep` | **Hopper** CPU array (Intel hop nodes, no GPU) |

## 3. Real results — Seattle return levels (tab:rl) — `run_pnw.py`, 2026-06-17

100-yr return level (degC) under the 1950 vs 2023 climate; CI Width = 90% interval width for
z100 at 2023; Coverage/CRPS = leave-one-year-out predictive (held-out annual maximum).

| Method | RL 1950 | RL 2023 | CI Width | Coverage | CRPS |
|--------|---------|---------|----------|----------|------|
| Stationary GEV MLE | 40.4 | 40.4 | 1.9 | 0.92 | 1.48 |
| NS GEV MLE | 39.2 | 41.4 | 4.4 | 0.88 | 1.41 |
| Bayes GEV MCMC | 40.7 | 40.7 | 2.5 | 0.92 | 1.48 |
| Hill | 39.9 | 39.9 | --- | --- | --- |
| **GBC-QNN (ours)** | 38.3 | 42.0 | 4.8 | 0.86 | 1.49 |
| **GBC + Horseshoe** | 38.3 | 42.2 | 2.1 | 0.86 | 1.49 |

Method notes (honest design, propagated to the caption): stationary methods use the same level both
years; NS methods (NS GEV MLE, GBC) shift with the HadCRUT5 covariate (T_1950=-0.227, T_2023=+1.100).
GBC-QNN return level = detrend the maxima to the target climate via the estimated trend, then apply
the validated stationary GBC functional (`train_functional_iqn`); CRPS/coverage from
`gbc_crps_coverage_loyo`. GBC+Horseshoe propagates the spatially-pooled xi posterior (tab:spatial,
tighter than a single-station fit) with a data bootstrap for (mu,sigma). Hill RL via the Weissman
high-quantile estimator (k=floor(n/4)); CI/coverage n/a. The amortized GBC interval (4.8) is wider
than the MLE delta interval (1.9, known to be overconfident for return levels); spatial pooling
sharpens it to 2.1, comparable to the Bayes MCMC interval (2.5), at equal predictive skill.

## 3b. Real results — spatial horseshoe (tab:spatial)

| Station | xi (MLE) | xi (HS) | 90% CI | kappa | Trend (degC/dec) |
|---------|----------|---------|--------|-------|------|
| SEA | -0.23 | -0.23 | [-0.29,-0.18] | 0.62 | +0.50 |
| PDX | -0.24 | -0.24 | [-0.29,-0.19] | 0.57 | +0.46 |
| GEG | -0.21 | -0.21 | [-0.28,-0.14] | 0.57 | +0.05 |
| EUG | -0.31 | -0.30 | [-0.37,-0.23] | 0.50 | +0.42 |
| PDT | -0.13 | -0.15 | [-0.25,-0.04] | 0.51 | +0.13 |

**Key findings (overturn the fabricated draft):**
1. **All five stations are short-tailed** (xi in [-0.31,-0.13], every 90% CI strictly negative). The
   draft's "heavier-tailed continental stations (Spokane/Pendleton)" claim is FALSE: GEG/PDT are the
   *least* negative but still short-tailed. The horseshoe shrinks little (kappa~0.5-0.6) because the
   stations genuinely agree, which is itself the finding.
2. **The 2021 heat dome (SEA 42.2 degC) is ~the 100-yr level under the 2023 climate** (GBC RL2023
   ~42.0-42.2), i.e. roughly a 1-in-100-yr event *today* but far rarer under the 1950 climate
   (RL1950 ~38.3) -- a clean non-stationary attribution signal.
3. **CIs are much narrower than the draft** (MLE width 1.9 vs draft's 6.2) because short tails are
   well constrained.

### Status / next
- [x] Data + methods (gbc v0.4.0); tab:rl + tab:spatial REAL, traced to ledger + tab/*.tex.
- [ ] tab:attr (2021 record/attribution probabilities, 4 models) + 6 figures from real fits.
- [ ] `run_sim.py` on Hopper (Intel CPU array) -> tab:sim.
- [ ] Rewrite manuscript numbers/narrative from ledger; finish math fixes (done: Stein/Hill/threshold/horseshoe/record); fix citations; RTC; latexdiff; final audit.
