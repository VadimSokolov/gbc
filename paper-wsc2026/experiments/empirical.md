# Empirical Lab Notebook: GBC-EVT (WSC 2026 con126s1)

All paper numbers must trace to a run recorded here and to `results/numbers.txt`.
Estimators live in the `gbc` library (`gbc.evt_inference`, v0.6.0). **All results below
post-date the 2026-09-01 estimator corrections in section 0.1; anything produced under
v0.5.1 or earlier is superseded and must not be quoted.**
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

## 0.1 Estimator corrections, 2026-09-01 (supersedes every earlier number)

Three defects in `gbc` reached every published GBC number. Each passed the full
mechanical audit, because each was a case of code whose *name* matched the paper
while its *behaviour* did not. Fixed in v0.6.0; regression tests in
`gbc/tests/test_regressions.py` fail on v0.5.1 and pass on the fix.

| Defect | Was | Now | Measured effect |
|--------|-----|-----|-----------------|
| `loss.py` "monotonicity" penalty | one-sided penalty on the sign of `y - Q(tau)`; never compared two tau, so it could not detect a crossing, and it was active at the correct quantile | `relu(Q(tau_lo) - Q(tau_hi))` on an independently drawn second level; zero at every non-crossing configuration, so it leaves the pinball minimiser at tau | targeted level was pushed outward (tau=.95 -> .963, tau=.05 -> .037): every "90%" GBC interval was really **92.5%**. After the fix a trained net returns 0.8966 for nominal 0.90 |
| `shrinkage.py` global auxiliary | rate `1 + tau0^2/tau^2` | rate `1/tau0^2 + 1/tau^2`, the conditional for `tau ~ C+(0, tau0)` | at `tau0 = sd(xi_mle) = 0.058`, median tau fell 0.649 -> 0.0215 and kappa rose 0.50-0.62 -> 0.55-0.84. The old sampler barely shrank |
| `evt.py` shape prior | `np.clip(xi, lo, hi)` | inverse-CDF truncated normal | clipping left a **4.8% atom at xi = 0.5** in the amortized nets' training set. Now 0.0; KS vs `scipy.truncnorm` p = 0.80 |
| `run_scale.py` shape prior (found 2026-09-01, after the first three) | `np.clip(rng.normal(0, .3), -.8, .5)` in `_sim_standardised` | `_truncated_normal(rng, 0, .3, -.8, .5)` | the `evt.py` fix never reached here: `run_scale.py` has its own inline simulator and does not call `simulate_gev_training_data`. Same 4.74% atom at 0.5 and 0.39% at -0.8, now 0.0. Affects **tab:scale, tab:sim, tab:heavytail, fig:scale and the timing row**, since `run_sim.py`, `run_heavytail.py`, `run_timing.py` and `run_epochsel.py` all import `train_amortised`/`_sim_standardised` from it |

The MCMC baseline was not affected *by these*: `gev_mcmc` returns `-inf` outside
the shape bounds, which is genuine truncation by rejection. It had a defect of
its own, found in the next round; see 0.2.

The fourth row is worth its own note. Three drivers were audited, fixed and rerun
before anyone checked whether the package fix had reached every *call site*: the
paper's Method text writes the prior as `xi ~ N(0, .3^2)_[-.8, .5]`, truncation
notation, and `gbc.evt` had been corrected to match, but `run_scale.py` samples
its own standardised prior inline and kept the clip. The lesson is that fixing a
library function is not the same as fixing the study: grep for the *defect
pattern* across the whole tree, not for callers of the fixed function.
`run_pnw.py` and `run_attr.py` were unaffected, since they go through
`gbc.evt_inference` and so through the corrected `simulate_gev_training_data`.

What the fourth fix changed, rerun 2026-09-01 on hop063/hop067 (jobs 9511731,
9511732). `run_pnw.py` and `run_attr.py` were not rerun because they cannot be
affected, so tab:rl, tab:spatial and tab:attr stood at that point. All three were
rerun in 0.2, for the Jacobian and for the attribution interval.

| | before (clipped prior) | after (truncated) |
|---|---|---|
| tab:sim short, RMSE diff (MLE-GBC) | -0.07 [-0.14, -0.00] | -0.07 [-0.14, -0.01] |
| tab:sim **light**, RMSE diff | +0.13 [-0.07, +0.35], straddles 0 | **+0.45 [+0.26, +0.66], excludes 0** |
| tab:sim light, GBC RMSE / width | 4.17 / 15.41 | 3.85 / 14.38 |
| tab:sim heavy, RMSE diff | +3.39 [2.44, 4.28] | +3.69 [2.63, 4.70] |
| tab:sim heavy, GBC RMSE / width / bias | 7.54 / 26.90 / -2.00 | 7.24 / 24.04 / -3.21 |
| tab:scale MAD vs MLE | 0.26 | 0.27 |
| timing: train / GBC / MCMC / ratio / break-even | 1663 s / 1.07 ms / 364 ms / 339x / 4581 | 1639 s / 1.06 ms / 370 ms / 351x / 4440 |
| tab:heavytail | 31.9 mm, 44% [36,50], cov 0.91 | 31.9 mm, 44% [35,50], cov 0.91 |

The light-tail row is the one that matters: the 4.7% atom at xi = 0.5 had put
prior mass on shapes far heavier than the light regime, widening GBC's intervals
there and hiding a real advantage. With the atom gone GBC beats the MLE at
xi = 0.1 on *narrower* intervals at identical coverage, and the paper's
simulation narrative changes from "neither wins in the short and light tails" to
"the advantage tracks how hard xi is to estimate", monotone in the tail index.
MCMC diagnostics were bit-identical across *this* rerun (worst Rhat 1.0135, min
bulk ESS 247, median bulk ESS 1101, 3/25 above 1.01), as they had to be: the
baseline never touched the defective code. The 0.2 Jacobian fix then moved them.

The loss fix costs a second forward pass per training step, so training wall-clock
roughly doubles and the amortization break-even moves; inference cost per station
is unchanged, since that is still one forward pass.

A fourth change is an addition, not a correction. The GPD tail head of the
paper's eq:splice was specified but never implemented. It now exists as
`gbc.iqn.IQN(tail=True, tau0=0.9)`: a two-output sub-network gives
`(sigma_u, xi)` with `softplus` and `0.9*tanh` links, spliced continuously at
`u = Q(tau0 | y)`, and `loss_fn` adds a pinball term at a level drawn uniformly
on `(tau0, 1)`. It is on for every network whose target is a fresh annual
maximum (`train_predictive_iqn`, and the `"draw"` net in `run_scale.py`) and off
for networks whose target is the scalar `z_N`, where tau indexes posterior
uncertainty about a parameter rather than the tail of a new observation.

Why it was needed: the body-only Seattle predictive net saturated at 40.38 degC,
below both the detrended record maximum it was conditioned on (43.19) and the
fitted GEV upper endpoint (45.13), so it could not emit a value that had already
been observed. That was a network defect, not a short-tail artifact: the fitted
GEV puts P(>= 42.2) = 8.4e-3. With the head, a fine-grid probe reached 45.21
degC and placed the crossing near P(>= 42.2) = 1.62e-2. The production driver
instead counted equal weights on the truncated grid [0.005, 0.995], giving
0.0115 and 1/87. That count omits the fixed upper-tail mass and is not a full
predictive probability. The GBC row is therefore withdrawn from tab:attr rather
than repaired without a complete rerun of its record bootstrap.

Residual non-monotonicity after the loss fix, measured on a trained body-only
net over a tau grid, is -0.115 degC worst-case (the penalty discourages
crossing, it does not forbid it). Not reported in the WSC paper.

## 0.2 Second review round, 2026-09-01 (four found in review, one after)

An external review of the corrected worktree found four further defects. A fifth
turned up while reconciling the prose against the ledger. All five are the same
class as 0.1: code whose name or docstring matched the paper while its behaviour
did not, each passing the full mechanical audit.

| Defect | Was | Now | Effect |
|--------|-----|-----|--------|
| `evt_inference.gev_mcmc` prior | log-posterior omitted the `-log(sigma)` Jacobian, so a prior stated on `log sigma` was applied as a density in `sigma` | prior extracted to `gev_logprior` with the Jacobian, testable on its own | the effective prior on `log sigma` was `N(log sigma_ref + 0.25, 0.5^2)`, centring sigma **1.284x high**. Every MCMC-dependent number was rerun |
| `evt.poisson_loglik` Poisson mean | GEV survivor `1 - exp(-Lambda(u))` | the exponent measure `Lambda(u) = (1 + xi (u-mu)/sigma)^(-1/xi)` | the survivor understates the Poisson mean by 5.1% at a 0.90 threshold (ratio 0.9491) and more at lower ones. No published number changed: no driver calls this |
| `iqn.sample_iqn` name | called "sample", returns `linspace(0.005, 0.995, B)` evaluated deterministically | same values, docstring now says they are quadrature nodes and must not be bootstrapped | `run_attr.py` had bootstrapped them, so the attribution 90% CI reported Monte Carlo variation the estimate does not have. Replaced by resampling the conditioning record (400 replicates, each a forward pass, which is what amortization buys). The 2023 GBC return-period CI goes from **[71, 114] to [38, 53800] years**, so the interval that made GBC look sharper than the parametric fit was an artifact of resampling a deterministic grid |
| `pyproject.toml` floor | `numpy>=2.0` | `numpy>=1.24` plus a `np.trapezoid`/`np.trapz` shim in `welfare.py` | the published runs used numpy 1.24.4, so the declared environment could not have produced them |
| `experiments/methods.py` (found after the review) | a second, full copy of eight estimators including a `gev_mcmc` whose log-posterior lacked the Jacobian, beside a docstring stating the correct prior | thin re-export from `gbc.evt_inference`, smoke test kept | no published number: nothing imports it. But it shipped in the reproduction bundle as a second, wrong answer, and its own smoke test ran it |

Provenance work in the same round, since a number nobody can trace is a number
nobody can check:

- `tools/numcheck.py` walks every numeric literal in the manuscript prose (float
  bodies excluded, as those are `\input` from generated `tab/*.tex`) and reports
  the ones matching neither `results/numbers.txt` nor a generated table.
  `tools/numcheck.allow` records, one line per literal with a reason, the ones
  that legitimately need no run.
- It found that the stated horseshoe narrowing range `0.42-0.77` contradicted
  the paper's own kappa column: `sqrt((1-kappa)/kappa)` on the published kappas
  gives `0.45-0.90`, so the range was left over from before the 0.1 shrinkage
  fix. Ranges and min/max summaries are the dangerous class, because no table
  shows them and no compile checks them. `run_pnw.py` now ledgers
  `sp:shrink_pct_{min,max}`, `sp:ciratio_{min,max}` and a per-station
  `endpoint`; `run_attr.py` ledgers `attr:gbc_n_exceed` and `attr:gbc_n_draw`.
- `results/sim_log.txt` and `results/attr_log.txt`, hand-redirected stdout, were
  deleted: `sim_log.txt` still showed the pre-0.1 light 4.17 / heavy 7.54 beside
  a `tab/sim.tex` reading 3.85 / 7.24.
- `make_figures.py` wrote to `ROOT/wsc/fig`, which in the bundle would have
  silently created `paper-wsc2026/wsc/fig` while the README promised `fig/`. It
  now falls back to `fig/` where there is no `wsc/` tree, and
  `tools/sync_bundle.py` ships `tab/`, `results/` and `fig/scale_panels.pdf` as
  well as the code, so a reader has something to diff a rerun against.

## 1. Methods (gbc.evt_inference)

| Method | Function | Role |
|--------|----------|------|
| Stationary GEV MLE | `fit_stationary_gev`, `return_level_ci_delta` | baseline |
| Non-stationary GEV MLE | `fit_ns_gev`, `ns_params_at` | baseline (HadCRUT5 covariate) |
| Hill | `hill` | baseline tail index |
| Bayes GEV MCMC | `gev_mcmc` | RW-Metropolis, priors mu~N(ref,5^2), log sig~N(ref,.5^2), xi~N(0,.3^2)[-1,2] (rejection, genuinely truncated) |
| GBC-QNN | `train_functional_iqn` + `gbc_return_level_posterior`; `gbc_crps_coverage_loyo` | ours |
| Horseshoe-pooled GEV | `horseshoe_posterior` + fixed-shape GEV bootstrap | spatial comparator |

### 1.1 GBC-QNN hyperparameters and retrospective sensitivity
IQN: hdim/nh/lr per `gbc.iqn` defaults, i.e. full-batch Adam, lr 1e-3, weight decay 1e-4, cosine
annealing with T_max set to the budget. Predictive IQN trains on `simulate_gev_training_data`
(target = fresh GEV draw); functional IQN targets `z_N(theta)`. n_sim=20000-40000; the amortized
nets behind tab:scale use 2000 full-batch steps.

### 1.2 Training-budget selection (`run_epochsel.py`, rerun 2026-09-01, job 9513422)
The justification that stood here was leaky. It read:

> **Undertraining check:** epochs<=700 gives diffuse predictive (CRPS~16, coverage 1.0);
> epochs>=1500 calibrates (predictive q[.05,.5,.95] match empirical within 0.5 degC).

Both halves of that check are computed against the *observed* station record, so the budget was
tuned, however weakly, on the data the paper then evaluates on. A later sensitivity analysis lives
entirely inside the simulator: 60000 prior-predictive draws split 75/25, six candidate budgets x
3 seeds, each cell trained by the *deployed* `gbc.iqn.train_iqn` (not a lookalike loop) and scored
by mean pinball loss over tau in {.05,.1,.25,.5,.75,.9,.95} on the held-out simulated quarter.
This sweep covers only the return-level target. It neither prospectively selects the deployed
budget, which remains 2000, nor validates the predictive network used for coverage and CRPS.

Values below are the 2026-09-01 rerun under the corrected loss (0.1), which lowers every cell:
the old penalty was active at the wrong quantile, so the loss it reported was not the loss the
network was minimising.

| epochs | val pinball (mean of 3 seeds) | sd | min |
|--------|--------|---------|-----|
| 250    | 0.199207 | 0.003880 | 0.194861 |
| 500    | 0.176757 | 0.001017 | 0.175587 |
| 1000   | 0.170539 | 0.000066 | 0.170478 |
| 2000   | 0.167132 | 0.000184 | 0.166927 |
| 3000   | 0.166047 | 0.000072 | 0.165966 |
| 4000   | **0.165472** | 0.000139 | 0.165337 |

Selected 4000; the deployed 2000 is **+1.00%** worse (was +0.81% under the old loss). Refitting the
amortized net at 4000 and re-scoring all 112 CONUS stations moves z100 by median 0.051 degC,
max 0.416 degC, r=0.9997, so neither tab:scale nor fig:scale turns on the budget. Seed spread
collapses as the budget grows (sd 0.0039 at 250 down to 0.00014 at 4000), so the sweep is ranking
signal, not noise.

The paper now says what happened: 2000 was used in the reported runs after an observed-data check,
the later simulator-only return-level sweep prefers 4000, and the deployed network is retained
because the sensitivity check changes the reported levels little. Ledger rows
`epochsel:{deployed_epochs,selected_epochs,deployed_excess_pct,z100_median_abs_diff,z100_max_abs_diff}`.

**Isolation note.** This job ran in a separate tree (`~/gbc-evt-epochsel`) with its own empty
`results/numbers.txt`, so it could run beside the driver suite without the two `reset_ledger()`
calls clobbering each other. Its four ledger rows were merged into the main ledger afterwards.
An earlier attempt (job 9512714) was cancelled by mistake at 1h36m, after the budget table and
before the z100 step; the table was salvaged from its stdout and the run repeated in full here.

**On the cross-machine check.** The pre-fix sweep had been run twice, on a Hopper Intel node
(job 9504749) and on the local 8-core machine, agreeing to 0.008% across all 18 cells. That check
belonged to the superseded numbers and is not carried forward: this rerun is Hopper-only. What it
established, that the sweep is machine-independent at this precision, is not re-verified here.

## 2. Experiment plan (paper table/figure -> experiment)

| Paper object | Experiment | Script | Compute |
|--------------|-----------|--------|---------|
| Table 1 `tab:rl` | SEA 6-method return levels, LOYO CRPS/coverage | `run_pnw.py` | local CPU |
| Table 2 `tab:spatial` | 5-station xi MLE vs horseshoe, kappa, trends | `run_pnw.py` | local CPU |
| Table 3 `tab:attr` | heat-dome exceedance probabilities, 3 parametric fits | `run_attr.py` | local CPU |
| Table 4 `tab:scale`, Fig 1 `fig:scale` | 112-station amortized z100, LOYO calibration, agreement | `run_scale.py` then `make_figures.py` | local CPU (statistics); **Hopper** Intel node (timings) |
| Table 5 `tab:heavytail` | 103-station precipitation, 25-yr subsamples, clustered bootstrap | `run_heavytail.py` | local CPU |
| Table 6 `tab:sim` | sim study: R=300 reps, GEV(35,3,xi), n=70, three tail regimes | `run_sim.py` (`sim.slurm`) | **Hopper** Intel node, single process (not an array) |
| Wall-clock + MCMC diagnostics | pinned single-thread timings, 4 chains/station, Rhat and ESS | `run_timing.py` (`timing.slurm`) | **Hopper** Intel hop node, exclusive, 1 thread |
| Training-budget sensitivity | validation pinball on held-out simulated draws, 6 budgets x 3 seeds | `run_epochsel.py` (`epochsel.slurm`) | **Hopper** Intel node, 8 worker processes |

Wall-clock is a property of a machine, so the published timings come only from
`run_timing.py` on the recorded benchmark node. `run_scale.py` reads
`results/timing.json` when it exists and prints a warning when it does not, so a
laptop rebuild of the tables cannot silently publish a laptop's timings.

## 3. Real results, Seattle return levels (tab:rl), `run_pnw.py`, rerun 2026-09-01

100-yr return level (degC) under the 1950 vs 2023 climate; CI Width = 90% interval width for
z100 at 2023; Coverage/CRPS = leave-one-year-out predictive (held-out annual maximum).
All values from `results/numbers.txt` (keys `rl:*`) and `tab/rl.tex`.

| Method | RL 1950 | RL 2023 | CI Width | Coverage | CRPS |
|--------|---------|---------|----------|----------|------|
| Stationary GEV MLE | 40.4 | 40.4 | 1.9 | 0.92 | 1.48 |
| NS GEV MLE | 39.2 | 41.4 | 4.4 | 0.88 | 1.41 |
| Bayes GEV MCMC | 40.7 | 40.7 | 2.6 | 0.92 | 1.48 |
| Hill | 39.9 | 39.9 | n/a | n/a | n/a |
| **GBC-QNN (ours)** | 38.5 | 41.9 | 4.9 | 0.88 | 1.47 |
| **Horseshoe-pooled GEV** | 38.3 | 42.2 | 1.9 | n/a | n/a |

Method notes (honest design, propagated to the caption): stationary methods use the same level both
years; NS methods (NS GEV MLE, GBC) shift with the HadCRUT5 covariate (T_1950=-0.227, T_2023=+1.100).
GBC-QNN return level = detrend the maxima to the target climate via the estimated trend, then apply
the stationary GBC functional (`train_functional_iqn`, body only); CRPS/coverage from
`gbc_crps_coverage_loyo`, whose predictive net carries the tail head. Horseshoe-pooled GEV propagates the
spatially-pooled xi posterior (tab:spatial) with a data bootstrap for (mu,sigma), so it never
evaluates the predictive network and gets no Coverage/CRPS entry.
Hill RL via the Weissman high-quantile estimator (k=floor(n/4)); CI/coverage n/a.

Changes from the pre-fix (2026-06-17) run: GBC-QNN 38.3 -> 38.5, 42.0 -> 41.9, CI 4.8 -> 4.9,
coverage 0.86 -> 0.88, CRPS 1.49 -> 1.47; the horseshoe-pooled GEV CI moved 2.1 -> 1.9. The coverage column moved
because the old "90%" intervals were really 92.5% (see 0.1). The horseshoe interval now coincides
with the stationary MLE delta interval (1.9) rather than sitting near the Bayes MCMC interval
(2.6); the earlier note claiming the latter comparison is withdrawn.

The 0.2 Jacobian fix moved the Bayes GEV MCMC row only in the third digit: z100 40.716 both years
(unchanged at one decimal) and CI width 2.5 -> 2.6. That is the expected size. With n=78 maxima the
likelihood dominates a prior on log sigma this wide, which is exactly why the defect survived
inspection through the chain and needed a test on the prior density itself
(`TestMCMCPriorJacobian`), not on the posterior.

## 3b. Real results, spatial horseshoe (tab:spatial), rerun 2026-09-01

| Station | xi (MLE) | xi (HS) | 90% CI | kappa | Trend (degC/dec) |
|---------|----------|---------|--------|-------|------|
| SEA | -0.23 | -0.23 | [-0.26,-0.20] | 0.84 | +0.50 |
| PDX | -0.24 | -0.23 | [-0.26,-0.21] | 0.78 | +0.46 |
| GEG | -0.21 | -0.22 | [-0.25,-0.18] | 0.80 | +0.05 |
| EUG | -0.31 | -0.25 | [-0.33,-0.22] | 0.55 | +0.42 |
| PDT | -0.13 | -0.20 | [-0.25,-0.10] | 0.70 | +0.13 |

Two summaries the paper quotes as ranges, now ledgered by the driver rather than computed by hand
(`sp:shrink_pct_{min,max}`, `sp:ciratio_{min,max}`): deviations from the group mean (fixed at
`mean(xi_mle)`, which is what `horseshoe_posterior` uses) shrink by **64.0 to 79.8%**, and the 90%
horseshoe credible width is **0.457 to 0.819** of the single-station normal width `2 * 1.645 * SE`.
The paper had said 0.42-0.77, left over from before the 0.1 shrinkage fix; see 0.2.

**Findings:**
1. **All five stations are short-tailed** (xi in [-0.31,-0.13], every 90% CI strictly negative,
   marginally and without a simultaneous claim). Fitted upper endpoints run 44.3 (SEA) to 54.7
   (PDT) degC.
2. **The horseshoe pools heavily**, which reverses the pre-fix reading. With the corrected global
   auxiliary, deviations from the group mean shrink 64-80% and the shape intervals narrow to
   0.46-0.82 of the single-station width (single-station 90% widths: SEA 0.143, PDX 0.107,
   GEG 0.149, EUG 0.143, PDT 0.228). The **withdrawn** claim was "the horseshoe shrinks little
   (kappa ~0.5-0.6) because the stations genuinely agree, which is itself the finding": that
   was an artifact of the `shrinkage.py` rate bug, not a property of the data.
3. **EUG and PDT, the two sites farthest from the group mean, keep the lowest kappa** (0.55, 0.70)
   and the widest posteriors: the horseshoe's heavy-tailed local scale letting an outlier retain
   its own uncertainty. This is the behaviour the prior is chosen for and it was invisible before
   the fix.
4. **The 2021 heat dome (SEA 42.2 degC)** lies beyond 1/10^4 under the fitted 1950
   NS-GEV climate and near 1/338 under the fitted 2023 climate; see the parametric
   `attr:` rows of `results/numbers.txt`. Both intervals are unbounded above. The
   GBC row is withdrawn because its bounded deterministic grid did not estimate
   a full predictive probability.

### Status / next
- [x] Data + methods prepared for gbc v0.6.0; the release is not yet tagged.
- [x] Three estimator defects fixed, GPD tail head implemented, 14 regression tests (0.1).
- [x] Tables regenerated on the local or Hopper environments recorded in the experiment map;
      timings, simulation and budget sensitivity use Hopper.
- [x] Manuscript prose rewritten from the rerun ledger; audit and WSC checks pass at 12 pages.
- [x] `run_epochsel.py` rerun under the corrected loss (job 9513422) -> 1.2 refreshed.
- [ ] Remaining comparator gap: no non-neural baselines beyond MLE/MCMC/Hill (no L-moments
      or PWM, no profile likelihood, no Bayesian GEV under the amortized net's own shape prior).

## Drafting sessions

- 2026-09-02: revised the abstract, introduction, GBC-QNN method, experiments and
  discussion under the AI-drafting pre-prompt. The manuscript AI-writing scan
  moved from 2 hits before the pass to 0 after cleanup. No citations were added;
  the existing neural-EVT references were repositioned and characterized more
  precisely. Manual rewrites withdrew the GBC heat-dome row, separated
  predictive from functional return levels, and made the timing and budget
  limitations explicit.
