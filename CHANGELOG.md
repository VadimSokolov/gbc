# Changelog

Numbers produced by `train_iqn` are only comparable within a version whose
training path is unchanged. Each entry says whether fresh training reproduces
the previous version.

## 0.7.0 (2026-09-04)

Training output **does not** reproduce 0.6.0. The monotonicity penalty is
sampled differently and is now evaluated in the causal models, so a fresh fit
under the same seed gives a different network. Results in
`paper-wsc2026/results/` and `tab/` were produced under the 0.6.0 training
path and have not been regenerated.

### Metrics

- `crps_samples` is exact and deterministic by default, using the
  sorted-sample identity for the pairwise term in O(B log B). The previous
  default drew one unseeded random permutation, so identical inputs returned
  different values on every call: on the motorcycle data at B=500, thirty
  calls had standard deviation 0.21 and range 1.03, larger than the 0.07 gap
  between GBC and hetGP in the book's Chapter 5 table. Any CRPS computed by an
  earlier version carries that noise; recompute before comparing methods.
- `method="mc"` keeps the single-permutation estimator for very large B. It
  requires an explicit integer `seed`, uses an isolated generator, and leaves
  NumPy's global random state untouched.

### Prediction

- `rearrange_quantiles(quantiles, taus, target_taus=None)` performs monotone
  rearrangement (Chernozhukov, Fernandez-Val and Galichon, 2010): sort along
  the quantile grid, then interpolate to requested levels.
- `predict_iqn(..., rearrange=True, rearrange_grid_size=1000)` evaluates a
  dense grid, rearranges, and interpolates back to the requested levels.
  `sample_iqn(..., rearrange=True)` sorts each column. Both default to
  `False`; existing prediction paths are unchanged. Rearrangement preserves
  each column's multiset, so sample-based CRPS is invariant to it.

### Training

- `loss._sample_quantile_pair` draws the pair of levels used by the crossing
  penalty: the first level is Uniform(0,1) as before; the second is, with
  probability one half, within 0.05 of the first, and otherwise an
  independent uniform draw. Measured on the motorcycle data (seed 0, 3000
  epochs) against 0.6.0: predictions shift by up to 15.2 on a response
  standard deviation of 48.1; crossing pairs 34.2% to 33.5%; exact CRPS
  10.433 to 10.530; 90% coverage 0.872 to 0.887. The change did not reduce
  crossings on the one dataset measured; it is shipped on the maintainer's
  decision, documented here so the effect is not mistaken for a modelling
  result.
- `composite_loss` raises `ValueError` when the crossing weight is nonzero
  and the second-level pair is not supplied, instead of silently dropping the
  term.
- `CausalIQN` and `CausalIQNv2` now evaluate the crossing penalty. In every
  earlier version both called `composite_loss` without the second level, so
  the configured weight was requested and never applied. The two levels are
  evaluated under the same dropout realisation so the penalty compares
  quantile levels rather than dropout noise. Chapter 13 results under 0.7.0
  therefore differ from earlier versions.

### Tests

- `conftest.py` defines a `slow` marker and `--runslow`. The two
  `TestGPDTailHead` fits are marked slow; they are mini-batched and now assert
  that the body fits below the splice before making any claim about the tail,
  so an underpowered network cannot pass them vacuously.
- 307 tests, 305 in a default run.

## 0.6.0

Not tagged. The point at which the two development lines of the package were
merged: the EVT line (`evt`, `evt_inference`, `survivor`, `shrinkage`,
`diagnostics`, the GPD tail head on `IQN`, the two-level crossing penalty, the
WSC 2026 reproduction bundle) and the book line (`ssm`, `spatial_gnn`,
`portfolio`, `abc`, `testing`, mini-batch and device support in `train_iqn`).
`train_iqn` carries the keyword arguments of both. 25 modules.

## 0.5.1 and earlier

See the git history. 0.5.x tagged the EVT modules and the reproduction
bundle; 0.3.0 introduced `survivor`, `evt` and `shrinkage`; 0.2.0 was the
book companion release with the multivariate, causal and welfare modules.
