# Quantile Ordering and Deterministic CRPS

## Goal

Make the existing soft quantile-crossing penalty effective in every IQN training path, add a deterministic exact empirical CRPS calculation with an explicitly seeded approximation for large sample sets, and provide monotone rearrangement as an opt-in prediction operation without changing default IQN predictions or samples.

The crossing penalty remains a training regularizer, not a monotonicity guarantee. Rearrangement is the operation that guarantees ordered values on its evaluation grid.

## Affected files

- `/Users/vsokolov/Dropbox/papers/gbc/gbc/loss.py`: validate paired crossing-loss inputs and provide local plus global quantile-pair sampling.
- `/Users/vsokolov/Dropbox/papers/gbc/gbc/iqn.py`: use the shared pair sampler and add rearrangement to prediction and sampling APIs.
- `/Users/vsokolov/Dropbox/papers/gbc/gbc/causal.py`: activate the paired crossing penalty in both causal IQNs and compare outputs under matched dropout masks.
- `/Users/vsokolov/Dropbox/papers/gbc/gbc/metrics.py`: replace the randomized default CRPS estimator with the exact empirical score and expose seeded Monte Carlo as an opt-in method.
- `/Users/vsokolov/Dropbox/papers/gbc/gbc/__init__.py`: export the public rearrangement helper.
- `/Users/vsokolov/Dropbox/papers/gbc/tests/test_core.py`: cover crossing-loss validation and local pair sampling.
- `/Users/vsokolov/Dropbox/papers/gbc/tests/test_causal.py`: prove that the crossing weight affects both causal losses and that paired dropout masks match.
- `/Users/vsokolov/Dropbox/papers/gbc/tests/test_metrics.py`: compare exact CRPS with a brute-force reference and test seeded approximation behavior.
- `/Users/vsokolov/Dropbox/papers/gbc/tests/test_rearrangement.py`: test the transform, prediction flags, dense-grid monotonicity, and sample multiset preservation.
- `/Users/vsokolov/Dropbox/papers/gbc/README.md`: document deterministic CRPS and opt-in rearrangement.
- `/Users/vsokolov/Dropbox/papers/GBC.md`: record the corrected loss, completed decisions, interfaces, and verified repository state.

## Interface contracts

### Paired crossing loss

`composite_loss` accepts either both `f_other` and `tau_other`, or neither. Supplying only one is an error. If the crossing weight is nonzero, omitting the pair is an error. A zero crossing weight permits an unpaired call.

The internal quantile-pair sampler returns two distinct levels in the unit interval. Half of its pairs are local by default, within a configurable radius, and the rest span the interval independently. The first level retains the uniform marginal distribution used by the pinball loss.

`IQN`, `CausalIQN`, and `CausalIQNv2` all use this sampler. The causal models pass the second prediction into `composite_loss`. For `CausalIQNv2`, paired predictions use the same dropout realization so the penalty compares quantile levels rather than dropout noise. Random-number state advances as one stochastic forward pass.

### CRPS

`crps_samples(y, samples, *, method="exact", seed=None)` retains the existing two-positional-argument call. The default computes empirical CRPS exactly using the sorted-sample identity, with time complexity dominated by sorting and no Monte Carlo randomness.

`method="mc"` uses one random permutation and requires an explicit integer seed. It uses an isolated NumPy generator and does not mutate global random state. Unsupported methods, empty sample sets, incompatible shapes, and unseeded Monte Carlo calls raise `ValueError`.

### Rearrangement

`rearrange_quantiles(quantiles, taus, target_taus=None)` accepts a one-dimensional curve or a matrix whose first axis is the quantile grid. Grid levels must be finite and strictly increasing. Values must be finite and agree with the grid length. The function sorts along the grid axis, then optionally interpolates each curve to finite target levels inside the supplied grid. A one-dimensional input produces a one-dimensional result, and a matrix preserves its non-grid axis.

`predict_iqn` gains keyword-only `rearrange=False` and `rearrange_grid_size=1000`. The false path is unchanged. The true path evaluates an adaptive dense grid that contains the requested interior probability range, rearranges each observation's curve, and interpolates back to the requested levels in their original order.

`sample_iqn` gains keyword-only `rearrange=False`. The true path sorts each observation's fixed quantile-grid samples. Sorting preserves each sample multiset and therefore preserves exact empirical CRPS.

## Edge cases

- A crossing pair in reverse level order must be normalized before applying the hinge.
- Local pair sampling near zero or one must remain inside the unit interval.
- The causal dropout pairing must restore global CPU and relevant CUDA random states after the second evaluation.
- Exact CRPS with one predictive sample reduces to mean absolute error.
- CRPS accepts only a one-dimensional observation vector and a two-dimensional sample matrix with matching observation count.
- Rearrangement rejects empty grids, non-increasing grids, non-finite values, targets outside the grid, and invalid dense-grid sizes.
- Repeated or unsorted requested prediction levels retain their requested output order after interpolation.
- Rearrangement remains off by default so saved workflows and published prediction paths do not change.
- Tail-enabled IQNs are rearranged numerically on the selected finite grid, without changing the analytic GPD tail head.

## Acceptance criteria

- Tests fail before implementation for each newly specified behavior.
- Both causal loss functions respond to a nonzero crossing weight.
- The loss API can no longer silently discard a requested crossing penalty.
- Local pair tests exercise nearby levels, and dense-grid tests demonstrate monotone rearranged output.
- Exact CRPS matches a brute-force all-pairs calculation to numerical precision and is deterministic across repeated calls.
- Seeded Monte Carlo CRPS repeats for the same seed and rejects a missing seed.
- Rearranged samples contain exactly the same per-observation values as raw samples and have identical exact CRPS.
- Existing calls without rearrangement preserve their shapes and numeric behavior.
- The default test suite passes, followed by the slow suite if its runtime remains practical.
- README and `GBC.md` describe the shipped behavior without claiming that the soft penalty guarantees monotonicity.
