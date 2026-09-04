# Restore Independent IQN Quantile Pairs

## Goal

Release version 0.7.1 with the plain `IQN` training random stream restored to its 0.6.0 behavior while retaining exact CRPS, rearrangement, strict crossing-loss validation, and the activated causal crossing penalty.

## Files to change

- `/Users/vsokolov/Dropbox/papers/gbc/gbc/iqn.py`: restore two direct independent uniform draws in `IQN.loss_fn` and remove its dependency on the local-pair sampler.
- `/Users/vsokolov/Dropbox/papers/gbc/tests/test_regressions.py`: assert the exact random-draw contract for body-only and tail-enabled IQN losses.
- `/Users/vsokolov/Dropbox/papers/gbc/docs/specs/quantile-ordering-and-crps.md`: mark its local-pair requirement as superseded for plain IQN while retaining it for causal models.
- `/Users/vsokolov/Dropbox/papers/gbc/CHANGELOG.md`: document the 0.7.1 training-path restoration, correct the table path, and attribute the slow-test work to 0.6.0.
- `/Users/vsokolov/Dropbox/papers/gbc/pyproject.toml`: set the distribution version to 0.7.1.
- `/Users/vsokolov/Dropbox/papers/gbc/gbc/__init__.py`: set the runtime version to 0.7.1.
- `/Users/vsokolov/Dropbox/papers/gbc/CITATION.cff`: set the citation version to 0.7.1.
- `/Users/vsokolov/Dropbox/papers/gbc/README.md`: describe independent plain-IQN pair sampling and update the verified test count if the regression test changes it.

`/Users/vsokolov/Dropbox/papers/GBC.md` is intentionally excluded. Its coordination row requires Codex to leave it unchanged and provide results for later reconciliation.

## Interface contracts

`IQN.loss_fn(x, y, w=(0.3, 0.3, 0.4), w_tail=0.4)` keeps its signature and loss terms. It draws `tau` and `tau_other` with exactly two consecutive calls to `torch.rand(1).item()`. Both levels are independent Uniform(0,1) draws. It evaluates both predictions and passes the pair to `composite_loss`, including when the crossing weight is zero, to preserve the 0.6.0 random stream and optimization path.

For a tail-enabled IQN, the dedicated tail level remains the next random draw after the two crossing-loss levels. No additional random draw may occur inside `IQN.loss_fn` before that tail draw.

`loss._sample_quantile_pair` remains available and unchanged for `CausalIQN` and `CausalIQNv2`. The causal models continue to evaluate their crossing penalty under matched dropout masks. `composite_loss` continues to reject a nonzero crossing weight without both paired inputs.

Given the same code dependencies, device, thread configuration, data, parameters, and seed, plain `train_iqn` output in 0.7.1 must reproduce the 0.6.0 path. Version 0.7.1 does not reproduce the short-lived 0.7.0 plain-IQN path. Causal training continues the corrected 0.7.0 path and does not reproduce 0.6.0.

The package version is 0.7.1 in `pyproject.toml`, `gbc.__version__`, and `CITATION.cff`. The annotated tag `v0.7.1` must dereference to the release commit.

## Edge cases

- A zero crossing weight still consumes the second IQN level and evaluates its prediction because removing it would change the seeded training path.
- Tail-enabled training consumes its dedicated tail level as the third `torch.rand(1)` draw.
- Mini-batch permutations remain in their existing positions in the random stream.
- The local-pair helper remains in use by causal models only and must not be deleted.
- Exact equality is promised only within the same numerical environment; the existing cross-machine PyTorch reproducibility caveat remains valid.
- Exact CRPS and both rearrangement flags remain unchanged.
- Release notes must describe general package behavior and must not use a paper-specific workflow as the rationale for this repair.
- User-facing documentation must not claim that plain IQN deliberately mixes nearby and distant pairs after the restoration.

## Acceptance criteria

- A regression test fails on 0.7.0 because `IQN.loss_fn` consumes more than two random draws before body-only loss evaluation.
- The regression test passes after restoration for body-only and tail-enabled IQNs and verifies the exact draw order.
- The existing tests proving the `composite_loss` guard and active causal penalty continue to pass.
- A seed-0, 3000-epoch motorcycle fit produces a sample matrix exactly equal to the saved 0.6.0-path baseline, with maximum absolute difference zero.
- The default and slow test suites pass.
- Python compilation and staged-diff checks pass.
- A freshly built wheel reports version 0.7.1, and all three source version declarations agree.
- `CHANGELOG.md` states that 0.7.1 restores 0.6.0 plain-IQN training, retains the 0.7.0 causal fix, and does not reproduce 0.7.0 plain-IQN training.
- `CHANGELOG.md` identifies the table path as `paper-wsc2026/tab/` and places the slow-test work under 0.6.0.
- README describes the plain IQN penalty as comparing independently sampled quantile levels.
- GitHub `main`, both local clones, and the annotated `v0.7.1` tag resolve to the same release commit.
