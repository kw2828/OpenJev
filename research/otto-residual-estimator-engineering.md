# Bayesian residual comparison: implementation status

**The numerical integration passes 278 fabricated tests and lint. No new empirical
result is available.** The [proposed experiment](otto-residual-estimator-design.md)
still requires a qualified collection/evaluation runner and a source-bound
registration before execution. The previous study's **DEV FAIL 6/13** is unchanged;
its unused TEST split remains closed.

The experiment asks whether accumulated directional precision improves memory
corrections when the recurrent representation and teacher access stay fixed.
Full-covariance recursive least squares is the sole candidate. Ordinary recurrent
prediction, continued recurrent training, last-error correction and normalized
delta memory remain the usefulness controls. A diagonal approximation and three
weaker corrections test separate explanations for any gain.

## Implemented components

| Component | Responsibility |
| --- | --- |
| [Cache contract](../src/openjev/research/otto_residual_contract.py) | Validate complete episodes, observed-answer masks, chronological features, exact query outputs and the 72 development / 27 confirmation view rosters. |
| [Frozen feature extraction](../src/openjev/research/otto_residual_features.py) | Run both fixed recurrent predictors, construct the shared trace cue, reset each episode and account for both sets of forward calls. |
| [Estimator replay](../src/openjev/research/otto_residual_replay.py) | Apply all nine methods with matched inputs, read before updating memory, and round corrected action scores once to float32. |
| [Selection and decision rule](../src/openjev/research/otto_residual_gate.py) | Select the candidate's prior ratio on development only; enforce all 13 usefulness conditions and report covariance/attenuation contrasts separately. |

The replay layer makes no neural-model calls. Feature extraction does, and those
calls remain part of the experiment's cost. The cache is an offline way to share
identical features across comparisons, not a measured deployment speedup.

## What the checks establish

The test suite includes 117 checks for the existing Bayesian reference and 161
checks for the new integration. Independent test code compares the online full
posterior with a batch information-form solution for all four registered prior
ratios. Other checks cover:

- Exact agreement with the original uncached cue computation at the same batch
  geometry, including chunk boundaries and a complete 2,188-step episode.
- Fresh state between episodes, first-query exclusion and rejection of overlong
  episodes before either recurrent model runs.
- No influence from masked teacher labels or later observations on earlier
  predictions; each query's forecast precedes its answer-dependent update.
- Identical full-posterior evolution for all three weaker-read controls.
- A numerical example that distinguishes one final float32 rounding from an
  earlier rounding of the correction.
- Strong-baseline, paired-seed, support and full-scope guards; an attractive
  pooled average cannot override a failed condition.

The [first attempt](../output/otto-residual-estimator-engineering-v1/attempt-01/receipt.json)
passed all 278 tests and failed seven mechanical lint checks. After those fixes,
the [second attempt](../output/otto-residual-estimator-engineering-v1/attempt-02/receipt.json)
passed the same 278 tests and scoped lint. Both retained their logs, checked all
145 prior-study source pins unchanged, and closed their child process groups.
No empirical arrays or checkpoints were accessed, based on source review of the
tests; this was not enforced by an operating-system sandbox. The feature tests
ran fabricated inputs through synthetic-weight recurrent models.

## Fresh cohort reservation

The [scoped reservation and reviewed matches](../output/otto-residual-estimator-v1/seed-review-01.json)
reserve 18 originating environment seeds, each with three collector paths:

| Split | lambda3 | lambda4 | Paths |
| --- | --- | --- | ---: |
| Development | 314000001-314000003 | 315000001-315000003 | 18 |
| Confirmation | 316000001-316000006 | 317000001-317000006 | 36 |

The three reused fit seeds, 309000001-309000003, identify existing TRAIN-derived
weights and are not presented as fresh environment seeds. No fit is selected
from the previous development result.

The first reservation is preserved: an unrelated fit seed occupied its proposed
310-million block. The replacement scan covered 3,464 source/metadata files and
found no exact seed collision. Five conservative text matches were reviewed:
three were operation/sample counts and two were decimal fragments, not seeds.
Both scans and their full file hashes remain available. This is a scoped
reservation, not proof that the numbers were never used anywhere else.

## Next execution boundary

The pure numerical components do not authenticate checkpoint provenance, native
runtime inputs or empirical phase admission. The next runner must bind those
identities, enforce resource caps, preserve complete predictions, and support an
independent audit of the saved outputs. Confirmation feature extraction and
evaluation require the completed development decision and original successful
producer/auditor closures.

Freeze all 54 path identities and collector rotations together. Collect only
the 18 development paths first. Collect the 36 confirmation paths only after a
development pass, because the collector itself decodes its saved arrays to
verify the serialization roundtrip. This keeps confirmation data generation as
well as evaluation behind the same boundary. Keep confirmation's master episode
indices 18-53; do not restart its identity numbering. If existing metrics need
the internal stage label `test`, map it explicitly from this new study's
`confirm` phase without resolving any old-study TEST path or roster.

No Bayesian advantage, calibrated probability, autonomous control improvement,
biological-wiring benefit or architectural novelty follows from these tests.
