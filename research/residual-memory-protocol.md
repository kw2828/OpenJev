# Residual uncertainty in compact memory

Prospective fixed-checkpoint diagnostic, 24 September 2026. The previous
query-centered learned candidate remains failed. This study qualifies an
established control for future architecture work; it cannot establish novelty.

## Mechanism and controls

The parent Nyström model retains full covariance within its rank-16 feature
space, but discards the residual kernel outside that space. Far from its fixed
inducing grid, features approach zero and predictive variance approaches the
observation-noise floor. This is an analytic failure mechanism, not yet a
causal explanation of all parent errors.

Let phi(x)=K(x,Z) chol(K(Z,Z)+1e-6 I)^(-T), with the same 4x4 grid Z on
[-2,2]^2 as the parent. Define d(x)=amplitude^2-||phi(x)||^2. Use every final
static Nyström checkpoint from the parent, fit seeds 11,23,37. No parameter
selection, re-estimation, training or calibration occurs. The inherited
parameters were trained under the original low-rank model, not the correction.

For each checkpoint compare four fixed conditions:

1. **SoR:** original low-rank prior and observation variance 0.0225.
2. **Query only:** identical posterior means and identity weights to SoR;
   add d(query) to its final component variance. This is a deliberately partial
   mechanism probe, not a coherent replacement inference model.
3. **FIC:** independent residual approximation. Add d(x) to noise at archive
   observations, joint request likelihood and query prediction. Weighted
   information updates are precision += phi phi^T/(noise+d) and
   information += phi*y/(noise+d), starting at I and zero. Retain full
   covariance within this feature space and condition on request examples once.
4. **Full fitted GP:** full kernel conditioning at the exact same inherited
   length and amplitude. This isolates covariance approximation from kernel
   hyperparameter differences.

Also run the true-generating-kernel GP once per dataset and always-defer scoring.
FIC is established sparse GP methodology, not a new architecture. Relevant
sources include [the unifying sparse-GP framework](https://www.jmlr.org/papers/v6/quinonero-candela05a.html)
and [variational inducing-variable learning](https://proceedings.mlr.press/v5/titsias09a.html).
VFE is a different training objective and is not implemented here. These
approximations do not inherit the full generating model's calibration guarantee.

All residuals below -1e-12 fail. Negative residuals within floating-point
tolerance are explicitly set to zero and counted. No other variance repair,
retry or solver fallback is allowed. Noise makes the full GP systems positive
definite. This FIC approximation has independent residuals per observation;
it does not retain a shared residual for duplicate input coordinates. All
scientific coordinates are continuous draws and exact duplicates are absent
almost surely. Replaying a request rebuilds its conditional distribution;
it never adds an observation to the frozen archive.

## Fresh data and public boundary

Use the unchanged parent GP generator with namespace 442260924. The generating
RBF kernel has length and amplitude 1, with independent noise variance 0.0225.
Each context has four requests, each revealing four examples from a uniformly
selected unknown archive block and one query coordinate. The target is noisy.
Sample each block jointly over all its archive and request coordinates.
The predictor sees only the archive, its own request examples and query x.
Hidden block IDs, targets and other request outcomes never enter prediction.

Three fresh cohorts of 128 contexts per population, seed namespace+offset+cohort:

| Population | Offset | Blocks | Observations/block | Coordinate extent |
| --- | ---: | ---: | ---: | --- |
| BASE | 100 | 4 | 16 | [-2,2]^2 |
| SHIFT | 200 | 8 | 16 | [-3,3]^2 |
| LONG | 300 | 4 | 128 | [-2,2]^2 |

SHIFT changes geometry and block count jointly. LONG tests a longer archive
with identical coordinate support. There are 1,152 independent contexts and
4,608 fresh requests. Requests within a context are correlated; the three
inherited fits share parent training data. Average scores, not predictions,
equally across requests, contexts, cohorts and paired fits. No model ensemble.

## Outcomes and fixed qualification rule

Use the parent's negative/defer/positive costs: wrong sign costs 1, correct
sign costs 0, defer costs 0.2; ties choose the first action. Conditional regret
is excess expected cost under the true public-information GP mixture, integrating
the unknown identity. Full fitted GP may have nonzero regret; true GP's zero
regret is definitional. Also record noisy-target NLL, Brier, MSE, action rates,
and always-defer regret. Retain all 117 prediction/metric groups.

Qualify FIC as an improved conventional control only if all eleven conditions
hold, using means over all three fit scores and cohorts:

- On BASE, SHIFT and LONG: NLL no more than 0.02 nat/request worse than SoR.
- On BASE and LONG: regret <= 1.05 times SoR regret plus 1e-6.
- On SHIFT: regret at least 25% lower than SoR; NLL at least 0.1 nat lower;
  regret strictly lower than always defer.
- On each SHIFT cohort separately: mean regret strictly lower than SoR.

Also report, without promotion thresholds, posterior identity-weight L1 error
and sign-probability error versus full GP at the same fitted kernel for SoR,
query-only and FIC. These diagnostics motivate or reject later correlation
retention work; they do not establish that a learned selector can close a gap.
No retuning, replacement seed, extra condition or reinterpretation of failure.

## Resources, stopping and evidence

One-thread CPU float64; no learned model training. Total run cap 1,200 seconds.
Any numerical failure stops this run with partial artifacts and an error receipt.
Save all data, predictions, extracted parameters, three unchanged checkpoints,
residual-roundoff counters, source snapshots and original process closures.
Authenticate the entire parent manifest before and after the run.

All four implementations cache their archive, including full GP. Low-rank
inference retains only fixed-size feature and posterior arrays; full GP retains
coordinates, its archive factor and solved coefficients. Record actual retained
array bytes separately from transient input archive bytes. This is not peak RSS.
On the first context of each population, time three warmups and 20 repeats of
archive construction plus four sequential requests. Keep all per-repeat values.
Equal inputs do not imply equal numerical kernels or compute budgets.

Independent saved-array auditing must reconstruct dense SoR, FIC and full-GP
covariances, score all groups and reproduce all eleven conditions without
training, changing parameters or generating another dataset. Opaque checkpoint
hashes and extracted scalars establish lineage; they are not optimizer replay.
