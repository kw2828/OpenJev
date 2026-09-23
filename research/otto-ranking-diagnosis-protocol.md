# Saved forecast errors and action rankings

This retrospective diagnostic follows the complete [separate-readout failure](otto-separate-prior-results.md). Aggregate results have already been exposed. The analysis is specified before its new per-row array reads, but it is not an untouched evaluation or a new efficacy test. The original gates remain FAIL 11/19, 10/19 and 14/29.

## Question and fixed membership

Does ordinary nonquery score error fall on the same rows where legal-action ranking improves, or do changes mainly concern other pairs of actions? All-four prior MSE at genuine queries describes a different output and population. It must not be used as evidence of a nonquery ranking/MSE mismatch.

Use all 24 final models, all three fit seeds, both settings, all 36 VALID trajectories and the held-score reference once. Authenticate the original training plan, collection identities, training receipt and supervising parent, independent audit and its parent, and every saved prediction/window payload. All 140 original scientific sources remain unchanged. There are no model, optimizer, teacher, simulator or training calls, and no new predictions.

Individual scope summaries cover full nonquery paths, initial steps 1-3, primary nonquery steps at or after 5, and primary ages 1, 2 and 3 separately. Within each scope, average supported rows inside each episode, then divide by the number of declared episodes. Unsupported episodes retain a zero contribution. Overall, setting, collector and originating-case groups retain their declared membership. The three collectors for a case are paired paths, not independent cases. Preserve every seed; three-seed means are descriptive.

## Exact row quantities

Use the original strict float32 `1e-10` near-minimum sets and first legal predicted near-minimum action. Agreement accepts any teacher near-minimum. Raw gap subtracts the actual minimum teacher float32 score in float64. A correct near-minimum action can have a small positive gap.

For legal set L of size m, compute ordinary legal-centered MSE using separately centered float64 conversions of the original float32 scores. Independently decompose it as:

`MSE = sum_{a<b in L} [((p_b-q_b)-(p_a-q_a))^2] / m^2`.

Keep three additive terms: pairs entirely within the teacher near-best set, pairs crossing its boundary, and pairs entirely outside it. Record arithmetic residuals. The boundary term is a diagnostic decomposition, not a new loss or a causal proof that the other terms are useless.

Retain the teacher margin to the nearest legal action outside its near-best set, with explicit support and null conditional means when no such action exists. Classify the predicted near-best set as wholly within, wholly outside, or mixed relative to the teacher set; distinguish mixed selections that are correct or wrong. These classes are exhaustive.

Keep strict minima separate from tolerance-based choices. For first exact teacher minimum a and first exact predicted minimum b, verify:

`(q_b-q_a) + [(p_b-q_b)-(p_a-q_a)] = p_b-p_a`.

This witness diagnoses score differences and is not substituted for the deployed tie rule. No empirical quantile bins, post hoc cutoffs, best-seed selection or threshold sweep is allowed.

## Fixed paired contrasts and denominators

Within each architecture, compare shared-MSE to shared-AUX, separate-MSE to separate-AUX, shared-MSE to separate-MSE, and shared-AUX to separate-AUX. Also compare explicit correction to ordinary GRU at each of the four readout/objective combinations. These are twelve contrast types at three paired seeds, totaling 36 contrasts.

For every episode and scope, retain all CC, CW, WC and WW correctness transitions, including zero cells. Report raw counts and additive contributions to agreement, gap, MSE and its three pair components using the full scope denominator, not the transition count. Agreement change must equal WC mass minus CW mass; all four cells reconstruct gap/MSE changes. Separately retain initial and later-phase contributions under the full nonquery denominator so they sum exactly to the full-path change. Do not subtract differently weighted full and primary means to infer early effects.

The output retains per-episode records, all setting/case/collector groups, all fit seeds and equal-seed means. Reconcile original agreement, raw gap and centered-MSE summaries on the same scopes. A separately implemented checker reconstructs numerical quantities from the saved arrays without importing the diagnostic reducer. Byte/provenance and process-lifecycle helpers can be shared; numerical reducers cannot.

## Execution and interpretation

Qualify both implementations with fabricated ties, changing legal sets, unequal episode lengths, zero-support scopes, transition accounting and error decompositions. Freeze protocol, source pins, qualification and original input identities before the diagnostic array reads. Diagnostic and checker each run once under their own original **240-second supervisor**, one numerical thread, **2 GiB RSS** and **128 MiB output** bounds. Outputs use exclusive directories. Preserve failures; no resume, scientific retry or cap extension. Plotting and publication may read only closed results.

The analysis has no new scientific pass/fail gate. If lower nonquery MSE accompanies worse ranking and the decomposition shows gains concentrated away from relevant boundaries, that motivates a separately frozen ranking-sensitive objective. A different result requires a different interpretation. Age-dependent errors may motivate a memory-propagation question but cannot establish its cause.

[Policy Distillation](https://arxiv.org/abs/1511.06295) and [Smart Predict, then Optimize](https://arxiv.org/abs/1710.08005) provide established decision-focused controls. Query-written delta memory has prior art in [fast-weight programmers](https://proceedings.mlr.press/v139/schlag21a.html) and [DeltaNet](https://arxiv.org/abs/2406.06484). Neither an objective change nor that memory rule alone establishes novelty. Any later architecture claim still requires fresh fitting, a strong ordinary recurrent control, matched exposure, paid computation, an unseen scenario shift and autonomous utility evidence.
