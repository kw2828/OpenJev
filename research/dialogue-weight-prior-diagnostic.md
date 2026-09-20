# Does the prescribed training loss tilt the keep/update decision?

Prospective saved-output diagnostic, 20 September 2026. This note specifies
the next comparison; no corrected predictions have been computed. The
[commitment diagnostic](dialogue-commitment-results.md) remains mixed, and the
original token-alignment campaign remains **FAIL, 18/22**. This proposal does
not reopen either result or admit new training.

## Why this control comes first

The [training objective](../scripts/study_dialogue_typed.py) gives each of three
transition strata one third of total loss mass. The authenticated
[training plan](../output/dialogue-token-alignment-scientific-v1/training-01/plan.json),
SHA256 `e700080ee2dd28c83c0c13a0dad2640cdc83fb1992efb4bfd3777a90111877c1`,
records 29,211 FIT rows:

| Training stratum | Rows | Per-row weight, N / (3 × count) |
|---|---:|---:|
| Unmentioned retention | 17,666 | 0.5511717423 |
| Assigned retention | 9,246 | 1.0531040450 |
| Changed state | 2,299 | 4.2353197042 |

This deliberately emphasizes changes. The saved-distribution swap did not
isolate an architectural cause: its pooled retention recovery was uneven
across seeds and reversed under equal-service weighting. A correction fixed
entirely by FIT counts is a cheaper control than another learned head.

## One fixed correction

Use all three original model families and all three original seeds. Preserve
the normalized uncorrected distributions and report all nine corrected ones,
18 model/readout/seed cells in total. Do not add hybrids, tune correction
strength, search a threshold or choose a seed. This is a separate posthoc
development diagnostic, not a pre-training hypothesis or fresh confirmation.

For a supported candidate c and the supplied previous index j, define a
hypothetical transition stratum using public inputs only:

```text
g(c, j) = changed                 if c != j
        = unmentioned_retention  if c == j and previous candidate type is NONE
        = assigned_retention     otherwise

w_g      = N / (3 * n_g)
a[c]     = original_log_probability[c] - log(w_g(c,j))
corrected_log_probability[c] = a[c] - logsumexp(a[supported candidates])
```

Use the pinned integer FIT counts above and evaluate the analytic weight
formula in float64. The training implementation casts its weights to float32;
this diagnostic inverts the declared analytic objective, not a bit-exact
finite-precision optimizer. Record those analytic weights and their float32
training representations. Targets, observed transition labels and held-out
frequencies enter scoring only, never correction construction.

At the ideal weighted-cross-entropy population optimum,
`q_weighted(c|x) ∝ w_g(c,j) * p(c|x)`, so division by the prescribed weight
recovers the unweighted conditional distribution. This is a mathematical
motivation, not a calibration guarantee. Model misspecification, regularization,
finite optimization and service shift all limit that interpretation.

All alternatives share the changed-state weight, so their conditional
distribution remains unchanged. The correction increases stay log-odds by
`log(n_retention / 2299)`, choosing the retention count from the previous type.
Do not describe fewer false updates alone as better performance: an increased
keep preference can also miss real changes.

## Inputs, numerical checks and reporting

Reuse the [previous diagnostic's fixed saved-input pins](dialogue-alignment-commitment-diagnostic.md#frozen-saved-input-identities):
the completed-run receipt, training plan, canonical evaluation rows, all nine
prediction files and original report/audit receipts. No raw corpus, new gold,
feature cache, encoder or checkpoint access is required. The correct previous
candidate remains a privileged evaluator input.

Authenticate every selected member before array decoding. Preserve the
13,599-row canonical order, float32 source schema, supported candidate counts
and off-support negative infinity. Promote to float64 and use stable log-sum-exp.
Apply the same source-mass limit `2e-6` and final unit-mass limit `1e-12` as the
completed commitment diagnostic, with no clipping or probability floor. Keep
first-index argmax tie handling and record exact ties. Require conditional
alternative reconstruction within `atol=1e-10, rtol=1e-12`.

Before any real calculation, test candidate permutation equivariance of the
distribution, label independence, NONE versus assigned previous types,
two-candidate support, padding, extremely small finite alternative mass,
normalization, the analytic keep-odds shift, and raw versus normalized NLL.
Freeze implementation, tests and this specification before execution. An
independent primary checker must use separate arithmetic.

Retain all/seen-service/held-out-service panels, every service and the existing
all/changed/retained/unmentioned/assigned strata. Report counts, accuracy,
wrong-selected-branch and within-branch errors, NLL, Brier, supported rare
recall/false-positive rates, and paired repairs/harms. Show row, equal-service
and equal-dialogue weighting, every seed and equal-seed means. The primary
support stays 7,819 rows: 578 changed and 7,241 retained. Preserve zero-support
rates as null. Do not use a new weighted utility score to hide the tradeoff.

The producer and independent primary audit each have a 60-second wall limit,
1 GiB process-lifetime peak-RSS limit and 64 MiB output limit. Use exclusive
directories, preserve failures and recheck hashes at completion. Report the
correction's cost separately from the original model-inference cost. No new
pass/fail gate or architecture-admission decision is implied by this diagnostic.

## What a later training comparison would need

If the fixed correction changes the tradeoff materially, the smallest fresh
training control is the same aligned scorer with the original objective versus
ordinary row-uniform cross-entropy, across three paired seeds. That six-fit
comparison still needs its own frozen recipe, cost allocation and useful-gain
criteria. A weighting explanation should be credited to the objective before
claiming a new memory or commitment mechanism.

Simply writing categorical cross-entropy as a keep/change loss plus a
conditional alternative loss changes no objective. For scores s, previous
index j, correction r and `A = logsumexp(s[alternatives])`, ordinary candidate
softmax has keep log-odds `s[j] + r - A`. Replacing that with `s[j] + r`
also changes common-score-shift invariance and the equal-score prior from
`1 / candidate_count` to `1/2`. Those are confounds, not an isolated test of
factorization.

A possible later comparison can instead contrast that coupled keep logit with
`s[j] - mean(s[alternatives]) + r - log(number_of_alternatives)`. Both are
invariant to a common score shift and agree when alternative scores are equal.
The intervention then removes alternative-score dispersion from commitment.
It still requires matched controls and is not established novelty. Shared
features would still receive gradients through the commitment head.

## Relevant prior art

[Menon et al., ICLR 2021](https://arxiv.org/abs/2007.07314) study frequency-based
logit adjustment during training and after fitting. Our proposed correction
undoes a known conditional stratum weighting; it is not a reproduction of their
long-tail benchmark recipe or their balanced-error target.

[Ren et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html)
study Balanced Softmax and complementary sampling under long-tailed label
distributions. This supports controlling the objective and sampling before
attributing a difference to architecture; it does not predict our result.

[Morin and Bengio, AISTATS 2005](https://proceedings.mlr.press/r5/morin05a.html)
provide an established hierarchical conditional-probability model. Probability
factorization alone is not a new architecture contribution.
