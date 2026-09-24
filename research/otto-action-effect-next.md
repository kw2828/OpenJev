# Next: separate observation uncertainty from decision approximation

**Prospective diagnostic only. No new execution is admitted.** The [paired action-effect study](otto-action-effect-results.md) closed technically complete with independent audit agreement, but both registered gates failed at **6/18 cells**. Every comparison against the identical `paired_recurrent` architecture failed the new gate. Original collection, fit, and audit process closures and opaque payload hashes were independently checked against [the saved closure](../output/otto-action-effect-v1/closure-01.json), SHA-256 `63973e02c90224a7224f4626af3ddc17509d5ee18e6f5aa0625dd0b5863f8d2d`.

Against that matched recurrent control, ratios of the three fit-seed means showed action-effect error reductions of **22.78% at lambda 3 and 19.44% at lambda 4**. Long-horizon teacher-cost decision gap improved 12.34% at lambda 3 but worsened 19.79% under the lambda 4 shift. Normal-observation decision gap worsened 8.51% and improved 6.54%, respectively. These averages do not override the per-seed gate: the effect threshold passed 4/6 matched cells and the long-gap threshold only 1/6. Better effect forecasts did not reliably produce better decisions. Conditional-cost misalignment remains a hypothesis, not an established cause.

## The next diagnostic

The student predicts before future odors arrive, while its cost targets come from a teacher after those observations. For a fixed horizon and declared conditioning information `I`, define `mu_a(I) = E[C_a | I]`. For an action selected using only `I`:

```text
E[C_selected - min_a C_a | I]
  = mu_selected - min_a mu_a
    + min_a mu_a - E[min_a C_a | I].
```

The first term measures decision approximation relative to the same-information optimal action. The second measures the advantage of seeing future observations. This concerns teacher-cost labels, not demonstrated environment return.

A reference conditioned on full public history, the authenticated sampling law, and the legacy teacher filter has more information than the student's 31-feature prefix representation. It is therefore not automatically the irreducible floor for student inputs. Report future-observation uncertainty separately from remaining compression and approximation error.

Before any registration, specify terminal conditioning, legal-action support, zero-support cases, horizon weights, and the exact estimand. The existing metric averages surviving rows within each case; its random denominator prevents silently substituting a rowwise decomposition. Start with a separately named fixed-horizon diagnostic and retain the original metric.

Integrate teacher costs across hypothetical observation histories weighted by the strict sampling-law posterior. Evaluate the teacher on each history's legacy filter state. **Do not replace `E[Q(belief)]` with `Q(E[belief])`.** If enumeration is impractical, declare an approximate Monte Carlo reference, uncertainty, convergence checks, and all teacher/filter costs.

Then compare sampled-cost and conditional-mean-cost training with identical architecture, inputs, forecast objectives, optimization, and charged label budgets. No oracle state or future odor enters student inference. Use fresh evaluation seeds and a separate frozen protocol; keep this failed study and old TEST closed. Earlier [protected-readout](otto-protected-readout-results.md) and [action-focused](otto-action-focused-results.md) failures do not justify repeating the same decision-loss recipe.

## What would count as a contribution?

[Predictive State Temporal Difference Learning](https://arxiv.org/abs/1011.0041) already connects observable-history representations to value estimation. [Value equivalence](https://arxiv.org/abs/2011.03506) already defines model sufficiency through specified Bellman updates. Neither guarantees that this cost head is sufficient. A later architectural contribution would need a specific bounded representation that preserves decision-relevant conditional quantities, matched controls, and gains under observation shifts. The proposed diagnostic determines whether that is the problem worth solving.
