# Saved teacher-target noise diagnosis

22 September 2026. Across all 558 TRAIN anchors, estimated finite-sample noise is **99.883% of the observed centered-target second moment**. The two fixed eight-replicate halves have correlation **-0.05218** and weighted first-argmin agreement **45.14%**. These exploratory estimates make label precision a concrete candidate explanation for the [failed matched teacher-learning experiment](otto-teacher-learning-results.md). They do not prove zero population signal, identify the cause of policy failure, or establish that 64 replicates will improve control.

The method, source and exact inputs were [frozen before cost decoding](../output/otto-target-noise-v1/plan-01.json). The single saved-data run completed successfully in 0.130521 seconds with peak RSS 34,635,776 bytes, within 120 seconds, one thread, 1 GiB RSS and 64 MiB output. It made **zero new rollout, teacher, model, optimizer or simulator calls**. No anchors or actions were selected using precision, ties or outcomes.

## Fixed calculation

For each anchor and replicate, subtract the mean cost across that anchor's eligible actions. With 16 centered replicate vectors, compute each action's sample variance with denominator 15, then divide by 16 to estimate variance of the sampled mean. Average over eligible actions. Separately compute the action-mean square of the full-panel centered means. Subtract estimated noise from that observed moment, retaining negative results.

Calculations use integer centered numerators `A*C - sum(C)` and exact integer sufficient statistics before final floating-point division. The original row weights are `558/(144*n_episode)`, verified byte-for-byte against both target arms' saved float64 and float32 weight columns. All-anchor and regime summaries use those same weights; regime summaries renormalize within the group. Both settings therefore have equal total episode weight, despite 280 versus 278 anchors. Units for all moments are squared moves.

The two predetermined splits are replicate IDs 0-7 and 8-15. Their action-mean cross moment estimates shared signal under independent replicate sampling conditional on a fixed anchor. The reported correlation is the weighted cross moment divided by the square root of the two weighted split second moments. Centering over actions makes the pooled weighted action mean zero. Undefined correlations would remain null, with no row exclusion.

| Group | Anchors / episodes | Observed moment | Estimated noise | Untruncated signal | Noise / observed | Split cross moment | Split correlation |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 558 / 144 | 3.659986 | 3.655705 | 0.004281 | 99.883% | -0.401232 | -0.052180 |
| lambda3 | 280 / 72 | 2.361576 | 2.512203 | -0.150627 | 106.378% | -0.496730 | -0.095248 |
| lambda4 | 278 / 72 | 4.958396 | 4.799207 | 0.159189 | 96.789% | -0.305733 | -0.030109 |

The corrected signal is negative at 262/558 individual anchors. Its all-anchor estimate is only 0.004281 moves², while the independent split cross-moment estimate is negative. Neither estimate is clipped to make a positive signal claim. Different finite-sample signal estimators can disagree; their discrepancy is not a new pass/fail rule. Common random numbers across first actions are preserved within each replicate. The interpretation of variance-of-the-mean relies on independent, identically distributed replicate vectors conditional on each anchor; the 558 states themselves need not be independent.

## Split rankings and ties

Ties mean exact equality of integer half-panel action-cost sums. “First” selects the lowest numerical action in a tied set. Set agreement requires identical complete argmin sets; overlap requires at least one shared minimum. Fractions before parentheses are raw counts; percentages in parentheses use original episode weights. Final-column tie counts are raw anchors for the left/right halves.

| Group | First-action agreement | Exact argmin-set agreement | Argmin-set overlap | Tied left / right |
|---|---:|---:|---:|---:|
| all | 252/558 (45.14%) | 235/558 (42.19%) | 270/558 (48.44%) | 146 / 170 |
| lambda3 | 118/280 (42.01%) | 111/280 (39.58%) | 128/280 (45.83%) | 72 / 88 |
| lambda4 | 134/278 (48.26%) | 124/278 (44.79%) | 142/278 (51.04%) | 74 / 82 |

These are descriptive consistency measures, not probabilities that the chosen action is correct. No true best-action labels are available. No confidence intervals treat correlated anchors as independent observations. Every row, its eligible actions, all 16 replicate/action costs and success flags, and its moments are retained in [rows.jsonl](../output/otto-target-noise-v1/run-01/rows.jsonl). The [summary](../output/otto-target-noise-v1/run-01/summary.json) additionally retains weighted tie frequencies and both split second moments.

## Implication for the next comparison

A fixed 16-versus-64-replicate comparison is informative here. Under the same conditional sampling assumptions, quadrupling independent replicates projects the mean-noise moment from 3.655705 to **0.913926 moves²**, and halves standard errors. This is a projection, not observed 64-replicate precision or a policy-performance forecast. The nearly zero corrected signal estimate means even that reduction may be insufficient.

The decisive test should preserve all 558 anchors, existing IDs 0-15, public features, original R16 global scale, episode weights, initialization and optimizer budgets. Append only the predeclared additional replicate IDs, then compare all paired final models on fresh full-horizon cases with the unchanged analytic controller. Do not keep only confident panels, renormalize R64 separately, select winning seeds or substitute training loss for autonomous competence. This diagnosis supplies evidence for testing precision; it does not itself admit or pass that experiment.

## Provenance and limits

The diagnostic authenticated the original empirical plan and successful worker, the completed V2 audit and its successful original parent, all 151 inherited source pins, and the exact panel/metadata/weight payload hashes before decoding costs. It consumed all **32,304 continuation records**. It did not rerun the historical audit, compressed sampler streams, public filtering or neural scores. The prior audit's limitations therefore remain relevant. This analysis uses TRAIN labels only, with no fresh evaluation or learning.

- [Source](../scripts/diagnose_otto_target_noise.py), [preparation and original execution evidence](../output/otto-target-noise-v1/engineering-review-01.json), [frozen plan](../output/otto-target-noise-v1/plan-01.json), [completed receipt](../output/otto-target-noise-v1/run-01/receipt.json), [all row results](../output/otto-target-noise-v1/run-01/rows.jsonl) and [aggregate results](../output/otto-target-noise-v1/run-01/summary.json).
- [Original label-study worker](../output/otto-teacher-learning-v1/run-01/receipt.json), [complete panels](../output/otto-teacher-learning-v1/run-01/panels.jsonl), [completed V2 audit](../output/otto-teacher-learning-v1/audit-02/receipt.json) and [its original successful parent](../output/otto-teacher-learning-v1/audit-supervision-02.terminal.json).

Plan SHA-256: `8e8041d43f4e7ab1a56163d8cd2730137aa5fdcc415705d330b161a0f3f90031`. Diagnostic source: `f7402db74f5cc0e66337ecaa639b78417afc8d8452ac616f672a9570e386e0b5`. Receipt: `7a9bece4a8e54915e69f5198134fc671996bb80d66da5ba9ac137aa7dcff41ca`. There is no efficacy gate in this exploratory diagnosis.
