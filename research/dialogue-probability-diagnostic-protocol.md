# Diagnose the completed observation-learning log-loss regression

21 September 2026. This is a new, post hoc diagnostic of exposed official DEV,
specified after seeing V2's complete aggregate results and before computing the
diagnostics below. V2 remains a scientific continuation failure: six conditions
passed and the unseen-NLL condition failed. No new model, training, actor rollout,
API call, fitted calibrator, official TEST access or architecture comparison is
admitted by this protocol.

## Inputs and authentication

Use every saved prediction from all four V2 arms and seeds 6901, 6902 and 6903.
Each fit has the same 62,329 canonical endpoints. Bind the complete run, frozen
plan, supervisor launch/terminal, producer report/summary, successful independent
audit/summary and actual saved-reader exit review by external SHA-256 pins.
The qualified V2 auditor authenticates full run/freeze/source/report membership
before evaluator rows or prediction arrays are decoded. Its 64 scientific
sources remain unchanged. Also authenticate the independent audit's complete
file membership and its exact producer/run/plan joins. Reconstruct the raw
baseline and compare it with the published fit metrics before new analysis.

Freeze this protocol, diagnostic source and tests in a separate plan after
synthetic qualification. Every output uses a new exclusive directory. Limits:
300 suspend-inclusive seconds, 4 GiB RSS and 128 MiB output. Record start, actual
work, source/input/output hashes, native elapsed time and terminal status. Stop
and preserve any failure; no automatic retry or partial-result interpretation.
Do not alter a completed V2 artifact or its seven-condition decision.

## Fixed descriptive calculations

Report each fit separately, with all/seen/unseen panels and each panel's three
strata. Empty cells stay undefined. Report seed means only as equally weighted
descriptions of repeated optimization on the same DEV examples.

1. Partition raw target NLL by correct and incorrect top-choice decisions.
   Report counts, NLL sums, conditional means and contributions divided by the
   full panel/stratum count. Verify the partition reconstructs the baseline.
2. Report top-choice confidence bins with lower bounds
   `0, 0.5, 0.7, 0.9, 0.95, 0.99`. Bins include their lower bound and exclude
   their upper bound; the final bin has no finite upper bound. Record counts,
   mean confidence, accuracy and confidence minus accuracy. Separately count
   confidence above 1. This preserves permitted raw float32 roundoff within
   V2's 2e-6 mass tolerance without clipping, dropping or renormalizing rows.
   These bins describe top-choice reliability, not full multiclass calibration.
3. For target-probability thresholds `1e-2, 1e-4, 1e-6`, report strict-below
   counts and raw NLL contributions. Compare target log probabilities with log
   thresholds, preserving finite NLL under probability underflow. Thresholds
   are nested and their contributions must not be added together.
4. For the primary trainable_numbers versus frozen_numbers pair, report all
   four correctness transitions on identical endpoints for each seed and group.
   At each tail threshold also use the common subset where either model is
   below the threshold, plus its complement. Report both models' NLL sums,
   conditional means, full-group contributions and the signed difference.
   Each exhaustive partition must reconstruct the original paired NLL gap.
5. Show the complete output-only temperature grid
   `0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4` for every fit. For T other than 1, divide
   supported saved logs by T and normalize with float64 masked log-sum-exp.
   T=1 is an explicit identity branch preserving raw saved logs and the original
   baseline exactly. Report NLL and candidate-sum Brier, validating finite
   supported logs, negative-infinite padding, mass and unchanged canonical
   first-argmax choices. No transformed output feeds recurrent state.

No temperature is optimized, chosen, deployed or used to reverse V2's failure.
The grid measures sensitivity on exposed DEV. A positive result would not prove
that weight training caused overconfidence, that tails alone explain the gap,
or that a calibrator generalizes. The matched decompositions determine how much
of the gap is inside each fixed subset without causal attribution.

## What can follow

A future calibration experiment needs separately frozen dialogues excluded from
weight fitting, the same fitting procedure for frozen and trainable models, all
seeds, and unchanged actor state. Establish split eligibility before obtaining
new predictions. This diagnostic does not itself authorize temperature selection
on DEV or a claim about recurrent, connectome or world-model superiority.

Temperature scaling is established prior art:
[Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html).
Distribution shift remains a separate concern:
[Ovadia et al., NeurIPS 2019](https://papers.nips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html).
Those papers motivate a practical control, not a novelty claim for this work.
