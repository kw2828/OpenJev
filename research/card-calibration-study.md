# Does confidence limit learned card memory?

**Prospective experiment. No calibration fit or new test game has run yet.** This study keeps all 18 trained memories and the exploration controller fixed, changes only their output probabilities, and evaluates 3,584 fresh games. Its purpose is to establish a stronger baseline before proposing another memory architecture.

A [saved-output diagnostic](../output/card-confidence-diagnostic-v1/attempt-01/short.md) found that three strong families identify previously seen hidden cards correctly about 99.4% of the time, while their mean maximum probability is about 81%. Most repeated mismatches involve a correct top-ranked answer whose rank differs from the pending card. This is a diagnosis from already exposed histories, not evidence that a correction works.

## Fixed comparison

For each of the 18 final checkpoints, fit one inverse temperature on all 64 previously published C-controller trajectories. These former test histories are now explicitly **calibration training data** for this separate study. Their old evaluation results and continuation decisions remain unchanged.

Then evaluate three choices on the same 64 new, previously unused deck seeds:

| Choice | Beliefs supplied to the unchanged exploration controller |
| --- | --- |
| Baseline | Original model probabilities |
| Temperature | Probabilities raised to one fitted positive power, then normalized |
| Hard | One-hot at the lowest-index highest-probability rank |

All public visibility overrides, memory writes, action costs and weights stay fixed. Exact public memory and the last 32 reveals each run once per new deck. That gives 18 fits x 3 policies x 64 decks plus 2 references x 64 decks: **3,584 games**, not 3,584 independent layouts.

The scalar minimizes NLL with equal weighting over episodes, nonempty decision boundaries within each episode, and queried positions within each boundary. Only previously revealed, currently hidden cards supply labels. Fit beta in [0.05, 20] by a fixed convex search; preserve zeros and record numerical underflow. Hard predictions have infinite NLL whenever their chosen rank is wrong. They are a nonprobabilistic control, not a smoothed probabilistic model.

On the **new baseline trajectories**, also score all three transformations on exactly the same saved predictions and public histories. This isolates proper-score transfer from the different histories generated during gameplay.

## Continuation rule

Temperature must satisfy every condition:

1. Improve mean native return across all 18 fits by at least 0.03.
2. Improve return for each of the three paired-fit groups, averaging the six families within each group.
3. Reduce NLL by at least 5% on the fixed new baseline prefixes.
4. Do not increase their mean Brier score.
5. Do not lower any family's mean return by more than 0.01.

These five requirements expand into 12 check rows. The hard control is descriptive; it cannot replace temperature as the claimed winner after seeing the results. The original proposed architecture remains **FAIL, 1/6**, even if calibration passes. A pass earns a stronger control for later work, not novelty, transfer or conference readiness.

No new neural training is allowed. Budgets are 18 scalar fits, 900 seconds for calibration, then 18 checkpoint restores and at most 372,736 native actions within 1,800 seconds. Preserve an interrupted attempt; no automatic retries or budget extensions. The first new deck with Kalman pair 0 is the preset replay, regardless of outcome.

[Protocol](../evidence/card-calibration-v1/protocol.json) · [New seeds and fixed order](../evidence/card-calibration-v1/inputs.json) · [Calibration design](../output/card-confidence-direction-v1/calibration-design.md) · [Prior controller results](card-controller-comparison.md)

Temperature scaling is established calibration methodology, not our architectural contribution. See [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html). Here we apply a power transform to saved float32 softmax probabilities cast to float64; these are not the original pre-softmax logits.
