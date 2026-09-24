# Proposal: supervise the learned filter on its public prefix predictions

**Executed as [finite-prefix-learning-v1](finite-prefix-learning-results.md).** The original prospective design is retained below. The [completed shared-filter study](finite-shared-filter-results/report.md) failed all three criteria for the shared model: 6/24, 9/21 and 2/8 conditions passed. Its lower mean H8 regret and observed KL than the two controls are descriptive differences, not a passing result. The next question is whether direct observable prediction during prefix filtering improves held-out decisions under the same fixed training budget.

## One intervention, two arms

Use the same eight-state `shared_filter` model for both arms, paired initialization and three fit seeds per arm:

- **Endpoint only:** the existing four-component H1/H2 forecast objective, evaluated on cases surviving the prefix. Here “endpoint” means the current forecast objective after prefix encoding, not a new final-horizon-only loss.
- **Endpoint plus prefix:** the identical objective plus mean prefix negative log-likelihood, with coefficient **1**, fixed before data generation.

The reset emission, shared action-observation operators, fixed cost readout, optimizer and training schedule remain identical. No warm start, weight sweep, partial-sharing search, extra architecture or posterior-supervision target is included.

## Predict first, then assimilate

For the initial ordinary odor, predict `p(o0) = sum_s E0[o0,s] / 8` from the uniform reset prior, score the realized odor, then condition. This event has four outcomes and occurs before any action or found hazard.

For each subsequent public action, compute the full five-event distribution from the learner's current normalized state and its branch operators. Score the realized ordinary odor or found event **before** conditioning on that label. Ordinary events update the normalized state. Found ends the observed prefix and produces absorbing zero state. No true hidden state, belief array or oracle boundary posterior enters either learned model.

Define the added loss as the sum of valid prefix-event NLLs divided by the total number of valid prefix events in the fixed TRAIN attempt pool. This is a global event mean, not a separate division by each realized sequence length. Include the first found event; exclude every post-found padding position. A zero assigned probability for an observed event is an explicit failure, never epsilon-clipped.

## Retain the attempted population

The current generator discards prefixes that terminate in found. It cannot supply this experiment unchanged. A new, separately tested collector and adapter must retain **all attempted prefixes**, including found-terminated ones, with explicit lengths, event masks and endpoint-eligibility masks. No replacement sampling and no likelihood fitted only to survivors.

Use fresh TRAIN and DEV attempt pools and a new source-bound registration. Both arms receive the same attempts and the same surviving endpoint-training cases. Keep the endpoint objective normalized over eligible cases and prefix NLL over valid events; freeze these dataset denominators, the corresponding minibatch inclusion-probability scaling, and one paired minibatch schedule before execution. Do not shrink the loss scale merely by dividing each minibatch by a full-dataset denominator. A minibatch with no eligible endpoint targets contributes zero endpoint loss, not a resampled batch. This prevents termination frequency or batch composition from silently changing the relative loss weight.

For a uniformly sampled attempt batch of actual size `b` from `N` attempts, use `(N / b) * (sum_eligible endpoint_loss / N_surviving + lambda * sum_valid prefix_event_NLL / E_valid)`, with `lambda` equal to 0 or 1 by arm. Use the actual size of a final partial batch. Keep Adam steps on all-ineligible endpoint-only batches with explicit zero gradients so update counts remain equal. Stored optimizer moments can still move parameters on such a step; record it as a zero-endpoint batch, not as an unchanged model.

Use six fits total, 480 epochs, batch size 64, Adam at 0.003 and gradient clipping at 5, with equal optimizer-update counts across paired arms. Preserve every final checkpoint before fresh DEV generation. The added loss deliberately uses more observed supervision; equal epochs and updates do **not** mean equal gradient examples, operations or runtime. Report active-event counts, eligible cases, all update counts, parameters, per-fit and inference timings, and complete original phase costs.

## Evaluation and limits

Retain the [same three criteria](finite-shared-filter-protocol.md) and thresholds, requiring every fit seed to pass: short-horizon learning at H1/H2, blind extrapolation at H4/H8, and observed filtering extrapolation at H4/H8. Preserve support minima, positive reference denominators, all failed cells and paired differences. Add fresh-prefix NLL over the entire DEV attempt pool as a descriptive diagnostic, with its event and attempt denominators. It cannot rescue failed decision or forecast criteria.

Lower training loss alone is insufficient: it can reflect better fit to observed prefix events without useful held-out control. Even a criterion pass would establish a bounded supervision result, not recovery of the true state. The known, fixed cost basis remains privileged, has rank at most three, and distinguishes only four decision signatures across eight states. There is no calibration, native-transfer, robotics or novelty claim.

## Established context

[PSIM (2016)](https://proceedings.mlr.press/v48/sun16.html) learns filtering directly as a composition of predictors in predictive-state space. It motivates evaluating the learner's inference procedure rather than assuming a fitted latent model filters well. This proposal does not implement PSIM or inherit its guarantees.

[Recurrent Kalman Networks (2019)](https://proceedings.mlr.press/v97/becker19a.html) combine end-to-end learning with structured, factorized Kalman updates and locally linear dynamics. They support structured recurrent inference as an established approach; their Gaussian setting differs from these categorical operators and supplies no guarantee here.

[PSRNNs (2017)](https://proceedings.neurips.cc/paper/2017/file/2bb0502c80b7432eee4c5847a5fd077b-Paper.pdf) connect observation-gated bilinear filtering with recurrent networks and combine two-stage-regression initialization with backpropagation through time. That combination is established prior work. This proposal keeps the current paired random initialization and isolates the prefix-loss intervention.
