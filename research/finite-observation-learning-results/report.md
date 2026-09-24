# Synthetic observation learning diagnostic

**Base: BASE_TRAINABLE_FAIL. Shift: SHIFT_TRANSFER_FAIL.** These are separate prespecified checks of the tied retentive arm. All five arms remain reported.

![All arms, seeds and horizons](benchmark.png)

This is an eight-state synthetic classification-cost problem. It does not establish an architectural improvement, robotics gain, native-environment transfer or novelty. Earlier study outcomes remain unchanged.

All 15 models trained from scratch for 48 epochs on horizons 1 and 2. All final checkpoints were saved before either DEV set was generated. Evaluation uses horizons 1, 2, 4 and 8. The shifted sensor law is not announced to the model; its observation KL is descriptive and is not part of the shift gate.

[Frozen protocol](../finite-observation-learning-protocol.md). Original source pins, process closures and opaque payload hashes were checked before reading the independent audit.

| Split | Attempted | Retained | Found during prefix |
|---|---:|---:|---:|
| train | 512 | 485 | 27 |
| base | 128 | 121 | 7 |
| shift | 128 | 118 | 10 |

Excluded prefixes were not replaced. Future found events remain in the targets. Blind targets include surviving probability mass without renormalization. Observed targets condition on each realized history. Observed cost and survival targets become zero after found; the event-law target is the absorbing found outcome.

| Gate | Passed | Conditions passed | Failed conditions |
|---|---|---:|---|
| BASE_TRAINABLE_FAIL | False | 9/27 | 420261001_h1_observed_kl, 420261001_h2_observed_kl, 420261001_h4_half_mse, 420261001_h4_half_regret, 420261001_h8_half_mse, 420261001_h8_half_regret, 420261002_h1_observed_kl, 420261002_h2_observed_kl, 420261002_h4_half_mse, 420261002_h4_half_regret, 420261002_h8_half_mse, 420261002_h8_half_regret, 420261003_h1_observed_kl, 420261003_h2_observed_kl, 420261003_h4_half_mse, 420261003_h4_half_regret, 420261003_h8_half_mse, 420261003_h8_half_regret |
| SHIFT_TRANSFER_FAIL | False | 9/21 | 420261001_h4_half_mse, 420261001_h4_half_regret, 420261001_h8_half_mse, 420261001_h8_half_regret, 420261002_h4_half_mse, 420261002_h4_half_regret, 420261002_h8_half_mse, 420261002_h8_half_regret, 420261003_h4_half_mse, 420261003_h4_half_regret, 420261003_h8_half_mse, 420261003_h8_half_regret |

The gate checks every tied-retentive fit seed, not only its average. The complete condition booleans and all 120 model metric rows are retained in [summary.json](summary.json).

## Three-fit means at longer horizons

| Arm | Regime | Horizon | Blind regret | Blind cost MSE | Observation KL | Shuffled-prefix regret |
|---|---|---:|---:|---:|---:|---:|
| tied_dense | base | 4 | 0.607855 | 0.124998 | 0.7149 | 0.607855 |
| tied_dense | base | 8 | 0.535408 | 0.101389 | 0.728155 | 0.535408 |
| tied_dense | shift | 4 | 0.51196 | 0.0944174 | 0.340311 | 0.51196 |
| tied_dense | shift | 8 | 0.449609 | 0.0772569 | 0.332378 | 0.449609 |
| untied_dense | base | 4 | 0.573388 | 0.1245 | 0.714903 | 0.578374 |
| untied_dense | base | 8 | 0.527318 | 0.101257 | 0.728157 | 0.527318 |
| untied_dense | shift | 4 | 0.50287 | 0.0946741 | 0.340276 | 0.501946 |
| untied_dense | shift | 8 | 0.427387 | 0.0767611 | 0.332399 | 0.424175 |
| tied_retentive | base | 4 | 0.593246 | 0.126117 | 0.747259 | 0.592369 |
| tied_retentive | base | 8 | 0.510776 | 0.101556 | 0.702859 | 0.552691 |
| tied_retentive | shift | 4 | 0.449673 | 0.0938832 | 0.387008 | 0.510569 |
| tied_retentive | shift | 8 | 0.423085 | 0.0776319 | 0.33891 | 0.449502 |
| untied_retentive | base | 4 | 0.589107 | 0.12628 | 0.747303 | 0.595753 |
| untied_retentive | base | 8 | 0.531736 | 0.102021 | 0.702945 | 0.515055 |
| untied_retentive | shift | 4 | 0.491922 | 0.0948153 | 0.387168 | 0.505927 |
| untied_retentive | shift | 8 | 0.44887 | 0.0772917 | 0.339005 | 0.445076 |
| gru | base | 4 | 0.601757 | 0.142961 | 0.177911 | 0.59811 |
| gru | base | 8 | 0.550008 | 0.12115 | 0.185123 | 0.557653 |
| gru | shift | 4 | 0.488167 | 0.10451 | 0.213465 | 0.526821 |
| gru | shift | 8 | 0.438829 | 0.092832 | 0.246521 | 0.449076 |

Means average the scores of three separate policies. They are not ensemble decisions or confidence intervals. Regret compares the chosen decision with the minimum exact expected cost. The uniform-state reference knows the world dynamics but discards the prefix; it is not the optimal history-ignorant policy because retained prefixes condition the state distribution. Only this reference uses the frozen 1e-12 tie allowance. Model decisions use raw argmin.

The prefix shuffle moves the complete public history and its length by a fixed cyclic offset within each regime. It holds future actions and targets fixed. This is a descriptive sensitivity check.

## Recorded work

Completed fits: 15. Optimizer updates: 5760. Completed training case exposures: 349200.

| Original phase | Wall seconds through cleanup |
|---|---:|
| qualify | 2.856 |
| fit | 23.845 |
| audit | 0.508 |

| Target construction | Seconds |
|---|---:|
| base | 0.039 |
| shift | 0.035 |
| train | 0.099 |

| Arm | Seed | Parameters | Parameter bytes | Updates | Fit seconds |
|---|---:|---:|---:|---:|---:|
| tied_dense | 420261001 | 8778 | 49728 | 384 | 2.053 |
| untied_dense | 420261001 | 9618 | 56448 | 384 | 1.603 |
| tied_retentive | 420261001 | 8778 | 49728 | 384 | 1.566 |
| untied_retentive | 420261001 | 9618 | 56448 | 384 | 1.469 |
| gru | 420261001 | 11181 | 44724 | 384 | 1.206 |
| untied_dense | 420261002 | 9618 | 56448 | 384 | 1.514 |
| tied_retentive | 420261002 | 8778 | 49728 | 384 | 1.456 |
| untied_retentive | 420261002 | 9618 | 56448 | 384 | 1.481 |
| gru | 420261002 | 11181 | 44724 | 384 | 1.130 |
| tied_dense | 420261002 | 8778 | 49728 | 384 | 1.442 |
| tied_retentive | 420261003 | 8778 | 49728 | 384 | 1.502 |
| untied_retentive | 420261003 | 9618 | 56448 | 384 | 1.589 |
| gru | 420261003 | 11181 | 44724 | 384 | 1.133 |
| tied_dense | 420261003 | 8778 | 49728 | 384 | 1.467 |
| untied_dense | 420261003 | 9618 | 56448 | 384 | 1.499 |

| Prediction regime | Sum of 15 prediction-view seconds |
|---|---:|
| base | 0.154 |
| shift | 0.149 |

Whole-phase timings include process startup and cleanup. Fit timers include Adam construction, training, order/epoch logs and final checkpoint writes; initial model construction and state hashing are outside those timers. Prediction timers include public tensor copies, all three prediction routes (blind, observed and shuffled blind), validation and output copies. These components are nested inside the fit phase and must not be added to it. Shared updates do not imply matched parameter count, precision or total compute.

## Limits of the evidence

- The caller supplies original process/source/payload authentication before these numerical reads.
- Predictions, trained checkpoints and optimizer updates are not re-executed; this audits saved outputs only.
- Checkpoint bytes, paired schedules, barrier and inference-state records agree; historical execution order remains a producer attestation.
- Case namespace and public target construction are checked, but RNG sampling histories are not independently replayed.
- Shuffled predictions are scored as saved; their producer permutation and causal input isolation remain source-reviewed.
- The uniform-state known-dynamics reference is history-ignorant, not an optimal history-ignorant policy.
- An oracle-realizable synthetic world does not establish native-environment transfer or architectural novelty.
