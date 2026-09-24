# Fresh-seed replication of learned cost readouts

**REPLICATION_CONTINUATION_FAIL (6/8 conditions).**

This prospective continuation rule is separate from the three absolute learning criteria below. A continuation pass permits consideration of a separately registered follow-up; it does not admit execution or establish that blind extrapolation passes. All earlier study outcomes remain unchanged.

Two arms share the same learned reset emission, recurrent branch operators and coefficient-one prefix loss. `fixed_softened` uses 0.9C; `learned_readout` uses 0.25 minus a softmax over decisions in each latent-state column, initialized to match 0.9C within 1e-12.

![Every fitted seed, endpoint forecasts, prefix likelihood and fitting cost](benchmark.png)

The learned head adds 32 logits, or 24 identifiable head degrees of freedom. Learned versus softened isolates trainability from initialization. This replication does not repeat the original fixed-exact-C learned arm, so gains here do not independently repeat that stronger anchor comparison and might partly undo softening. Both arms start with world-aligned readouts.

All learned models receive public history only. Every attempted prefix, including its first found event, is retained for coefficient-one event-mean NLL; post-found padding is excluded. Endpoint targets exist only for surviving prefixes and use a separate fixed denominator. There is no oracle posterior or belief-supervision input to a learner. The exact correctness reference alone receives the true boundary posterior and dynamics.

[Frozen protocol](../finite-cost-readout-replication-protocol.md). Original qualification, fit and audit closures, source pins and complete payload hashes were authenticated before reading these metrics. Exact-control forecasts match independently reconstructed targets within 1e-12.

## Separate prospective continuation rule

| Condition | Outcome |
|---|---|
| 425261001_h4_strictly_lower_regret | PASS |
| 425261001_h8_strictly_lower_regret | PASS |
| 425261002_h4_strictly_lower_regret | PASS |
| 425261002_h8_strictly_lower_regret | PASS |
| 425261003_h4_strictly_lower_regret | PASS |
| 425261003_h8_strictly_lower_regret | PASS |
| learned_observed_filtering_extrapolation | FAIL |
| learned_short_horizon_learning | FAIL |

All eight conditions are required: learned short-horizon and observed-filtering criteria, plus strictly lower H4/H8 blind regret than softened for each of the three paired seeds. A tie fails. Means and prefix NLL cannot rescue a condition. These are prospective rules for this replication, not revisions to earlier gates.

## Unchanged absolute criteria

| Criterion | Fixed softened | Learned readout |
|---|---|---|
| SHORT_HORIZON_LEARNING | FAIL (11/24) | FAIL (21/24) |
| BLIND_EXTRAPOLATION | FAIL (9/21) | FAIL (14/21) |
| OBSERVED_FILTERING_EXTRAPOLATION | FAIL (2/8) | FAIL (7/8) |

Every fit seed must pass each criterion and minimum support. H1/H2 learning requires blind cost MSE and regret at most half the positive uniform reference and observed KL at most 0.1 nats. Blind H4/H8 extrapolation uses the same cost thresholds and H8 survival MAE at most 0.05. Observed H4/H8 filtering requires KL at most 0.1 and receives intervening observations. Prefix likelihood cannot rescue these criteria.

## Longer-horizon means

| Arm | H | Blind regret | Blind cost MSE | Observed cost MSE | Observed KL | Blind survival MAE |
|---|---:|---:|---:|---:|---:|---:|
| fixed_softened | 4 | 0.463759 | 0.119596 | 0.16587 | 0.117215 | 0.00555164 |
| fixed_softened | 8 | 0.439449 | 0.093656 | 0.165907 | 0.13839 | 0.00794042 |
| learned_readout | 4 | 0.187608 | 0.052273 | 0.0546675 | 0.065843 | 0.00602899 |
| learned_readout | 8 | 0.195671 | 0.0523046 | 0.0586194 | 0.08042 | 0.00799658 |

Means describe three separately fitted policies, not an ensemble or confidence interval.

## Paired differences against the softened control

Every difference is learned minus the named control on the same seed and cases. Negative favors learned for that metric. The six strict regret comparisons enter the separately declared continuation rule. Other differences are descriptive arithmetic, not significance tests or additional gates.

| Control | Seed | H | Regret difference | Blind MSE difference | Observed MSE difference | KL difference |
|---|---:|---:|---:|---:|---:|---:|
| fixed_softened | 425261001 | 4 | -0.0597665 | -0.0320654 | -0.0682219 | -0.0134113 |
| fixed_softened | 425261002 | 4 | -0.569702 | -0.120732 | -0.176993 | -0.103221 |
| fixed_softened | 425261003 | 4 | -0.198985 | -0.049173 | -0.0883913 | -0.0374824 |
| fixed_softened | 425261001 | 8 | -0.0756282 | -0.011816 | -0.0684838 | +0.00908436 |
| fixed_softened | 425261002 | 8 | -0.486428 | -0.0836265 | -0.164942 | -0.116691 |
| fixed_softened | 425261003 | 8 | -0.169277 | -0.0286118 | -0.0884381 | -0.0663036 |

## Readout changes for every fit

| Arm | Seed | Mean absolute change | Maximum absolute change | Initial SHA256 | Final SHA256 |
|---|---:|---:|---:|---|---|
| fixed_softened | 425261001 | 0 | 0 | d6f63ee233f1e1f1b9fb9420937ea8dead56a67164de9955b109207636bdb337 | d6f63ee233f1e1f1b9fb9420937ea8dead56a67164de9955b109207636bdb337 |
| learned_readout | 425261001 | 0.260399 | 0.95471 | 887658d8e3c8cd134a01881d4b04ab78eddc5554d0f55d7f23c023094b2baafc | 395fb8bf31b5eb351baee5af6c42f6dfbf9ba985300dc67ec37ed8b8750e8be9 |
| learned_readout | 425261002 | 0.367537 | 0.966505 | 887658d8e3c8cd134a01881d4b04ab78eddc5554d0f55d7f23c023094b2baafc | 4596f74f0d04e6ec07c5cca2ecccf18f2cda49467665d2bd04bf8bff26683861 |
| fixed_softened | 425261002 | 0 | 0 | d6f63ee233f1e1f1b9fb9420937ea8dead56a67164de9955b109207636bdb337 | d6f63ee233f1e1f1b9fb9420937ea8dead56a67164de9955b109207636bdb337 |
| fixed_softened | 425261003 | 0 | 0 | d6f63ee233f1e1f1b9fb9420937ea8dead56a67164de9955b109207636bdb337 | d6f63ee233f1e1f1b9fb9420937ea8dead56a67164de9955b109207636bdb337 |
| learned_readout | 425261003 | 0.254028 | 0.966621 | 887658d8e3c8cd134a01881d4b04ab78eddc5554d0f55d7f23c023094b2baafc | dd274b38fcb7dabcd0c2c2f94d0b2586223c211fa96b4d7e9b0a102a649afda2 |

The audit checks these saved 4-by-8 matrices and their hashes of little-endian float64 bytes. Every initial/final matrix is retained in the summary; all learned-head matrices are also printed below. Matrix movement is not evidence of latent-state recovery or alignment.

### Learned head, seed 425261001

| Decision | Initial eight-state costs | Final eight-state costs |
|---|---|---|
| 0 | -0.675, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, -0.675 | -0.748678, 0.248224, 0.242801, 0.245187, 0.246615, -0.232313, 0.239606, -0.73885 |
| 1 | 0.225, -0.675, 0.225, 0.225, 0.225, 0.225, -0.675, 0.225 | 0.24959, -0.459336, 0.246808, 0.247847, -0.27948, 0.245559, 0.244637, 0.24375 |
| 2 | 0.225, 0.225, 0.225, -0.675, -0.675, 0.225, 0.225, 0.225 | 0.249347, 0.249065, 0.197502, 0.185528, 0.234852, -0.254237, -0.72971, 0.247381 |
| 3 | 0.225, 0.225, -0.675, 0.225, 0.225, -0.675, 0.225, 0.225 | 0.249741, -0.0379531, -0.687112, -0.678562, -0.201988, 0.240991, 0.245466, 0.247719 |

### Learned head, seed 425261002

| Decision | Initial eight-state costs | Final eight-state costs |
|---|---|---|
| 0 | -0.675, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, -0.675 | 0.2435, 0.2489, 0.244704, 0.248302, 0.249504, -0.735519, 0.191518, -0.748984 |
| 1 | 0.225, -0.675, 0.225, 0.225, 0.225, 0.225, -0.675, 0.225 | -0.74071, -0.723162, 0.248561, 0.249452, 0.24058, 0.249005, 0.24493, 0.249622 |
| 2 | 0.225, 0.225, 0.225, -0.675, -0.675, 0.225, 0.225, 0.225 | 0.24975, 0.249598, -0.740703, 0.243751, 0.242307, 0.239213, -0.683797, 0.249707 |
| 3 | 0.225, 0.225, -0.675, 0.225, 0.225, -0.675, 0.225, 0.225 | 0.24746, 0.224664, 0.247438, -0.741505, -0.732391, 0.247301, 0.247348, 0.249654 |

### Learned head, seed 425261003

| Decision | Initial eight-state costs | Final eight-state costs |
|---|---|---|
| 0 | -0.675, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, -0.675 | 0.244763, 0.243852, -0.738218, 0.249551, 0.24956, 0.249257, -0.206108, -0.742914 |
| 1 | 0.225, -0.675, 0.225, 0.225, 0.225, 0.225, -0.675, 0.225 | -0.72409, -0.706443, 0.249396, 0.249724, 0.247416, 0.249174, 0.234677, 0.247133 |
| 2 | 0.225, 0.225, 0.225, -0.675, -0.675, 0.225, 0.225, 0.225 | 0.249075, 0.249397, 0.242831, -0.749077, 0.244645, 0.245404, -0.27534, 0.247764 |
| 3 | 0.225, 0.225, -0.675, 0.225, 0.225, -0.675, 0.225, 0.225 | 0.230251, 0.213193, 0.24599, 0.249801, -0.741621, -0.743836, 0.24677, 0.248017 |

## Prefix likelihood and inference cost

| Arm | Seed | Attempts | Valid events | Prefix NLL | Prefix seconds | Endpoint seconds |
|---|---:|---:|---:|---:|---:|---:|
| fixed_softened | 425261001 | 128 | 1126 | 0.906871 | 0.002587 | 0.010774 |
| fixed_softened | 425261002 | 128 | 1126 | 0.936235 | 0.002501 | 0.010667 |
| fixed_softened | 425261003 | 128 | 1126 | 0.941065 | 0.002354 | 0.010853 |
| learned_readout | 425261001 | 128 | 1126 | 0.922017 | 0.002417 | 0.011187 |
| learned_readout | 425261002 | 128 | 1126 | 0.886286 | 0.002415 | 0.011287 |
| learned_readout | 425261003 | 128 | 1126 | 0.90589 | 0.002334 | 0.011383 |

Prefix timings include input copies, likelihood forwards, guards, output copies and scalar NLL, excluding state hashes and file writes. Endpoint timings include blind, observed and shuffled forecasts with copies and guards, excluding state hashes and scalar metrics. Neither is single-decision latency.

## Data, storage and measured work

| Split | Attempts | Endpoint eligible | Found-terminated | Valid prefix events |
|---|---:|---:|---:|---:|
| train | 512 | 478 | 34 | 4457 |
| base | 128 | 118 | 10 | 1126 |

All six fits use 480 epochs, batch size 64, Adam 0.003, global gradient clip 5 and the same forecast objective plus coefficient-one prefix NLL. Paired filter initialization, attempt pools and batch orders are shared. The learned and softened heads match initial functions and raw filter gradients before clipping; the additional head gradient can change the global clipping factor. Final checkpoints precede DEV generation.

| Arm | Seed | Parameters | Parameter bytes | Buffer bytes | Updates | Endpoint exposures | Prefix-event exposures | Fit seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_softened | 425261001 | 1088 | 8704 | 256 | 3,840 | 229,440 | 2,139,360 | 21.581 |
| learned_readout | 425261001 | 1120 | 8960 | 0 | 3,840 | 229,440 | 2,139,360 | 20.869 |
| learned_readout | 425261002 | 1120 | 8960 | 0 | 3,840 | 229,440 | 2,139,360 | 21.439 |
| fixed_softened | 425261002 | 1088 | 8704 | 256 | 3,840 | 229,440 | 2,139,360 | 20.832 |
| fixed_softened | 425261003 | 1088 | 8704 | 256 | 3,840 | 229,440 | 2,139,360 | 21.094 |
| learned_readout | 425261003 | 1120 | 8960 | 0 | 3,840 | 229,440 | 2,139,360 | 20.964 |

Fit time includes initial/final state and readout snapshots, optimizer construction, training, journals, buffer checks and checkpoint writing; model construction is excluded. All arithmetic is float64 except public float32 tokens. Storage excludes activations and optimizer state. Equal updates are not equal compute.

| Arm | Route | Filter rows | NLL rows | Head softmax calls | Head probability rows | Cost readout rows |
|---|---|---:|---:|---:|---:|---:|
| fixed_softened | training_blind | 5,506,560 | 0 | 0 | 0 | 1,376,640 |
| fixed_softened | training_observed | 5,506,560 | 0 | 0 | 0 | 1,376,640 |
| fixed_softened | training_prefix | 5,631,840 | 6,418,080 | 0 | 0 | 0 |
| fixed_softened | evaluation_blind | 2,832 | 0 | 0 | 0 | 2,832 |
| fixed_softened | evaluation_observed | 2,832 | 0 | 0 | 0 | 2,832 |
| fixed_softened | evaluation_shuffled | 2,832 | 0 | 0 | 0 | 2,832 |
| fixed_softened | evaluation_prefix | 2,964 | 3,378 | 0 | 0 | 0 |
| learned_readout | training_blind | 5,506,560 | 0 | 23,040 | 184,320 | 1,376,640 |
| learned_readout | training_observed | 5,506,560 | 0 | 23,040 | 184,320 | 1,376,640 |
| learned_readout | training_prefix | 5,631,840 | 6,418,080 | 0 | 0 | 0 |
| learned_readout | evaluation_blind | 2,832 | 0 | 48 | 384 | 2,832 |
| learned_readout | evaluation_observed | 2,832 | 0 | 48 | 384 | 2,832 |
| learned_readout | evaluation_shuffled | 2,832 | 0 | 48 | 384 | 2,832 |
| learned_readout | evaluation_prefix | 2,964 | 3,378 | 0 | 0 | 0 |

Readout snapshots are separately counted: 12 exports, including 6 learned-head softmaxes. They are inside fit timings but outside forward-route counters. Each learned forward head normalization processes eight latent columns. Work counts are source-constrained producer attestations checked against schedules and labels; they exclude backward operations, optimizer work, validation FLOPs and allocation.

| Original closed phase | Seconds |
|---|---:|
| qualify | 3.477 |
| fit | 128.221 |
| audit | 1.274 |
| Total of successful phases | 132.972 |

Whole-phase timings include launch and cleanup; nested timings must not be added again. Any earlier failed qualification remains separately preserved in the evidence package. [summary.json](summary.json) retains all 24 endpoint rows, six prefix rows, conditions, per-seed contrasts, readout snapshots and declared work.

## Interpretation limits

The original exact-C anchor arm is not repeated here. The zero-parameter exact correctness reference is different: it receives the true posterior and operators, and is not a learned comparison arm. Both fitted arms retain privileged world-aligned initialization. Real-valued learned costs are interior to the known bounds; float64 can round a cost to a boundary without clipping. Four centered costs do not identify every coordinate of an eight-state posterior. Lower event NLL or training loss cannot rescue decision criteria, and observed filtering consumes evidence that blind forecasts do not. No convergence, calibration, architectural novelty, native transfer or cross-study improvement claim is made. Earlier outcomes remain unchanged.
