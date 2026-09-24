# Shared learned filtering in a synthetic observation world

This diagnostic compares learned recurrent prefix filtering with shared versus separate forecast operators, and the existing GRU-prefix learned-operator model. Every learned arm receives public history only. The three prespecified criteria remain separate and apply to every arm.

![Every fitted model, seed and forecast horizon](benchmark.png)

| Arm | Prefix processing | Forecast operators |
|---|---|---|
| `shared_filter` | Learned reset emission and recurrent action/odor filtering | Same learned branch operators |
| `untied_filter` | Learned reset emission and separate recurrent action/odor filtering | Independently trainable learned branch operators |
| `gru_prefix` | Existing 31-to-28 GRU and eight-state projection | Learned branch operators |

The two filters condition a uniform reset prior using a learned four-odor emission map. This initial odor comes before any action or hazard. They then process the eight public action/odor pairs. Only the two filter arms have matched initial prefix functions; their untied operator parameters start as separate copies. All three arms have paired initial forecast operators. The GRU prefix is not initially function-matched.

All learned arms use the same privileged, fixed state-basis cost readout. None receives an oracle prefix posterior or future hidden state. Only the untrained exact/exact correctness control receives the true prefix posterior and known dynamics. Its five output fields match reconstructed TRAIN and DEV targets at all saved horizons within 1e-12. The dotted zero reference represents that control, not a trained result.

[Frozen protocol](../finite-shared-filter-protocol.md). Original process closures, registered source identities and payload hashes were authenticated before result metrics were read.

## Prespecified criteria

| Criterion | Shared filter | Untied filter | GRU prefix |
|---|---|---|---|
| SHORT_HORIZON_LEARNING | FAIL (6/24) | FAIL (7/24) | FAIL (6/24) |
| BLIND_EXTRAPOLATION | FAIL (9/21) | FAIL (9/21) | FAIL (9/21) |
| OBSERVED_FILTERING_EXTRAPOLATION | FAIL (2/8) | FAIL (2/8) | FAIL (2/8) |

Each criterion requires all three fit seeds and minimum support. At H1/H2, short-horizon learning requires blind cost MSE and regret at most half the positive uniform reference, and observed KL at most 0.1 nats. Blind extrapolation applies the cost thresholds at H4/H8 and requires H8 survival MAE at most 0.05. Observed filtering extrapolation requires KL at most 0.1 separately at H4/H8. It receives intervening observations.

All condition booleans, all 36 model rows, four uniform-reference rows, paired-fit metadata and current source pins are retained in [summary.json](summary.json). No seed or arm is selected after evaluation. Absolute criterion passes alone do not establish superiority of sharing.

## Three-fit means at longer horizons

| Arm | Horizon | Blind regret | Blind cost MSE | Observed KL | Blind survival MAE | Shuffled-prefix regret |
|---|---:|---:|---:|---:|---:|---:|
| shared_filter | 4 | 0.417234 | 0.108387 | 0.197721 | 0.00307741 | 0.550538 |
| shared_filter | 8 | 0.460997 | 0.0981933 | 0.152373 | 0.00548282 | 0.522479 |
| untied_filter | 4 | 0.523185 | 0.121232 | 0.39341 | 0.00269705 | 0.580966 |
| untied_filter | 8 | 0.513748 | 0.102412 | 0.349987 | 0.0043567 | 0.522676 |
| gru_prefix | 4 | 0.504022 | 0.119517 | 0.476096 | 0.00269995 | 0.569919 |
| gru_prefix | 8 | 0.4927 | 0.100624 | 0.43344 | 0.00435474 | 0.511237 |

## Paired differences at longer horizons

Each entry is **shared_filter minus the named control**, using the same fit seed and DEV cases. Negative values favor the shared filter for that metric. These are descriptive arithmetic differences, not confidence intervals, significance tests, superiority claims or additional passing criteria.

| Control | Seed | Horizon | Blind regret difference | Blind cost MSE difference | Observed KL difference |
|---|---:|---:|---:|---:|---:|
| untied_filter | 422261001 | 4 | -0.256715 | -0.0357549 | -0.245082 |
| untied_filter | 422261002 | 4 | +0.0142843 | +0.00724387 | -0.167957 |
| untied_filter | 422261003 | 4 | -0.0754216 | -0.0100221 | -0.174027 |
| untied_filter | 422261001 | 8 | -0.0558693 | -0.00462091 | -0.268535 |
| untied_filter | 422261002 | 8 | -0.0365111 | -0.00181662 | -0.157317 |
| untied_filter | 422261003 | 8 | -0.065873 | -0.0062173 | -0.166992 |
| gru_prefix | 422261001 | 4 | -0.188635 | -0.0265233 | -0.319905 |
| gru_prefix | 422261002 | 4 | +0.101473 | +0.011937 | -0.163965 |
| gru_prefix | 422261003 | 4 | -0.173201 | -0.0188036 | -0.351257 |
| gru_prefix | 422261001 | 8 | -0.0242106 | -0.00374438 | -0.302531 |
| gru_prefix | 422261002 | 8 | -0.00378273 | +0.000814855 | -0.243401 |
| gru_prefix | 422261003 | 8 | -0.0671154 | -0.00436155 | -0.297271 |

These means summarize separately fitted policies, not an ensemble or confidence interval. The shuffle moves complete public prefixes and lengths by one position, with forecast actions and targets unchanged. The uniform reference uses known dynamics while discarding the prefix; it is not the optimal history-ignorant policy.

## Data and measured work

| Split | Attempted | Retained | Found during prefix |
|---|---:|---:|---:|
| train | 512 | 481 | 31 |
| base | 128 | 122 | 6 |

Excluded prefixes were not replaced. Future found cases remain in the targets. Blind predictions carry unconditional surviving mass. After observed found, cost and survival become zero and the event law is found with probability one.

All nine learned fits use 480 epochs, H1/H2 training targets, batch size 64, Adam at 0.003 and gradient clip 5. The four-part objective combines blind cost MSE, observed cost MSE, half the sum of survival MSEs and observed event-law soft cross-entropy. There is no posterior-supervision or prefix-likelihood loss. Every final checkpoint precedes DEV generation. Equal epochs and data do not equalize computation.

| Arm | Seed | Trainable parameters | Parameter bytes | Updates | Case exposures | Fit seconds |
|---|---:|---:|---:|---:|---:|---:|
| shared_filter | 422261001 | 1,088 | 8,704 | 3,840 | 230,880 | 16.524 |
| shared_filter | 422261002 | 1,088 | 8,704 | 3,840 | 230,880 | 16.158 |
| shared_filter | 422261003 | 1,088 | 8,704 | 3,840 | 230,880 | 16.077 |
| untied_filter | 422261001 | 2,144 | 17,152 | 3,840 | 230,880 | 16.460 |
| untied_filter | 422261002 | 2,144 | 17,152 | 3,840 | 230,880 | 16.457 |
| untied_filter | 422261003 | 2,144 | 17,152 | 3,840 | 230,880 | 16.700 |
| gru_prefix | 422261001 | 6,412 | 30,800 | 3,840 | 230,880 | 16.316 |
| gru_prefix | 422261002 | 6,412 | 30,800 | 3,840 | 230,880 | 16.827 |
| gru_prefix | 422261003 | 6,412 | 30,800 | 3,840 | 230,880 | 16.345 |
| exact_exact | deterministic | 0 | 0 | 0 | 0 | Included in generation/exact-control work |

Fit timings include optimizer construction, training, journals, buffer checks and checkpoint writes, but exclude model construction. Parameter bytes exclude buffers, activations and optimizer storage. Filters use float64 arithmetic; the GRU prefix uses float32 recurrence with float64 projection/operators/readout.

| Arm | Seed | DEV cases | Evaluation seconds |
|---|---:|---:|---:|
| shared_filter | 422261001 | 122 | 0.015228 |
| shared_filter | 422261002 | 122 | 0.014478 |
| shared_filter | 422261003 | 122 | 0.012834 |
| untied_filter | 422261001 | 122 | 0.012607 |
| untied_filter | 422261002 | 122 | 0.012317 |
| untied_filter | 422261003 | 122 | 0.012363 |
| gru_prefix | 422261001 | 122 | 0.010943 |
| gru_prefix | 422261002 | 122 | 0.010412 |
| gru_prefix | 422261003 | 122 | 0.011298 |

Evaluation timings include input copies, blind and observed forecasts, the shuffled blind forecast, guards and prediction copies. They exclude state hashing and scalar metrics, and are not per-decision latency.

## Structural forward work by arm and route

| Arm | Route | Filter update rows | GRU update rows | Reset emission rows | Prefix / forecast softmax calls | Transition rows | Conditioned rows | Cost-readout rows |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| shared_filter | evaluation_blind | 2,928 | 0 | 366 | 6 / 6 | 2,928 | 0 | 2,928 |
| shared_filter | evaluation_observed | 2,928 | 0 | 366 | 6 / 6 | 2,928 | 2,835 | 2,928 |
| shared_filter | evaluation_shuffled | 2,928 | 0 | 366 | 6 / 6 | 2,928 | 0 | 2,928 |
| shared_filter | training_blind | 5,541,120 | 0 | 692,640 | 11,520 / 11,520 | 1,385,280 | 0 | 1,385,280 |
| shared_filter | training_observed | 5,541,120 | 0 | 692,640 | 11,520 / 11,520 | 1,385,280 | 1,372,320 | 1,385,280 |
| untied_filter | evaluation_blind | 2,928 | 0 | 366 | 6 / 6 | 2,928 | 0 | 2,928 |
| untied_filter | evaluation_observed | 2,928 | 0 | 366 | 6 / 6 | 2,928 | 2,835 | 2,928 |
| untied_filter | evaluation_shuffled | 2,928 | 0 | 366 | 6 / 6 | 2,928 | 0 | 2,928 |
| untied_filter | training_blind | 5,541,120 | 0 | 692,640 | 11,520 / 11,520 | 1,385,280 | 0 | 1,385,280 |
| untied_filter | training_observed | 5,541,120 | 0 | 692,640 | 11,520 / 11,520 | 1,385,280 | 1,372,320 | 1,385,280 |
| gru_prefix | evaluation_blind | 0 | 3,294 | 0 | 0 / 6 | 2,928 | 0 | 2,928 |
| gru_prefix | evaluation_observed | 0 | 3,294 | 0 | 0 / 6 | 2,928 | 2,835 | 2,928 |
| gru_prefix | evaluation_shuffled | 0 | 3,294 | 0 | 0 / 6 | 2,928 | 0 | 2,928 |
| gru_prefix | training_blind | 0 | 6,233,760 | 0 | 0 / 11,520 | 1,385,280 | 0 | 1,385,280 |
| gru_prefix | training_observed | 0 | 6,233,760 | 0 | 0 / 11,520 | 1,385,280 | 1,372,320 | 1,385,280 |

Counts aggregate all three fits and expose the separate prefix and forecast operator normalizations. They are source-constrained producer attestations, with fixed geometry and observed row totals checked by the audit. They exclude validation FLOPs, backward operations, optimizer work and allocations. Full counters are retained in the summary; these are not FLOP or matched-compute measurements.

| Original closed phase | Seconds |
|---|---:|
| qualify | 3.061 |
| fit | 149.573 |
| audit | 0.646 |
| Total of these three phases | 153.281 |

These are complete original supervisor durations, including launch and cleanup. Qualification includes engineering tests. Fit includes data generation, exact controls, learning, evaluation and output writes. Nested fit and inference timings must not be added again to these totals.

Recorded work: 34,560 optimizer steps; 2,077,920 case exposures; 1 untrained exact-control construction; 10 exact blind and 10 exact observed rollouts; 18 learned blind, 18 observed and 18 shuffled batch rollouts.

## Interpretation limits

This is a finite synthetic diagnostic of learned filtering under a fixed budget. The shared versus untied comparison tests parameter sharing within this model class; the GRU comparison also changes encoder structure, precision and capacity. Classical filtering structure is not a novelty claim. Known C and the filter reset prior remain privileged task assumptions. Four centered decision costs do not identify every coordinate of an eight-state posterior. There is no convergence, latent-recovery, architectural-superiority, native-environment, robotics, Doom, chess or connectome claim. Earlier study outcomes are unchanged; this is not a matched cross-study improvement claim.
