# Known and learned factors in a finite observation world

This diagnostic separates learning a useful prefix state from learning transition and observation operators. The three prespecified criteria remain separate; the ordinary GRU is descriptive.

![Every fitted model and both forecast panels](benchmark.png)

| Prefix state | Operators | Arm | Privileged information |
|---|---|---|---|
| Exact posterior | Exact world | `exact_exact` | Exact prefix posterior and known dynamics; no learned parameters |
| Learned from public history | Exact world | `learned_exact` | Known dynamics and fixed state-basis cost readout |
| Exact posterior | Learned | `exact_learned` | Exact prefix posterior and fixed state-basis cost readout |
| Learned from public history | Learned | `learned_learned` | Fixed state-basis cost readout |

All four factor cells use the same exact linear cost readout. The ordinary GRU is a separate control with learned prediction heads. Exact posterior inputs are privileged and do not represent a deployable policy.

The exact/exact control has zero trained parameters. Its five output fields agree with independently reconstructed targets at every saved TRAIN and DEV horizon within 1e-12. The chart marks the corresponding zero-error reference; this is a correctness control, not a fitted architecture result.

[Frozen protocol](../finite-factor-learning-protocol.md). All original process closures, registered source identities and payload hashes were authenticated before metrics were read.

## Prespecified criteria

| Criterion | Learned prefix / exact operators | Exact prefix / learned operators | Both learned |
|---|---|---|---|
| SHORT_HORIZON_LEARNING | FAIL (8/24) | PASS (24/24) | FAIL (6/24) |
| BLIND_EXTRAPOLATION | FAIL (15/21) | PASS (21/21) | FAIL (9/21) |
| OBSERVED_FILTERING_EXTRAPOLATION | FAIL (7/8) | PASS (8/8) | FAIL (2/8) |

Each criterion requires all three fit seeds and minimum support. Short-horizon learning checks H1 and H2 separately: at most half the uniform-reference cost MSE and regret, plus observed KL at most 0.1. Blind extrapolation checks the same relative cost criteria separately at H4 and H8, plus H8 survival MAE at most 0.05. Observed filtering extrapolation checks KL at most 0.1 separately at H4 and H8. It receives intervening observations and is not blind extrapolation. Relative criteria require a strictly positive reference denominator.

All condition booleans, all 48 model metric rows, all four uniform-reference rows, and current source pins are in [summary.json](summary.json). There is no selected seed or replacement arm.

## Three-fit means at longer horizons

| Arm | Horizon | Blind regret | Blind cost MSE | Observed KL | Blind survival MAE | Shuffled-prefix regret |
|---|---:|---:|---:|---:|---:|---:|
| learned_exact | 4 | 0.274069 | 0.0991979 | 0.0934861 | 0.00184456 | 0.61284 |
| learned_exact | 8 | 0.243374 | 0.0786432 | 0.016981 | 0.00269063 | 0.566425 |
| exact_learned | 4 | 0.000103868 | 0.00304828 | 0.0109242 | 0.00143446 | 0.584337 |
| exact_learned | 8 | 0.000247742 | 0.0100585 | 0.0294225 | 0.00283621 | 0.529509 |
| learned_learned | 4 | 0.56937 | 0.125974 | 0.506532 | 0.0034429 | 0.590043 |
| learned_learned | 8 | 0.51471 | 0.101938 | 0.473709 | 0.0049933 | 0.509902 |
| gru | 4 | 0.495957 | 0.185967 | 0.205532 | 0.0235348 | 0.603912 |
| gru | 8 | 0.530811 | 0.170112 | 0.148137 | 0.0626664 | 0.528943 |

Means summarize separately fitted policies; they are not ensemble decisions or confidence intervals. For the exact-prefix arm, the shuffled view moves the oracle posterior together with its corresponding public prefix. The uniform-state reference uses known dynamics after discarding the observed prefix; it is not the optimal policy that ignores history.

## Data and measured work

| Split | Attempted | Retained | Found during prefix |
|---|---:|---:|---:|
| train | 512 | 484 | 28 |
| base | 128 | 120 | 8 |

Excluded prefixes were not replaced. Future found outcomes stay in the targets. Blind predictions retain unconditional surviving mass; observed predictions condition on the realized history and become absorbing after found.

All 12 learned models trained for **480 epochs** on H1/H2 targets, with batch size 64, learning rate 0.003 and gradient clip 5. All final checkpoints preceded DEV generation. This uses ten times the earlier 48-epoch schedule on fresh cases and different factor controls. It is not a matched causal comparison against that earlier study.

| Arm | Seed | Trainable parameters | Parameter bytes | Updates | Case exposures | Fit seconds |
|---|---:|---:|---:|---:|---:|---:|
| learned_exact | 421261001 | 5,356 | 22,352 | 3,840 | 232,320 | 17.525 |
| learned_exact | 421261002 | 5,356 | 22,352 | 3,840 | 232,320 | 16.446 |
| learned_exact | 421261003 | 5,356 | 22,352 | 3,840 | 232,320 | 16.924 |
| exact_learned | 421261001 | 1,056 | 8,448 | 3,840 | 232,320 | 7.211 |
| exact_learned | 421261002 | 1,056 | 8,448 | 3,840 | 232,320 | 7.114 |
| exact_learned | 421261003 | 1,056 | 8,448 | 3,840 | 232,320 | 7.726 |
| learned_learned | 421261001 | 6,412 | 30,800 | 3,840 | 232,320 | 17.105 |
| learned_learned | 421261002 | 6,412 | 30,800 | 3,840 | 232,320 | 16.635 |
| learned_learned | 421261003 | 6,412 | 30,800 | 3,840 | 232,320 | 17.292 |
| gru | 421261001 | 11,181 | 44,724 | 3,840 | 232,320 | 13.495 |
| gru | 421261002 | 11,181 | 44,724 | 3,840 | 232,320 | 13.680 |
| gru | 421261003 | 11,181 | 44,724 | 3,840 | 232,320 | 14.976 |
| exact_exact | deterministic | 0 | 0 | 0 | 0 | Included in generation/exact-control work |

Fit timings include Adam construction, training, durable journals, fixed-buffer checks and final checkpoint writing. They exclude model construction. Parameter storage excludes buffers, activations and optimizer storage. The factor arms have different trainable work, so this is not a matched-compute or speedup claim.

| Original closed phase | Seconds |
|---|---:|
| qualify | 3.053 |
| fit | 167.768 |
| audit | 0.779 |
| Total of these three phases | 171.600 |

These are complete native supervisor durations, including process launch and cleanup. Qualification includes engineering tests. The scientific fit phase includes exact-control forecasts, generation, training, evaluation and output writing. The audit reconstructs saved targets and scalar results without model or generator calls.

Recorded scientific work: 46,080 optimizer steps; 2,787,840 case exposures; 1 untrained exact-control construction; 10 exact blind and 10 exact observed batch rollouts; 24 learned blind, 24 observed and 24 shuffled batch rollouts. All declared work counters are retained in the summary.

## Interpretation limits

This is a diagnostic of useful forecast learning under a fixed budget, not a proof of convergence, latent-state recovery or architectural novelty. The fixed cost readout distinguishes four decision signatures, not every coordinate of an eight-state posterior. A privileged arm succeeding does not establish native-environment, robotics, Doom, chess or connectome transfer. Previous study outcomes remain unchanged.
