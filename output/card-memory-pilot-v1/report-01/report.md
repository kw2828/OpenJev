# Learned card-memory pilot

Continuation rule: **FAIL** (1/6 checks).

All 18 fits completed before evaluation: six modes, three paired initializations, 128 training episodes and 128 updates per fit. Each final policy played the same 64 fresh ConcentrationHard seeds. Public exact-table and last-32 references use the same decision rule.

| Memory | Mean native return | Successes | Old hidden-card accuracy | Parameters | End-to-end ms/action |
| --- | ---: | ---: | ---: | ---: | ---: |
| delta | -0.1103 | 0/192 | 66.52% | 2191 | 0.523 |
| gated_delta | -0.0802 | 0/192 | 65.21% | 2191 | 0.521 |
| kalman | -0.0805 | 0/192 | 62.35% | 2289 | 0.550 |
| innovation_local | -0.0781 | 0/192 | 63.05% | 2290 | 0.556 |
| innovation_matched | -0.0817 | 0/192 | 62.30% | 2290 | 0.560 |
| gru | -0.9654 | 0/192 | 9.38% | 865325 | 0.535 |
| exact | 0.7028 | 64/64 | n/a | symbolic | 0.099 |
| last32 | 0.6562 | 64/64 | n/a | symbolic | 0.095 |

Old-card accuracy measures development queries more than 32 actions since last public visibility. It is a separate metric from native return. Timings include model restore, public bookkeeping, game execution and episode storage on this host; they are not model-only latency or matched compute.

Strongest conventional learned arm: **gated_delta**. Local minus this baseline, by paired fit: -0.0018, +0.0000, +0.0081.

Local minus gain-matched diffuse control, by fit: -0.0054, +0.0072, +0.0090.

| Continuation criterion | Result |
| --- | --- |
| utility_over_strongest_conventional | FAIL |
| utility_positive_all_pairs | FAIL |
| local_over_matched_utility | FAIL |
| local_over_matched_positive_all_pairs | FAIL |
| long_age_accuracy | FAIL |
| long_age_coverage | PASS |

The proposed innovation rule is a heuristic extension of existing delta/Kalman memory. Its covariance is not calibrated uncertainty. The GRU has the same 512 mean-state scalars but many more parameters; Kalman variants add 32 covariance scalars. Every arm also receives public seen/matched/phase bookkeeping. This is not a connectome experiment, end-to-end RL, or a POPGym leaderboard reproduction.

No checkpoint selection, replacement seeds, or outcome-driven tuning occurred in this attempt. The next experiment, if warranted, must be separately specified on fresh data.
