# Card controller follow-up

Controller continuation: **PASS**. Three jointly required conditions expand to 13/13 check rows; these are not independent statistical tests.

A is the original picker; B normalizes rows and uses stable numerical ties; C additionally prefers an unseen endpoint within tied first-card pairs. All 18 original fitted models are frozen. The original architecture result remains FAIL, 1/6.

| Policy | Family | Mean return | Successes | Episode ms/action |
| --- | --- | ---: | ---: | ---: |
| A | delta | -0.106971 | 0/192 | 0.553 |
| A | gated_delta | -0.095553 | 0/192 | 0.554 |
| A | kalman | -0.083534 | 0/192 | 0.597 |
| A | innovation_local | -0.088942 | 0/192 | 0.591 |
| A | innovation_matched | -0.092548 | 0/192 | 0.588 |
| A | gru | -0.969952 | 0/192 | 0.565 |
| A | exact | 0.692308 | 64/64 | 0.126 |
| A | last32 | 0.651142 | 64/64 | 0.121 |
| B | delta | -0.176683 | 0/192 | 0.515 |
| B | gated_delta | -0.155048 | 0/192 | 0.512 |
| B | kalman | -0.155649 | 0/192 | 0.556 |
| B | innovation_local | -0.156550 | 0/192 | 0.551 |
| B | innovation_matched | -0.155950 | 0/192 | 0.545 |
| B | gru | -0.976562 | 0/192 | 0.525 |
| B | exact | -0.048978 | 0/64 | 0.103 |
| B | last32 | -0.216647 | 0/64 | 0.104 |
| C | delta | 0.295873 | 54/192 | 0.513 |
| C | gated_delta | 0.351362 | 68/192 | 0.512 |
| C | kalman | 0.370092 | 83/192 | 0.557 |
| C | innovation_local | 0.360577 | 80/192 | 0.554 |
| C | innovation_matched | 0.351062 | 78/192 | 0.546 |
| C | gru | -0.968149 | 0/192 | 0.525 |
| C | exact | 0.692308 | 64/64 | 0.104 |
| C | last32 | 0.671575 | 64/64 | 0.099 |

All paired fit differences follow. Each fit uses all 64 shared fresh seed identities. References are descriptive and excluded from the controller gate.

| Fit/reference | B minus A | C minus B | C minus B positive decks |
| --- | ---: | ---: | ---: |
| delta-pair0 | -0.069411 | +0.553185 | 64/64 |
| delta-pair1 | -0.054988 | +0.421875 | 64/64 |
| delta-pair2 | -0.084736 | +0.442608 | 64/64 |
| gated_delta-pair0 | -0.061298 | +0.592548 | 64/64 |
| gated_delta-pair1 | -0.052284 | +0.450421 | 64/64 |
| gated_delta-pair2 | -0.064904 | +0.476262 | 64/64 |
| kalman-pair0 | -0.077524 | +0.618089 | 64/64 |
| kalman-pair1 | -0.059495 | +0.441406 | 63/64 |
| kalman-pair2 | -0.079327 | +0.517728 | 64/64 |
| innovation_local-pair0 | -0.071214 | +0.603966 | 64/64 |
| innovation_local-pair1 | -0.064904 | +0.442909 | 64/64 |
| innovation_local-pair2 | -0.066707 | +0.504507 | 64/64 |
| innovation_matched-pair0 | -0.073017 | +0.599159 | 64/64 |
| innovation_matched-pair1 | -0.055889 | +0.420673 | 64/64 |
| innovation_matched-pair2 | -0.061298 | +0.501202 | 64/64 |
| gru-pair0 | -0.006310 | +0.009014 | 11/64 |
| gru-pair1 | -0.004507 | +0.008113 | 10/64 |
| gru-pair2 | -0.009014 | +0.008113 | 10/64 |
| exact | -0.741286 | +0.741286 | 64/64 |
| last32 | -0.867788 | +0.888221 | 64/64 |

| Required check | Value | Threshold | Result |
| --- | ---: | ---: | --- |
| learned_mean_C_minus_B_at_least_0.03 | 0.422877 | >= 0.03 | PASS |
| delta_mean_nonnegative | 0.472556 | >= 0 | PASS |
| gated_delta_mean_nonnegative | 0.506410 | >= 0 | PASS |
| kalman_mean_nonnegative | 0.525741 | >= 0 | PASS |
| innovation_local_mean_nonnegative | 0.517127 | >= 0 | PASS |
| innovation_matched_mean_nonnegative | 0.507011 | >= 0 | PASS |
| gru_mean_nonnegative | 0.008413 | >= 0 | PASS |
| delta_at_least_two_positive_fits | 3.000000 | >= 2 | PASS |
| gated_delta_at_least_two_positive_fits | 3.000000 | >= 2 | PASS |
| kalman_at_least_two_positive_fits | 3.000000 | >= 2 | PASS |
| innovation_local_at_least_two_positive_fits | 3.000000 | >= 2 | PASS |
| innovation_matched_at_least_two_positive_fits | 3.000000 | >= 2 | PASS |
| gru_at_least_two_positive_fits | 3.000000 | >= 2 | PASS |

Whole evaluation: 204.961 seconds. 392,810 saved native actions checked. All 64 layout hashes were reconstructed from the union of actual public reveals; 64 distinct layouts, with any duplicates retained.

Instrumented episodes include construction, reset, identity read, inference, public bookkeeping, native steps, close and NPZ I/O/hash. Shared restores and receipt writes are outer costs. A deliberately pays an additional choice pass for equivalence diagnostics; these are not intrinsic or matched-work speed comparisons.

Coverage, repeat selections, picker ties, component costs and every paired deck difference are included in summary.json.

Three jointly required controller conditions, not thirteen independent significance tests.
Sixty-four paired deck draws; trajectories diverge. No claim of 3840 independent decks.
B bundles normalization and stable ties. C adds a deterministic exploration heuristic.
No new architecture, probability calibration, biological mechanism, or optimal information gain claim.
Learned probabilities are authenticated saved outputs; no checkpoint forward replay was performed.
Original failed architecture criterion remains failed; this controller test cannot rescue it.
