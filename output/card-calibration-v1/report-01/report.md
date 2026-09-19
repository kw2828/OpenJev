# Card calibration transfer and native control

Temperature continuation: **FAIL**, 11/12 expanded checks across five jointly required conditions.

The former 64 C evaluation decks are calibration training for this explicitly new study. Eighteen scalar temperatures are frozen before 64 fresh paired decks. Neural weights and C selection rules are unchanged. Original architecture result remains FAIL, 1/6.

| Policy | Family | Mean native return | Successes |
|---|---|---:|---:|
| baseline | delta | 0.336038 | 67/192 |
| baseline | gated_delta | 0.382312 | 82/192 |
| baseline | kalman | 0.392829 | 89/192 |
| baseline | innovation_local | 0.390925 | 86/192 |
| baseline | innovation_matched | 0.384115 | 83/192 |
| baseline | gru | -0.975361 | 0/192 |
| temperature | delta | 0.657051 | 192/192 |
| temperature | gated_delta | 0.655749 | 192/192 |
| temperature | kalman | 0.653145 | 192/192 |
| temperature | innovation_local | 0.650841 | 192/192 |
| temperature | innovation_matched | 0.649740 | 192/192 |
| temperature | gru | -0.987079 | 0/192 |
| hard | delta | 0.659655 | 192/192 |
| hard | gated_delta | 0.658153 | 192/192 |
| hard | kalman | 0.653546 | 192/192 |
| hard | innovation_local | 0.654447 | 192/192 |
| hard | innovation_matched | 0.651342 | 192/192 |
| hard | gru | -0.930589 | 0/192 |
| shared reference, evaluated once | exact | 0.699219 | 64/64 |
| shared reference, evaluated once | last32 | 0.679688 | 64/64 |

| Fit | Beta | Temperature minus baseline | Hard minus baseline |
|---|---:|---:|---:|
| delta-pair0 | 3.67364 | +0.235577 | +0.240986 |
| delta-pair1 | 3.24416 | +0.376803 | +0.377103 |
| delta-pair2 | 4.20896 | +0.350661 | +0.352764 |
| gated_delta-pair0 | 3.27297 | +0.185096 | +0.191106 |
| gated_delta-pair1 | 3.16148 | +0.327224 | +0.327825 |
| gated_delta-pair2 | 4.07772 | +0.307993 | +0.308594 |
| kalman-pair0 | 3.08049 | +0.163161 | +0.167969 |
| kalman-pair1 | 3.3992 | +0.334435 | +0.337440 |
| kalman-pair2 | 4.19525 | +0.283353 | +0.276743 |
| innovation_local-pair0 | 2.93383 | +0.158053 | +0.160757 |
| innovation_local-pair1 | 3.26312 | +0.336538 | +0.342849 |
| innovation_local-pair2 | 3.64507 | +0.285156 | +0.286959 |
| innovation_matched-pair0 | 3.24902 | +0.158654 | +0.158053 |
| innovation_matched-pair1 | 3.20891 | +0.346154 | +0.351863 |
| innovation_matched-pair2 | 3.69788 | +0.292067 | +0.291767 |
| gru-pair0 | 0.098681 | -0.012620 | +0.054087 |
| gru-pair1 | 0.278274 | -0.009916 | +0.037861 |
| gru-pair2 | 0.203695 | -0.012620 | +0.042368 |

Proper scores below use identical saved fresh baseline prefixes, before current-visibility/unseen overrides. They average queries within a boundary, nonempty boundaries within an episode, episodes within a fit, and all 18 fits equally.

| Belief | Hierarchical NLL | Hierarchical Brier | Top-rank accuracy | Infinite NLL queries |
|---|---:|---:|---:|---:|
| baseline | 0.696981 | 0.221553 | 84.7310% | 0 |
| temperature | 0.444019 | 0.162069 | 84.7310% | 0 |
| hard | infinite | 0.305380 | 84.7310% | 63915 |

| Required check | Value | Threshold | Result |
|---|---:|---:|---|
| mean_temperature_minus_baseline_at_least_0.03 | 0.2280982905982906 | >= 0.03 | PASS |
| pair0_aggregate_positive | 0.14798677884615383 | > 0.0 | PASS |
| pair1_aggregate_positive | 0.2852063301282051 | > 0.0 | PASS |
| pair2_aggregate_positive | 0.25110176282051283 | > 0.0 | PASS |
| fresh_C_prefix_NLL_at_least_5_percent_lower | 0.4440193883430516 | <= 0.6621317657513325 | PASS |
| fresh_C_prefix_Brier_nonworse | 0.1620688982488979 | <= 0.22155283343280038 | PASS |
| delta_native_loss_at_most_0.01 | 0.32101362179487175 | >= -0.01 | PASS |
| gated_delta_native_loss_at_most_0.01 | 0.2734375 | >= -0.01 | PASS |
| kalman_native_loss_at_most_0.01 | 0.2603165064102564 | >= -0.01 | PASS |
| innovation_local_native_loss_at_most_0.01 | 0.2599158653846154 | >= -0.01 | PASS |
| innovation_matched_native_loss_at_most_0.01 | 0.265625 | >= -0.01 | PASS |
| gru_native_loss_at_most_0.01 | -0.01171875 | >= -0.01 | FAIL |

Scalar calibration: 32.522s, 1224 objective/derivative evaluations. Fresh native evaluation: 211.676s, 336,784 actions across 3,584 games. No new neural fit.

Actual instrumented wall time on a shared host; transformations, C diagnostics, native actions and serialization are included. Different trajectory lengths and transformation arithmetic preclude intrinsic equal-work speed claims. Per-policy restore metadata is repeated and counted once per controller.

Previously exposed 64 C decks are declared calibration training, never new test evidence.
Fresh proper scores use exactly baseline C histories and causal seen-hidden labels for every transform; native trajectories may diverge.
Hierarchy: equal fits, equal eligible episodes, equal nonempty boundaries, equal eligible queries; not a pooled-query score.
Hard wrong labels have infinite NLL, represented by null plus flags/counts; no smoothing or omitted errors.
Finite-beta log-domain likelihood distinguishes true zero support from floating exponent underflow.
All 18 inherited fits retained; no new neural weights or new memory architecture.
Exact and last32 each run once per deck, not replicated across belief policies.
Learned outputs are authenticated saved probabilities; this audit performs no checkpoint forward or native simulation.
Five jointly required conditions are not twelve independent statistical tests. The original architecture gate remains failed, 1/6.
