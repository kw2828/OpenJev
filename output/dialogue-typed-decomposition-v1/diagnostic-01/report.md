# Saved branch/value loss decomposition

Posthoc saved-output algebra. No new model, probabilities, choices, gate, fitting or confirmation set. Actual error type uses branch of selected candidate; branch-mass argmax is separately descriptive. The correct branch supplied for conditional_value_correct is label-privileged, not a deployable controller. Historical TRAIN exposure and privileged previous value remain. Separately trained models prevent causal representation claims.

Original experiment remains FAIL (5/9 checks passed).

| Fit | Total NLL | Branch NLL | Value NLL | Correct | Wrong branch | Wrong value |
|---|---:|---:|---:|---:|---:|---:|
| flat_balanced-6101 | 2.617557 | 1.947882 | 0.669675 | 267 | 259 | 52 |
| flat_balanced-6102 | 2.570637 | 1.978502 | 0.592135 | 308 | 212 | 58 |
| flat_balanced-6103 | 2.409477 | 1.703758 | 0.705719 | 301 | 206 | 71 |
| flat_stratum-6101 | 2.272073 | 1.835579 | 0.436493 | 374 | 155 | 49 |
| flat_stratum-6102 | 2.060626 | 1.683146 | 0.377480 | 443 | 82 | 53 |
| flat_stratum-6103 | 2.384691 | 1.980442 | 0.404249 | 412 | 117 | 49 |
| typed_balanced-6101 | 2.194627 | 1.679880 | 0.514747 | 217 | 315 | 46 |
| typed_balanced-6102 | 1.680112 | 1.148301 | 0.531811 | 330 | 175 | 73 |
| typed_balanced-6103 | 1.677834 | 1.142528 | 0.535306 | 275 | 236 | 67 |
| typed_stratum-6101 | 1.553922 | 1.296077 | 0.257845 | 354 | 190 | 34 |
| typed_stratum-6102 | 2.023522 | 1.709833 | 0.313689 | 334 | 201 | 43 |
| typed_stratum-6103 | 1.933299 | 1.470488 | 0.462811 | 254 | 272 | 52 |

Losses use equal-service weighting; counts cover the same 578 primary changed rows. Every fit and all descriptive contrasts are retained.

| Contrast | Seed | Branch loss change | Value loss change | Correct to wrong | Wrong to correct |
|---|---:|---:|---:|---:|---:|
| typing_balanced | 6101 | -0.268002 | -0.154928 | 70 | 20 |
| typing_balanced | 6102 | -0.830201 | -0.060323 | 30 | 52 |
| typing_balanced | 6103 | -0.561230 | -0.170413 | 76 | 50 |
| typing_stratum | 6101 | -0.539503 | -0.178648 | 89 | 69 |
| typing_stratum | 6102 | 0.026687 | -0.063791 | 127 | 18 |
| typing_stratum | 6103 | -0.509954 | 0.058562 | 176 | 18 |
| weighting_flat | 6101 | 0.112303 | 0.233181 | 130 | 23 |
| weighting_flat | 6102 | 0.295356 | 0.214654 | 147 | 12 |
| weighting_flat | 6103 | -0.276684 | 0.301470 | 128 | 17 |
| weighting_typed | 6101 | 0.383803 | 0.256902 | 169 | 32 |
| weighting_typed | 6102 | -0.561532 | 0.218122 | 71 | 67 |
| weighting_typed | 6103 | -0.327960 | 0.072495 | 56 | 77 |

Wrong branch means the branch of the actual candidate argmax. Summed branch mass can prefer a different branch. No saved decision is changed. All per-service groups, target types, paired error transitions and additive contributions are in summary.json.
