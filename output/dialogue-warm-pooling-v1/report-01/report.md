# Warm-start belief pooling: complete development study

All 12 adapted fits and 3 untouched references completed. Continuation: FAIL, 13/32 conditions passed.

Exposed full-cohort development comparison, all three fixed seeds. Raw probabilities only, no temperature or calibration. The trained encoders are frozen during matched head continuation with fresh optimizers. A pass permits a subsequent prospective study, not novelty, transfer, significance or ICLR readiness. The state-token arm is one added key/value pooling control, not a general prior-state encoder comparison.

| Fit | Unseen macro accuracy | Unseen changed accuracy | Unseen NLL | Unseen Brier | Seen macro accuracy |
|---|---:|---:|---:|---:|---:|
| pooled-6901 | 79.9357% | 81.3606% | 0.8257971 | 0.33302571 | 90.0686% |
| schema_attention-6901 | 79.8498% | 81.0067% | 0.8231631 | 0.33106569 | 90.0954% |
| belief_query-6901 | 79.2416% | 80.1416% | 0.82779056 | 0.33328255 | 90.0436% |
| state_token-6901 | 79.7507% | 80.7707% | 0.82290277 | 0.3312281 | 90.1142% |
| untouched-6901 | 79.1010% | 77.2316% | 0.80056337 | 0.31529375 | 88.6128% |
| pooled-6902 | 78.1976% | 77.8608% | 0.90137106 | 0.34493076 | 89.3460% |
| schema_attention-6902 | 78.6861% | 77.9394% | 0.90426789 | 0.34157579 | 89.5387% |
| belief_query-6902 | 78.8813% | 78.7652% | 0.90973776 | 0.34560458 | 89.4894% |
| state_token-6902 | 78.2976% | 77.4676% | 0.92014744 | 0.34742379 | 89.3978% |
| untouched-6902 | 79.2228% | 81.5965% | 0.87517073 | 0.35213394 | 89.9038% |
| pooled-6903 | 80.1131% | 81.8718% | 0.80601334 | 0.34080762 | 90.5622% |
| schema_attention-6903 | 80.1700% | 82.1471% | 0.79952624 | 0.33738382 | 90.2466% |
| belief_query-6903 | 79.8063% | 82.0684% | 0.81147954 | 0.34367287 | 90.2020% |
| state_token-6903 | 80.1219% | 81.9505% | 0.79882054 | 0.33695528 | 90.2165% |
| untouched-6903 | 79.6969% | 81.7538% | 0.80588546 | 0.34881066 | 89.7698% |

All three seeds remain visible. Macro accuracy equally weights the three strata; proper scores average endpoints. The gate uses exact integer-count fractions for accuracy and retention. Display rounding never changes its decisions.

| Comparator | Condition | Result |
|---|---|---|
| pooled | unseen_macro_gain_at_least_1pp | FAIL |
| pooled | unseen_macro_positive_at_least_two_seeds | FAIL |
| pooled | unseen_changed_gain_at_least_1pp | FAIL |
| pooled | unseen_nll_nonworse | FAIL |
| pooled | unseen_brier_nonworse | FAIL |
| pooled | seen_macro_decline_at_most_1pp | PASS |
| pooled | seen_assigned_retention_error_increase_at_most_half_pp | PASS |
| pooled | unseen_assigned_retention_error_increase_at_most_half_pp | PASS |
| schema_attention | unseen_macro_gain_at_least_1pp | FAIL |
| schema_attention | unseen_macro_positive_at_least_two_seeds | FAIL |
| schema_attention | unseen_changed_gain_at_least_1pp | FAIL |
| schema_attention | unseen_nll_nonworse | FAIL |
| schema_attention | unseen_brier_nonworse | FAIL |
| schema_attention | seen_macro_decline_at_most_1pp | PASS |
| schema_attention | seen_assigned_retention_error_increase_at_most_half_pp | PASS |
| schema_attention | unseen_assigned_retention_error_increase_at_most_half_pp | PASS |
| state_token | unseen_macro_gain_at_least_1pp | FAIL |
| state_token | unseen_macro_positive_at_least_two_seeds | FAIL |
| state_token | unseen_changed_gain_at_least_1pp | FAIL |
| state_token | unseen_nll_nonworse | FAIL |
| state_token | unseen_brier_nonworse | FAIL |
| state_token | seen_macro_decline_at_most_1pp | PASS |
| state_token | seen_assigned_retention_error_increase_at_most_half_pp | PASS |
| state_token | unseen_assigned_retention_error_increase_at_most_half_pp | PASS |
| untouched | unseen_macro_gain_at_least_1pp | FAIL |
| untouched | unseen_macro_positive_at_least_two_seeds | PASS |
| untouched | unseen_changed_gain_at_least_1pp | FAIL |
| untouched | unseen_nll_nonworse | FAIL |
| untouched | unseen_brier_nonworse | FAIL |
| untouched | seen_macro_decline_at_most_1pp | PASS |
| untouched | seen_assigned_retention_error_increase_at_most_half_pp | PASS |
| untouched | unseen_assigned_retention_error_increase_at_most_half_pp | PASS |

Actual supervised parent interval: 1301.172 seconds, including all loading, caching, full-DEV qualification and adaptation.

[summary.json](summary.json) retains every raw fit, all/seen/unseen panels, transition bins, candidate types, services, equal-seed means and every paired seed contrast. Recovery follows the same query after each model's previous scored error; unscored turns may intervene and the model-dependent subsets are descriptive only. Empty descriptive groups are null. No transformed-probability result, early checkpoint or favorable seed replaces the fixed raw comparison.
