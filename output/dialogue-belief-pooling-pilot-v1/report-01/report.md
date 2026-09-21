# Belief-conditioned pooling: complete development pilot

All 8 fits completed. Continuation: FAIL, 11/18 conditions passed.

Exposed development pilot: all four arms and both fixed seeds. Raw outputs only; no temperature or calibration. A pass admits a larger matched development experiment, not novelty, significance, transfer or ICLR readiness. The state-token arm is one added key/value pooling control, not a general prior-state encoder comparison.

| Fit | All-DEV macro accuracy | NLL | Brier |
|---|---:|---:|---:|
| pooled-7101 | 33.4375% | 1.0976582 | 0.56855137 |
| schema_attention-7101 | 33.5078% | 1.0982277 | 0.56829702 |
| belief_query-7101 | 33.5078% | 1.0982278 | 0.56829761 |
| state_token-7101 | 33.5078% | 1.0982288 | 0.56829529 |
| pooled-7102 | 33.3971% | 1.0909393 | 0.56504081 |
| schema_attention-7102 | 33.6148% | 1.0885246 | 0.56334626 |
| belief_query-7102 | 33.6148% | 1.0885139 | 0.56333873 |
| state_token-7102 | 33.6148% | 1.0885144 | 0.56334027 |

Accuracy equally averages three strata; proper scores average endpoints. Both fixed seeds are retained.

| Comparator | Condition | Result |
|---|---|---|
| pooled | macro_gain_at_least_1pp | FAIL |
| pooled | macro_strictly_positive_each_seed | PASS |
| pooled | nll_nonworse | PASS |
| pooled | brier_nonworse | PASS |
| pooled | unmentioned_retention_error_nonworse | FAIL |
| pooled | assigned_retention_error_nonworse | PASS |
| schema_attention | macro_gain_at_least_1pp | FAIL |
| schema_attention | macro_strictly_positive_each_seed | FAIL |
| schema_attention | nll_nonworse | PASS |
| schema_attention | brier_nonworse | PASS |
| schema_attention | unmentioned_retention_error_nonworse | PASS |
| schema_attention | assigned_retention_error_nonworse | PASS |
| state_token | macro_gain_at_least_1pp | FAIL |
| state_token | macro_strictly_positive_each_seed | FAIL |
| state_token | nll_nonworse | PASS |
| state_token | brier_nonworse | FAIL |
| state_token | unmentioned_retention_error_nonworse | PASS |
| state_token | assigned_retention_error_nonworse | PASS |

Actual supervised parent interval: 31.659 seconds, including shared caching and all eight fits.

[summary.json](summary.json) retains every fit, both seeds, seen/unseen panels, transition strata and changed subtypes. Missing descriptive support is null. Recovery concerns the next scored endpoint in the same dialogue/query; unscored turns may intervene and each model selects a different prior-error subset. It is descriptive and never changes the gate.
