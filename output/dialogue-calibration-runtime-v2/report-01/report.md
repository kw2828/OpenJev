# Complete output-temperature control after runtime V2 admission

Technical coverage: 12/12. Calibration control: FAIL (9/11). Original raw result remains FAIL (6/7).

Saved-only development control, not untouched confirmation. Temperatures use calibration TRAIN endpoints only. No model/encoder/tokenizer calls, checkpoint deserialization, optimizer or official TEST access. Upstream configuration, cohort selection and internal execution witnesses are authenticated through the frozen reader; the numerical scorer reuses the qualified original group definitions. Original raw FAIL6/7 and the V1 failed cost gate remain unchanged.

V1 cost admission remains failed. The separate V2 pilot passed all twelve runtime verification blocks; its success does not change any scientific condition.

| Fit | Beta | Fit location | Raw unseen NLL | Normalized unseen NLL | Calibrated unseen NLL | Calibrated unseen Brier |
|---|---:|---|---:|---:|---:|---:|
| frozen_original-6901 | 1.0549654 | interior | 0.77501905 | 0.77501905 | 0.79275541 | 0.39158816 |
| frozen_numbers-6901 | 1.0133888 | interior | 0.66994483 | 0.66994483 | 0.67427083 | 0.3491323 |
| trainable_original-6901 | 0.61988643 | interior | 0.76076295 | 0.76076296 | 0.56971825 | 0.2863535 |
| trainable_numbers-6901 | 0.57246633 | interior | 0.80056337 | 0.80056337 | 0.568053 | 0.28990544 |
| frozen_original-6902 | 1.0376078 | interior | 0.67560827 | 0.67560827 | 0.68538829 | 0.323909 |
| frozen_numbers-6902 | 1.0700306 | interior | 0.71910138 | 0.71910138 | 0.7407552 | 0.37846025 |
| trainable_original-6902 | 0.61041383 | interior | 1.0219501 | 1.0219501 | 0.7384641 | 0.37435907 |
| trainable_numbers-6902 | 0.62968082 | interior | 0.87517073 | 0.87517074 | 0.66066413 | 0.3286378 |
| frozen_original-6903 | 1.0620149 | interior | 0.7449109 | 0.7449109 | 0.76528542 | 0.36329303 |
| frozen_numbers-6903 | 1.0316303 | interior | 0.72674792 | 0.72674792 | 0.73776592 | 0.37599875 |
| trainable_original-6903 | 0.60367289 | interior | 0.89610876 | 0.89610876 | 0.64656964 | 0.34699151 |
| trainable_numbers-6903 | 0.59200367 | interior | 0.80588546 | 0.80588547 | 0.59807444 | 0.32052305 |

| Condition | Result |
|---|---|
| unseen_macro_gain_1pp | PASS |
| unseen_macro_strict_paired_wins | PASS |
| unseen_micro_nll_nonworse | PASS |
| unseen_micro_brier_nonworse | PASS |
| seen_macro_deficit_at_most_1pp | PASS |
| seen_assigned_retention_error_increase_at_most_half_pp | PASS |
| unseen_assigned_retention_error_increase_at_most_half_pp | PASS |
| trainable_numbers_unseen_micro_nll_improves_own_raw | PASS |
| trainable_numbers_unseen_micro_brier_nonworse_own_raw | PASS |
| frozen_numbers_unseen_micro_nll_nonworse_own_raw | FAIL |
| frozen_numbers_unseen_micro_brier_nonworse_own_raw | FAIL |

| Phase | Actual parent wall seconds |
|---|---:|
| prepare | 8.3768614 |
| qualify | 10.421521 |
| pilot | 49.405351 |
| infer | 148.97924 |
| Total of four separate phases | 217.18297 |

Every seed, paired change, factorial mean and normalization-only drift is retained in [summary.json](summary.json). Its fits index links the twelve complete diagnostics, including seen/unseen and per-service/type/bin panels. Temperature fitting used calibration endpoints only; unchanged predictions cannot establish a new memory or architecture benefit.
