# Observation-learning development comparison

Technical validity: all twelve fits completed and authenticated. Scientific continuation: **FAIL (6/7)**.

Official DEV is exposed development data. This changes observation learning and lexical matching, not recurrent architecture. No calibration, connectome or world-model advantage is established.

| Fit | Unseen macro accuracy | Unseen NLL | Unseen Brier | Seen macro accuracy | Seen assigned error | Unseen assigned error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen_original-6901 | 71.4906% | 0.775019 | 0.387912 | 79.9792% | 18.7651% | 30.7773% |
| frozen_numbers-6901 | 75.6336% | 0.669945 | 0.348283 | 85.4014% | 15.3630% | 29.6388% |
| trainable_original-6901 | 80.3047% | 0.760763 | 0.307952 | 90.0001% | 12.8138% | 20.8715% |
| trainable_numbers-6901 | 79.1010% | 0.800563 | 0.315294 | 88.6128% | 15.2855% | 22.9390% |
| frozen_original-6902 | 73.7024% | 0.675608 | 0.322517 | 80.4076% | 17.1562% | 34.0748% |
| frozen_numbers-6902 | 76.9896% | 0.719101 | 0.373711 | 85.6236% | 13.0561% | 27.3227% |
| trainable_original-6902 | 77.8080% | 1.021950 | 0.409357 | 90.4023% | 11.6604% | 20.8061% |
| trainable_numbers-6902 | 79.2228% | 0.875171 | 0.352134 | 89.9038% | 12.2419% | 21.8398% |
| frozen_original-6903 | 73.1111% | 0.744911 | 0.359915 | 79.6520% | 17.2822% | 31.2222% |
| frozen_numbers-6903 | 76.7663% | 0.726748 | 0.373751 | 85.3045% | 13.5698% | 26.5506% |
| trainable_original-6903 | 78.7582% | 0.896109 | 0.384001 | 91.1028% | 11.0206% | 20.5705% |
| trainable_numbers-6903 | 79.6969% | 0.805885 | 0.348811 | 89.7698% | 12.3195% | 20.5836% |

Seven predeclared conditions:

- unseen_macro_gain_1pp: PASS.
- unseen_macro_strict_paired_wins: PASS.
- unseen_micro_nll_nonworse: FAIL.
- unseen_micro_brier_nonworse: PASS.
- seen_macro_deficit_at_most_1pp: PASS.
- seen_assigned_retention_error_increase_at_most_half_pp: PASS.
- unseen_assigned_retention_error_increase_at_most_half_pp: PASS.

Whole execution: 17062.620 s. Training: 16447.147 s; final evaluation: 587.356 s; checkpoint writing: 0.652 s. Root and parent elapsed time include suspend. Fit phases retain V1 perf_counter diagnostics, can exclude suspend, and never determine admission; they are nested, not added to execution.

summary.json retains every arm/seed, literal reference, panel, service, transition/type support and false-positive denominator, four paired contrasts and their factorial interaction. Positive accuracy differences are better; positive NLL/Brier/error differences are worse. Missing support remains undefined.

Coverage and byte identities are checked from saved metadata. Internal normalization, gradients, parameter initialization and encoder execution are authenticated source-bound witnesses; no tensor execution is replayed. Reference semantics inherit the prepared public lexical registers. No model was called by this report.
