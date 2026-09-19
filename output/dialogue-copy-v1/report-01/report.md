# Dialogue copy development results

Saved-prediction cohort authentication and arithmetic; frozen prior metric helpers reused; no neural replay.

Fixed selective-copy continuation: **FAIL** (7/13 checks).
Official development only, with supplied service/slot/candidates. Official test remains untouched.

| Method | Parameters per fit | Training seconds, 3 fits | Evaluation seconds, 3 fits |
|---|---:|---:|---:|
| readout | 99458 | 202.094 | 4.695 |
| scalar | 99458 | 234.053 | 5.405 |
| selective | 99458 | 243.027 | 5.584 |
| selective_no_lexical | 99458 | 237.498 | 5.369 |
| candidate_gru | 103411 | 329.378 | 5.808 |

## Seen services

| Method | Macro accuracy (%) | Micro NLL | Brier | Revision accuracy (%) | Revision queries per fit |
|---|---:|---:|---:|---:|---:|
| readout | 73.1893 | 0.7246 | 0.3695 | 57.2614 | 241 |
| scalar | 79.1220 | 0.5230 | 0.2633 | 55.0484 | 241 |
| selective | 78.0051 | 0.5535 | 0.2830 | 56.1549 | 241 |
| selective_no_lexical | 71.4879 | 0.7066 | 0.3673 | 40.2490 | 241 |
| candidate_gru | 79.7646 | 0.4992 | 0.2523 | 56.4315 | 241 |
| Reference: none | 33.3333 | not probabilistic | not probabilistic | 0.0000 | 241 |
| Reference: literal | 54.7453 | not probabilistic | not probabilistic | 49.7925 | 241 |

## Unseen services

| Method | Macro accuracy (%) | Micro NLL | Brier | Revision accuracy (%) | Revision queries per fit |
|---|---:|---:|---:|---:|---:|
| readout | 65.7673 | 0.8573 | 0.4012 | 62.5616 | 203 |
| scalar | 72.5810 | 0.7516 | 0.3716 | 72.5780 | 203 |
| selective | 72.8918 | 0.7429 | 0.3707 | 73.5632 | 203 |
| selective_no_lexical | 51.7817 | 1.0151 | 0.4844 | 39.4089 | 203 |
| candidate_gru | 68.6825 | 0.7875 | 0.3581 | 70.6076 | 203 |
| Reference: none | 33.3333 | not probabilistic | not probabilistic | 0.0000 | 203 |
| Reference: literal | 65.7642 | not probabilistic | not probabilistic | 79.8030 | 203 |

Evaluation includes batch assembly and prediction storage; excludes encoder and lexical preparation. Each unique schema is advanced through public turns; dense padded work is charged. These batch timings are not incremental serving latency.

Macro equally weights the three state strata, then the three fits. No-lexical is an ablation, not a selectable primary. Missing required strata or revision support cannot pass their checks.

Lexical extraction, literal decisions, initialization digests and neural computation are source/test bound. This reporter reuses frozen cohort/metric functions and rechecks saved references, labels, probabilities and criteria; it is not an independent implementation of every pipeline stage. Raw dialogue/predictions remain local.
