# Normalized dialogue-copy replication

Technical validity: **PASS**. Original scientific continuation: **FAIL** (7/13).

Fresh final fits after an explicit state-normalization correction. Exposed development only; official test untouched. This numerical correction is not architectural novelty.

| Method | Parameters | Training seconds, 3 fits | Evaluation seconds, 3 fits |
|---|---:|---:|---:|
| readout | 99458 | 252.107 | 5.908 |
| scalar | 99458 | 290.398 | 6.538 |
| selective | 99458 | 292.445 | 6.218 |
| selective_no_lexical | 99458 | 276.860 | 6.518 |
| candidate_gru | 103411 | 341.243 | 6.965 |

## Seen services

| Method | Three-stratum macro (%) | Micro NLL | Brier | Revision (%) | Revision queries per fit |
|---|---:|---:|---:|---:|---:|
| readout | 73.1893 | 0.7246 | 0.3695 | 57.2614 | 241 |
| scalar | 79.1227 | 0.5229 | 0.2633 | 55.0484 | 241 |
| selective | 78.0051 | 0.5535 | 0.2830 | 56.1549 | 241 |
| selective_no_lexical | 71.4879 | 0.7066 | 0.3673 | 40.2490 | 241 |
| candidate_gru | 79.7788 | 0.4992 | 0.2523 | 56.4315 | 241 |
| Reference: none | 33.3333 | not probabilistic | not probabilistic | 0.0000 | 241 |
| Reference: literal | 54.7453 | not probabilistic | not probabilistic | 49.7925 | 241 |

## Unseen services

| Method | Three-stratum macro (%) | Micro NLL | Brier | Revision (%) | Revision queries per fit |
|---|---:|---:|---:|---:|---:|
| readout | 65.7673 | 0.8573 | 0.4012 | 62.5616 | 203 |
| scalar | 72.5810 | 0.7514 | 0.3714 | 72.5780 | 203 |
| selective | 72.8923 | 0.7429 | 0.3707 | 73.5632 | 203 |
| selective_no_lexical | 51.7817 | 1.0151 | 0.4844 | 39.4089 | 203 |
| candidate_gru | 68.6738 | 0.7875 | 0.3581 | 70.6076 | 203 |
| Reference: none | 33.3333 | not probabilistic | not probabilistic | 0.0000 | 203 |
| Reference: literal | 65.7642 | not probabilistic | not probabilistic | 79.8030 | 203 |

Technical validity authenticates every saved batch's raw incoming, feature-prior, result and released-mass witness and full public-update coverage. State arithmetic, initialization tensor digests and optimizer execution remain source/test-bound; no checkpoint deserialization or neural replay.

Whole execution includes authentication, monitored training/evaluation, hashes and I/O up to the terminal receipt. Fit times include checkpoint I/O; evaluation includes assembly and prediction storage. Shared encoder/lexical preparation is separate. These are instrumented batch costs, not incremental serving latency.

Corrected fresh replication on exposed development. Official test untouched according to authenticated preparation lineage. No new architecture, full-DST or untouched-confirmation claim.

Descriptive fresh-fit replication deltas, outside the continuation rule. V1 scalar normalization failed; these are not evidence for a novel architecture or an isolated causal effect of normalization.
Descriptive deltas are retained in summary.json.
