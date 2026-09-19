# Dialogue memory development results

Saved-prediction arithmetic and cohort authentication; no neural replay.
Official development only; supplied service/slot routing. No official-test or full-DST claim.

Fixed innovation continuation: **FAIL** (8/11, including all 21 fits).

| Method | Parameters per fit | Training seconds (three fits) | Evaluation seconds (three fits) |
|---|---:|---:|---:|
| current | 66675 | 95.012 | 5.555 |
| gru | 155523 | 111.769 | 6.245 |
| attention | 66676 | 112.868 | 6.345 |
| gated_delta | 66675 | 113.183 | 6.850 |
| kalman | 66675 | 140.746 | 6.955 |
| innovation_kalman | 66675 | 143.452 | 7.514 |
| carry | 166434 | 269.290 | 10.281 |

## Seen services

| Method | Macro accuracy (%) | Micro NLL | Brier | Revision accuracy (%) | Revision queries per fit |
|---|---:|---:|---:|---:|---:|
| current | 57.5178 | 0.9553 | 0.4909 | 38.0360 | 241 |
| gru | 57.5263 | 0.9239 | 0.4831 | 32.9184 | 241 |
| attention | 59.3263 | 0.9090 | 0.4748 | 31.3970 | 241 |
| gated_delta | 71.8206 | 0.7369 | 0.3679 | 44.1217 | 241 |
| kalman | 71.2958 | 0.7229 | 0.3622 | 37.8976 | 241 |
| innovation_kalman | 71.6520 | 0.7279 | 0.3617 | 38.1743 | 241 |
| carry | 59.4444 | 0.8969 | 0.4663 | 36.7911 | 241 |
| Reference: none | 33.3333 | not probabilistic | not probabilistic | 0.0000 | 241 |
| Reference: literal | 54.7453 | not probabilistic | not probabilistic | 49.7925 | 241 |

## Unseen services

| Method | Macro accuracy (%) | Micro NLL | Brier | Revision accuracy (%) | Revision queries per fit |
|---|---:|---:|---:|---:|---:|
| current | 47.9064 | 1.1770 | 0.5744 | 38.4236 | 203 |
| gru | 46.3174 | 1.2127 | 0.5834 | 38.2594 | 203 |
| attention | 46.9304 | 1.1458 | 0.5624 | 36.4532 | 203 |
| gated_delta | 52.4352 | 1.1587 | 0.5541 | 40.8867 | 203 |
| kalman | 52.1559 | 1.1652 | 0.5401 | 42.0361 | 203 |
| innovation_kalman | 53.9841 | 1.0700 | 0.5114 | 39.5731 | 203 |
| carry | 48.0983 | 1.0753 | 0.4918 | 34.6470 | 203 |
| Reference: none | 33.3333 | not probabilistic | not probabilistic | 0.0000 | 203 |
| Reference: literal | 65.7642 | not probabilistic | not probabilistic | 79.8030 | 203 |

Evaluation includes batch assembly and prediction storage, excludes encoder. Carry replays a prefix for each question; no incremental-dictionary speed claim.

All fit points and query/bin counts are retained in summary.json. Accuracy thresholds use exact count ratios. Micro NLL uses saved probabilities without flooring. Missing bins cannot satisfy their criteria. Raw text, weights and individual predictions stay local.

The reporter verifies labels against the hash-bound feature packet. Parser semantics, frozen encoder and neural forecasts are source/test bound, not independently rerun.
Literal-reference decisions and innovation diagnostics are source-bound; their generation is not replayed. Reference accuracies are independently recomputed from saved choices. Logical state payloads in summary.json exclude autograd and temporary copies; no peak-memory advantage is established.
