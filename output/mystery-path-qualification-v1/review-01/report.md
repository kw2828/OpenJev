# Mystery Path public-memory qualification

**FAIL: 16/17 checks passed.** Decision: `CLOSE_THIS_RECIPE`.

Completed saved-output qualification arithmetic; no model, environment, or RNG calls. This is not an architecture result, a pixels-only benchmark score, or proof of novelty.

All 64 layouts, four paired priority orders, and five controllers are retained.

Distinct saved layout hashes: 64/64; coincident generated layouts are retained.

| Controller | Successes / 256 | Actions / episode | Falls / episode | Whole ms / step |
|---|---:|---:|---:|---:|
| Full public map | 195 | 79.652 | 8.820 | 0.165 |
| Last 16 transitions | 58 | 102.094 | 20.941 | 0.189 |
| Last 32 transitions | 76 | 95.375 | 16.605 | 0.222 |
| Erase on failure | 34 | 112.227 | 29.859 | 0.150 |
| Known-route reference | 256 | 13.266 | 0.000 | 0.218 |

Actions and falls include failed episodes. Whole ms/step is total episode wall time divided by all executed environment steps.

| Full map versus | Both succeed | Full only | Comparator only | Neither |
|---|---:|---:|---:|---:|
| Last 16 transitions | 58 | 137 | 0 | 61 |
| Last 32 transitions | 76 | 119 | 0 | 61 |
| Erase on failure | 34 | 161 | 0 | 61 |
| Known-route reference | 195 | 0 | 61 | 0 |

| Controller | Peak serialized state bytes | Peak retained Python bytes |
|---|---:|---:|
| Full public map | 219 | 3905 |
| Last 16 transitions | 641 | 5875 |
| Last 32 transitions | 1213 | 9715 |
| Erase on failure | 148 | 2621 |
| Known-route reference | 107 | 1760 |

| Controller | Reset s | Environment s | Act s | Parse s | Observe s | Storage s | Evaluator s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full public map | 0.0785 | 0.5447 | 0.1579 | 1.0907 | 0.0467 | 0.9838 | 0.4609 |
| Last 16 transitions | 0.0881 | 0.7056 | 0.1567 | 1.3993 | 0.0533 | 1.2361 | 1.3060 |
| Last 32 transitions | 0.0895 | 0.6589 | 0.2379 | 1.3081 | 0.0508 | 1.1790 | 1.8972 |
| Erase on failure | 0.1026 | 0.7664 | 0.0619 | 1.5283 | 0.0651 | 1.3539 | 0.4239 |
| Known-route reference | 0.0890 | 0.0921 | 0.0433 | 0.1960 | 0.0002 | 0.2481 | 0.0712 |

| Exact check | Count | Required | Result |
|---|---:|---:|---|
| reference_95_percent | 256/256 | >= 244/256 | PASS |
| full_80_percent | 195/256 | >= 205/256 | FAIL |
| full_minus_last16_15_percentage_points_pooled | 137/256 | >= 39/256 | PASS |
| full_minus_last16_5_percentage_points_order0 | 33/64 | >= 4/64 | PASS |
| full_minus_last16_5_percentage_points_order1 | 33/64 | >= 4/64 | PASS |
| full_minus_last16_5_percentage_points_order2 | 36/64 | >= 4/64 | PASS |
| full_minus_last16_5_percentage_points_order3 | 35/64 | >= 4/64 | PASS |
| full_minus_last32_15_percentage_points_pooled | 119/256 | >= 39/256 | PASS |
| full_minus_last32_5_percentage_points_order0 | 29/64 | >= 4/64 | PASS |
| full_minus_last32_5_percentage_points_order1 | 30/64 | >= 4/64 | PASS |
| full_minus_last32_5_percentage_points_order2 | 31/64 | >= 4/64 | PASS |
| full_minus_last32_5_percentage_points_order3 | 29/64 | >= 4/64 | PASS |
| full_minus_erase_on_failure_15_percentage_points_pooled | 161/256 | >= 39/256 | PASS |
| full_minus_erase_on_failure_5_percentage_points_order0 | 39/64 | >= 4/64 | PASS |
| full_minus_erase_on_failure_5_percentage_points_order1 | 38/64 | >= 4/64 | PASS |
| full_minus_erase_on_failure_5_percentage_points_order2 | 42/64 | >= 4/64 | PASS |
| full_minus_erase_on_failure_5_percentage_points_order3 | 42/64 | >= 4/64 | PASS |

Per-order paired tables, all original records, and separate timing components are retained in [summary.json](summary.json).

- The four priority orders reuse 64 layouts; 256 layout/order pairs are not 256 independent layouts.
- All public policies share an explicit public-observation parser; the reference alone receives the route.
- Whole-episode time includes reset through NPZ writing/hashing, but excludes the outer JSONL append.
- Per-step wall time amortizes whole-episode work; it is not isolated policy latency.
- State figures are recorded retained-state estimates and serialized payload sizes, not process peak RSS.
- A failed gate closes this recipe without shrinking windows, replacing layouts, or changing thresholds.
- Prior Reacher and Pendulum failures remain unchanged; qualification cannot establish biological superiority.
- The completed audit replays the same official environment and checks the parser against native fields; it is not an alternative simulator or independent policy implementation.
