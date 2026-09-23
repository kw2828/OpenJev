# Sampled score-forecast evidence

All original phases completed: 90 full collection paths, twelve final fits,
and an independent saved-output audit. The fixed scientific continuation rule
**failed 41/45**. The four failed conditions compare the proposed residual GRU
with the ordinary direct GRU, which performs better in both settings.

- `precollection-freeze-01.json`: 122 held scientific source hashes and qualification inputs, recorded before collection.
- `seed-review-01.json`: scoped review of every declared seed before run.
- `selection-runtime-01/`: identical deterministic choices on both installed runtimes for all 2,188 lengths and all 54 selection seeds.
- `collection-engineering-01/`, `collection-qualification-01/`: 37 new fabricated checks plus original model/data qualifications.
- `fit-audit-engineering-01/`, `fit-audit-qualification-01/`: twelve new fabricated boundary checks plus unchanged original qualifications.
- `collection-plan-01.json`, `collection-01/`, `collection-supervisor-01.*`: all 90 paths, complete public histories and full VALID scores. TRAIN labels include explicit availability masks and fixed sampled windows.
- `training-plan-01.json`, `training-01/`, `training-supervisor-01.*`: all twelve final checkpoints, selected training arrays and weights, all validation forecasts, training events and metrics.
- `audit-01/`, `audit-supervisor-01.*`: independent reconstruction of selected bytes, census validation windows, forecasts, metrics and all 45 conditions. Zero new model, native or optimizer calls.
- `figure-01/`: all fits and family means in PNG/SVG, 624 metric rows, 45 conditions, fit metadata and rendering receipt.

There are 18 independent TRAIN cases and 12 independent VALID cases, each with
three collector paths. Both sensing settings occur in TRAIN. Forecasts on saved
collector paths do not establish autonomous control, true action regret,
deployment savings or a novel architecture. The earlier failed collection was
not resumed or fitted.

The GitHub release contains all original artifacts and inherited evidence with
an exact member manifest. Native runtime and large original upstream weights
remain external dependencies listed in that manifest. Historical absolute paths
are preserved as evidence, not rewritten as portable rerun instructions.

[Results](../../research/otto-sampled-forecast-results.md) ·
[Protocol](../../research/otto-sampled-forecast-protocol.md) ·
[Complete archive](https://github.com/kw2828/OpenJev/releases/tag/otto-sampled-forecast-v1).
