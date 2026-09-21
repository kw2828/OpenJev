# Saved probability diagnostic

The [results](../../research/dialogue-probability-diagnostic-results.md) describe
all twelve completed observation-learning fits. No new model was run. The
original seven-condition scientific comparison still fails, with six passed.

- [Protocol](../../research/dialogue-probability-diagnostic-protocol.md) and
  [plan](plan.json), published in commit `5ebada2` before diagnosis.
- [Qualification](qualification-01/receipt.json): 47 synthetic tests and lint
  passed, with pre-freeze repairs retained.
- [Independent source review](source-review-01.json): required authentication
  bindings repaired before qualification.
- [Complete summary](run-01/summary.json), [execution receipt](run-01/receipt.json)
  and [actual exit](actual-exit-01.json): all twelve fits, 6.465 seconds, zero
  model calls, original baseline reconstructed.
- [Independent numerical audit](independent-audit-01/result-01/receipt.json)
  and [actual exit](independent-audit-01/actual-exit.json): 27,320 scalar checks
  agree across every fit/group, temperature cell and paired decomposition.
- [Figure](figure-01/probability-sensitivity.png),
  [plotted values](figure-01/plotted-values.json) and
  [reproducible plot source](figure-01/plot.py): full temperature grid for all
  fits, no chosen temperature.

The summary contains aggregate counts and scores, without dialogue text or
individual saved predictions. Reproduction needs the locally retained V2 run
and frozen dependencies. Use a new output directory; existing outputs cannot
be overwritten:

```sh
.venv/bin/python scripts/diagnose_dialogue_probabilities.py \
  --plan output/dialogue-probability-diagnostic-v1/plan.json \
  --plan-sha256 b251d8d63713f2908f79fc327b90aa3e52c1e5108c8fc44c07f32b90bff04d36 \
  --out output/dialogue-probability-diagnostic-v1/reproduction
```

This is exposed-DEV diagnosis. The temperature curves are descriptive and do
not fit a calibrator, change model choices, alter actor state, or revise the
original study's failed continuation decision.
