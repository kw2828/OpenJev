# Canonical matched pin-factor quality study v2

Interrupted before evaluation: the original process and tool session were
absent when checked on September 18, 2026. Five fits completed and a sixth
retained 559 updates, for 8,239 logged updates total. No quality result exists.
The [interruption review](interruption-review.json) preserves every partial
artifact and known cost. The termination cause and terminal wall time are
unknown. A separately frozen [v3 recovery](../chess-pin-quality-v3/README.md)
repeats the unchanged scientific comparison from fresh initializations.
See the [research note](../../research/chess-pin-quality-study.md) and
[frozen protocol](protocol/plan.json), SHA-256 `9f9eba11e6abaa9b15c085a0098edbe938e7db848a323a3646f7f9468aadfc48`.

The v1 attempt is [preserved separately](../chess-pin-quality-v1/failure-review.json):
770 logged updates, zero completed fits, zero quality predictions and a
320.788-second cost. It was stopped on confirmed integer-versus-string
source-game ID incompatibility, not an observation timeout or a quality result.
V2 preserves integer IDs including zero, checks historical metadata before
training and rejects mixed/Boolean IDs. The signature requires every other
scientific criterion, model, data source, update count and budget to match v1.
All final heads restart fresh; no partial weights are reused.

The 24 fits retain 36,864 updates. Every fit must finish before evaluation:
110,592 fresh position records and 3,226,149 full native/cached score comparisons.
All sixteen quality checks and the numerical gate are required. The fixed
execution/audit ceilings remain twelve/two hours. This is adaptive development,
not independent confirmation, game strength or architecture novelty.

Thirty-one tests pass; [test record](tests.json) covers constructed fixtures,
statistical analysis, shared mechanics and comparator selection. The [launch observation](launch.json) is not proof of ongoing liveness or
completion.
Primary artifacts are in `runs/chess-pin-quality-v2/execution`.

The original audit command below is retained for reference. This incomplete
attempt cannot be reported as a completed study:

```sh
.venv/bin/python scripts/chess_pin_quality_study_v2.py audit \
  --plan evidence/chess-pin-quality-v2/protocol/plan.json \
  --execution runs/chess-pin-quality-v2/execution \
  --out evidence/chess-pin-quality-v2/audit
```

Do not restart on an observation timeout. Preserve all failure receipts,
partial artifacts and original criteria; repairs require a new version.
