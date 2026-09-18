# Trained pin-policy native-cost preparation

Scope: a pre-outcome timing protocol and tested runner. No trained-policy cost
run has started and no speed result is available.

The bound quality-v2 process disappeared before evaluation. This original
cost protocol is preserved without modification and cannot use v3 weights.
The separately frozen [cost-v2 recovery](../chess-pin-trained-cost-v2/README.md)
retains this comparison and binds it to fresh quality-v3 checkpoints.

The frozen `protocol/plan.json` binds the runner, tests, full quality-source
signature, quality-plan hash, canonical panel metadata, input hashes and
pre-outcome freeze state. The original quality execution is
`runs/chess-pin-quality-v2/execution`. Its completed full replay audit is a
mandatory prerequisite. The cost runner refuses to read fitted outputs when
that audit is missing.

Run only after the quality primary and full audit complete:

```sh
.venv/bin/python scripts/chess_pin_trained_cost.py run \
  --plan evidence/chess-pin-trained-cost-v1/protocol/plan.json \
  --out runs/chess-pin-trained-cost-v1/execution
```

Then independently replay the complete timing panel and saved arithmetic:

```sh
.venv/bin/python scripts/chess_pin_trained_cost.py audit \
  --plan evidence/chess-pin-trained-cost-v1/protocol/plan.json \
  --execution runs/chess-pin-trained-cost-v1/execution \
  --out evidence/chess-pin-trained-cost-v1/audit
```

Both stages have a fixed 1,800-second cap. Keep all failed attempts and original
quality/numerical criteria. A completed audit can authenticate a failed
scientific result. Cost comparisons are descriptive and cannot turn that
result into a successful architecture claim.

The test record is `tests.json`; the complete rationale and coverage are in
[the research note](../../research/chess-pin-trained-cost.md).
