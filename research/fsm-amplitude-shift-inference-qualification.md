# Amplitude-shift inference qualification

**Adapters and independent forecast arithmetic qualified on fabricated inputs.
The empirical comparison is still unregistered and unrun.** One retained
execution passed **98 tests plus Ruff**, with exit 0 and all 21 recorded source,
dependency and configuration identities unchanged. This is engineering evidence,
not evidence of improved forecasting or biological learning.

The new [deployment adapter](../src/openjev/research/fsm_shift_inference.py)
exposes the same physical-unit C100/H128 request for frozen VARX, affine/tanh
residual, author BLA and author NL-LFR models. It preserves the existing numerical
implementations and their original normalization. Residual weights have gradients
disabled. Each request starts from its own observed context, accepts no targets,
and returns owned predictions, forecast-start state, final state and finite JSON
diagnostics. The independent [NumPy replay](../scripts/fsm_shift_audit_math.py)
implements lag-model arithmetic separately and reuses the previously qualified
independent author recurrences. It does not import the producer models.

The retained checks cover:

- Native forecast and state parity, plus eight independent replay comparisons.
- Chronological lag and current-input alignment, physical normalization and
  separate output-only versus recurrent residual propagation.
- Request, batch and buffer isolation; future inputs cannot change conditioning
  state or earlier forecast outputs.
- Native 16/64-step author policy parity, bounded-solver status and work counts.
- Numeric deployment storage, finite output and diagnostic validation, and
  explicit overflow without clipping or replacement predictions.

Fixtures include VARX orders 32/64/96, a 28-state linear author model, small
nonlinear arithmetic examples and exact production-shape storage checks. They
do **not** establish worst-case full-geometry nonlinear timing or memory limits.
No saved checkpoint or measurement archive was decoded. No training, empirical
forecasting, timing benchmark or normalizer fitting occurred.

## Retained evidence

The [qualification receipt](../output/fsm-shift-engineering-v1/inference-qualification-01/receipt.json)
has SHA-256 `4c94085671ab3fe829ab023a972f423bcd3b7cf14857912d983ced6fa4115119`.
It retains exact commands, runtime versions, thread settings, elapsed process
times, logs, before/after source hashes and source snapshots. The
[original process observation](../output/fsm-shift-engineering-v1/inference-qualification-01/original-process-observation.json)
records the observed exit. The test log reports 98 passes in 0.70 seconds;
that duration is a test-suite observation, not inference latency.

| Qualified file | SHA-256 |
| --- | --- |
| Production adapter | `7fbde5b0c39353eced9ce78f5daef7a1ba10b700c7951a0aac3b6f12dc10c8ef` |
| Independent replay | `30f61937e457a82e0662a1f14aa4c95c992faef1567096cc85d922836dab87f3` |
| Adapter tests | `0d14cff9c7e2ed3c96326935bed36bdd319009ff3bc339650b90be542fa9bb81` |
| Independent hand fixtures | `964f939eaa0ecc009f4035e21d6aec9d8ad96b0047ccb99ee52e9a695d70f97b` |

The separately prepared [roster candidate](../output/fsm-shift-engineering-v1/model-roster-candidate.json)
lists 28 deployments, 18 families and 27 distinct checkpoint files. Its
[metadata receipt](../output/fsm-shift-engineering-v1/model-roster-preparation.json)
joins 79 file identities to closed audited inventories, without array or
checkpoint decoding. The reference remains the previously selected tanh
output-only family. Neither this roster nor the
[draft protocol](fsm-amplitude-shift-protocol.md) is an empirical registration.

## Next prerequisite

Qualify the complete evaluator and evidence auditor, including failure ledgers,
target-after-prediction extraction, independent scoring and confirmation gates.
Establish resource caps with fabricated full-geometry requests, then publish
source/model/runtime identities and the exact timing schedule before numerical
access. The previous [publication-copy limitation](fsm-author-publication-correction.md)
still applies. All official test members remain closed.
