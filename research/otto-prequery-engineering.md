# Prequery supervision: engineering record

The scientific comparison has since completed: [results and original receipts](otto-prequery-calibration-results.md). The preparation record below describes what was established before collection.

This is preparation for the [frozen protocol](otto-prequery-calibration-protocol.md),
not a scientific result. The earlier [cross-query result](otto-cross-query-forecast-results.md)
remains closed. The new implementation exposes and supervises a forecast already
computed inside the existing correction model; it does not add a recurrent call,
a parameter or a teacher request.

## Source and data separation

The original model, projector, metrics and prior study remain unchanged. New
wrappers expose prequery predictions, add separate auxiliary target arrays and
provide two explicit objective branches. The original-loss branch never reads
auxiliary targets or weights. Auxiliary supervision uses all four query scores
because the model's correction centers all four coordinates. Predictions made
before a query cannot use that query's observed score.

The new 87 seeds cleared a saved exact-integer collision review over 3,184
historical source and metadata files. New proposal files were excluded from that
historical-use scan. The scope is recorded; this is not a global proof that no
other project has ever used these integers.

## Completed qualification

[Component qualification](../output/otto-prequery-calibration-v1/engineering-01/receipt.json)
passed 92 fabricated tests and static checks, with source hashes unchanged across
execution. The checks include causal forecast capture, original prediction and
carry equality, masking and short episodes, target isolation, all-four gradient
coverage, sampled-data invariants and collection routing. No new empirical arrays,
native environment or teacher was used by these checks.

The first metadata-only collection-plan invocation used a relative output path.
The CLI rejected it before writing a plan or doing scientific work. That failure
is preserved in `collection-plan-admission-01.json`; the corrected absolute-path
invocation passed and froze `collection-plan-01.json`. No source, seeds, caps or
scientific settings changed. This was not a failed scientific collection.

Training, metric, audit and synthetic capacity qualification are recorded
separately before any fresh scientific collection. Their completion and the full
source freeze must be established before the collector is started.

## Final precollection checks

The separate metric, saved-output audit, training and capacity fixtures passed
16, 20 and 10 cases respectively. Together with the 92 component cases, there
were **138 newly recorded passing test executions**, including inherited helper
tests. Aggregate receipt copies were not counted as additional tests.

The synthetic capacity run completed under its original 120-second supervisor
in 2.298837458 seconds. Its four maximum-length batch measurements sum to
1.047522083 seconds. The prospective formula projects 3,513.97154892 seconds,
below the fixed 8,100-second admission threshold. This is a planning estimate,
not measured scientific training time. Auxiliary cells backpropagated through
all 69 chunks; original-loss cells backpropagated through 36 and still paid for
all 69 forward chunks.

Before the first metadata-review invocation, source review found that its AST
checker attempted to read a named seed constant as a literal. The original draft
is preserved as `review-source-draft-01.py`, with the reason and hashes in
`review-source-correction-01.json`. Only that metadata checker changed; no
scientific source or setting changed, and it had not yet been invoked.

The final freeze binds **136 source files**. The independent metadata review
passed before collection started. It checks the exact source union, all original
qualification receipts, fresh seed identities, the four-cell configuration, the
55-condition rule, and the completed original synthetic supervisor. No empirical
array or teacher was used by the review.

| Record | SHA-256 |
|---|---|
| [Precollection freeze](../output/otto-prequery-calibration-v1/precollection-freeze-01.json) | `1880f1cf74facc4a32cfbefcf71a7ea34304c1637ea270d2155f2015a6b26416` |
| [Independent metadata review](../output/otto-prequery-calibration-v1/precollection-review-01/receipt.json) | `642ae6f72e55772dcb6c18e86ace9c3630b13270be3002e932fe16f3921c7208` |
| [Capacity receipt](../output/otto-prequery-calibration-v1/capacity-01/receipt.json) | `24194af667c07e10ee354faff6bc982c704fe50009e1ce256900bc935ed48de8` |
| [Original capacity parent](../output/otto-prequery-calibration-v1/capacity-supervisor-01.terminal.json) | `9e3a77ae5fffa98cb2d75d5405eae1b7724732643cb6c7834e7d5c2a300f3bcc` |
