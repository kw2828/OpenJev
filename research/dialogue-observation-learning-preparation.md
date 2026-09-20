# Observation learning: complete input and workload preparation

Prospective preparation protocol, 20 September 2026. The
[six-cell qualification](dialogue-finetune-qualification-results.md) established
the tested cross-device gradient path. It does not establish worst-case memory
or a complete training allocation. This phase constructs all matched inputs and
cost metadata before any scientific fitting.

## Fixed comparison

Use a full factorial: frozen/trainable MiniLM crossed with original/number-normalized
lexical observations, three paired seeds 6901, 6902 and 6903. Keep the normalized
scalar memory mechanism fixed. There will be twelve fresh fits, each using
20 epochs, microbatch one dialogue and effective batch 32 distinct dialogues.
This is 1,280 updates per fit, 15,360 total, and 484,080 training dialogue visits.
Retain the existing three-stratum training weights. Accumulate weighted endpoint
loss sums divided by the total scored endpoints in that effective batch, then
clip and step once. The final one-dialogue batch uses its actual endpoint count.
Do not substitute equal-dialogue weighting or reuse trained historical weights.

The primary future comparison is trainable+numbers versus frozen+numbers.
The original lexical arms expose the normalization effect and its interaction
with encoder learning. All arms remain mandatory and all final fits must be
reported. This tests representation learning, not a new recurrent architecture.

## Actor and target boundaries

Authenticate the existing MiniLM checkpoint, feature packet, public TRAIN/DEV
streams, supplied schema catalog and original lexical cache. Preserve the exact
2,017 training and 2,363 development dialogues, all 51,741/62,329 scored endpoints,
and complete public USER streams. Preserve original encoder text and candidate
IDs. Every supplied question processes every public USER exchange, including
unscored ones. Only padding may pause state updates; no gold state or target
availability enters an actor observation.

Save actors and targets in separate files. Actors contain token IDs, public
identity maps and lexical offsets. Targets contain endpoint labels, transition
bins, supplied seen/unseen metadata and deterministic literal-register references.
Validate target bins against consecutive annotated values independently of the
actor stream. Gold values are used only by the loss and evaluator.

The numerical lexical control converts whole-token English cardinals zero
through ninety-nine and corresponding canonical digits in matching views only.
Both public utterances and ordinary candidate values use the same rule.
Hyphenated and space-separated tens-plus-unit forms are supported. Unsupported
maximal numeric compounds, ordinals, leading zeros and ambiguous compositions
remain unchanged. Number aliases retain distinct candidate IDs and therefore
tie ambiguously under the existing unique-longest-match rule. Apply this rule
to all six match/register-derived observations. Keep reserved-state and Boolean
cue features unchanged, with cues read from original public text.

This is an intentionally limited lexical control. It does not interpret intent,
negation, relevance, cross-service reference or sign/decimal semantics. Incidental
numbers may still produce incorrect matches. The original and normalized
deterministic literal-carry references will both be reported.

## Complete work inventory

Tokenize without truncation using the pinned local tokenizer. Preserve the
qualified 254-content-token chunking, batch size 32 and special-token pooling
contract. Produce per-dialogue work profiles for the entire training and
evaluation cohorts, sums for one complete visit, and maxima for token lengths,
chunk counts, padded attention work, sequential question updates and candidate
positions. Generate paired epoch orders without reading targets and enumerate
all effective batches, including the largest complete batches under each work
measure. These counts are not elapsed-time or memory predictions.

The preparation has a 300-second wall cap, 8-GiB process-RSS cap and 512-MiB
output cap. Load no neural weights and execute no neural forward. Source,
test and protocol hashes are fixed before execution; use an exclusive directory
and retain any failed attempt. No retry, shrinking or overwritten receipt.

## What follows preparation

Freeze a separate cost-only qualification for the effective-32 loss route and
metadata-selected worst single-dialogue and full-batch workloads. Include
fresh encoding, backward, recurrent state, gradient accumulation, validation,
synchronization, evaluation and checkpoint I/O. Only afterward specify the
complete twelve-fit allocation and hard execution cap, with explicit headroom.
The three earlier percentile timings cannot automatically admit this study.

The planned development continuation rule requires the primary to gain at
least 1 percentage point of unseen three-stratum macro accuracy, improve in at
least two paired seeds, and avoid worse unseen NLL and Brier. It may lose at
most 1 point of seen macro accuracy and increase assigned-retention error by at
most 0.5 points on each seen/unseen panel. All conditions must hold; these are
practical margins, not a statistical power calculation. Freeze the exact
reporter and all measures before fitting. Show every arm/seed and all transition,
service and TRUE/FALSE/DONTCARE supports even when a subgroup is absent.

Official DEV has already been exposed repeatedly and remains development
evidence. Official TEST stays sealed. A pass would justify a stronger baseline
and separate untouched confirmation, not biological-learning, calibrated-output,
world-model or ICLR-readiness claims.

The public source is the [Schema-Guided Dialogue dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/tree/e852981ae34990f4358979625854259302feaa78),
licensed CC BY-SA 4.0. Keep reversible token inputs, raw text, labels and
individual predictions local. Publish code, aggregate work metadata and receipts.
