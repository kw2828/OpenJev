# Observation learning: twelve-fit scientific comparison running

The corrected preparation completed for **4,380 dialogues and 114,070 scored
endpoints**. Independent saved-input auditing passed. The subsequent
[full-batch cost pilot](dialogue-observation-learning-cost-results.md) also passed
all 40 cells and 52 updates in 75.306 seconds, with an independent audit. These
steps prepare the matched encoder-learning comparison; they produce no accuracy
or architecture result.

The planned factorial uses frozen/trainable MiniLM and original/number-normalized
lexical observations under the same autonomous scalar memory. All twelve fresh
fits use three paired seeds, 20 epochs and effective batch 32. The primary must
beat the inexpensive number-normalized frozen baseline. The
[prospective preparation and scientific conditions](dialogue-observation-learning-preparation.md)
remain unchanged.

## Preparation evidence

| Scope | Training | Development |
| --- | ---: | ---: |
| Dialogues | 2,017 | 2,363 |
| Scored endpoints | 51,741 | 62,329 |
| Complete public USER exchanges | 20,600 | 23,810 |
| Public question updates per cohort visit | 91,856 | 117,386 |
| Encoder batches per cohort visit | 3,444 | 4,121 |
| Content tokens per cohort visit | 2,018,799 | 2,489,730 |
| Longest content text, tokens | 93 | 102 |

The complete dataset contains 44,022 unique tokenized texts. Per-cohort token
work includes repeated schema/context encoding across different dialogues;
it is not the unique-token count or measured model latency. No text requires
the qualified 254-token chunk split. Every original lexical position matches
the historical cache exactly. The numeric control changes only the six
match/register features; all four reserved/Boolean features remain unchanged.

The sole corrected preparation recorded **14.350 seconds** and **706,854,912
bytes** peak process RSS. Its 13 payload files occupy **238,671,492 bytes**.
It loaded no neural weights and made zero neural calls. Session **21472 exited
0**; the parent recorded **14.737 seconds**, no timeout and process group
**10020 absent**. The shared CPU window was released afterward.

- [Complete plan](../output/dialogue-observation-learning-v1/preparation-02/plan.json):
  `4c5b2ddead9626e3c4f90819cb50d829f3894ee1fa1bae4adfe249c0ac178c8e`.
- [Completion receipt](../output/dialogue-observation-learning-v1/preparation-02/completed.json):
  `d1461a1ea64b23338b2112d581479798c6618c8ccebfbde24ce131059474cf83`.
- [Independent audit](../output/dialogue-observation-learning-v1/preparation-audit-02/receipt.json):
  `4fefa46325c8c8e4a74a6f14dcf8a7961f13c875faeb14b3f03ee35169c2c23c`.

The audit authenticated all 39 sources, every actor/target join, 20,007,650
lexical positions, all token-derived workload profiles, 60 seeded epoch orders
and 3,840 effective batches shared across the four arms. It recomputed maxima
and original training-only loss weights. It did not rerun tokenization or
lexical extraction and did not score model or literal-reference accuracy.

## Preserved first failure

The first preparation, frozen in `2710899`, stopped after preparing 2,017 TRAIN
and two DEV records. A new workload checker had incorrectly required bare IDs
to be unique across splits. Independent inspection found 298 repeated IDs,
each naming different public content across TRAIN and DEV. The actual failure
was DEV `10_00001`.

That attempt recorded **7.816 seconds** and **584,794,112 bytes** peak RSS, with
zero neural calls. Its 192,472,519 partial bytes remain local. Session **1041
exited 1** and process group **9322 was absent**. It was not resumed. The
[explicit additive correction](dialogue-observation-learning-preparation-correction.md)
was published in `cd55ca8` before the separate V2 execution; all original
sources, tests and failure evidence remain intact. The correction keeps
`(split, dialogue_id)` uniqueness and changes no cohort, target, observation or
scientific condition. [Failure manifest](../output/dialogue-observation-learning-v1/failed-preparation-publication-01/manifest.json).

## Scientific execution

The twelve fits require 15,360 optimizer updates, 484,080 training dialogue
visits and 12,417,840 supervised endpoint presentations. The completed cost pilot
tested three largest full batches, three training extremes, a one-dialogue
tail and three evaluation extremes under all four arms. It used synthetic
targets and included backward, effective-batch weighting and checkpoint I/O.
Measured work supports a prospective eight-hour allocation with explicit
headroom. Runner, metric and saved-results checks are complete. The independent
source review binds all 53 source files. The metadata freeze completed in
1.832 seconds with zero model calls; its parent exited 0 with process group
14697 absent. Independent saved-only freeze auditing also passed.

- [Frozen scientific plan](../output/dialogue-observation-learning-v1/scientific-freeze-01/plan.json):
  `acb79b4600c66966762895d28eb2dc1d2be15c761d677c5e87c5750dde47f237`.
- [Freeze completion](../output/dialogue-observation-learning-v1/scientific-freeze-01/completed.json):
  `c3a53a94dd4a36e677b36d4e358bb779abd6419711941a792e50ce48feab36c3`.
- [Scientific source review](../output/dialogue-observation-learning-v1/scientific-source-review-01/receipt.json):
  `45921e885a9cd1cf67036e5d2dfd065e5a3e3e767777d4c62210e12a599fd815`.
- [Independent freeze review](../output/dialogue-observation-learning-v1/scientific-freeze-review-01/receipt.json):
  `c9fd029c80fee2c7f7d790e75fb612b5f529e0ea40b5c189d08cfa5605fb3d18`.

The frozen plan fixes all twelve fits, row order, scorer, seven primary
conditions and independent process watchdog. The eight-hour allocation is a
hard stop, not a promise of completion. A failure preserves the partial attempt;
it does not permit a replacement seed, resume or cap extension.

The scientific worker started on September 20, 2026 after the frozen plan was
published in `b5d57a9`. Session **90741**, process group **14919**, is running
with one CPU thread and MPS. **Both frozen-encoder controls for seed 6901 have
completed**, and `trainable_original-6901` is running. Each completed fit recorded
all **1,280 optimizer updates, 40,340 training dialogue visits and 62,329 final
development predictions**. Their recorded operation totals match the frozen
plan, and both encoders remained unchanged. The first trainable fit's initial
update recorded nonzero embedding and attention gradients. This establishes
execution only. No fit has been selected or scored for task quality; the
twelve-fit study remains incomplete.

| Completed fit | Total recorded seconds | Final DEV predictions |
| --- | ---: | ---: |
| frozen_original-6901 | 915.03 | 62,329 |
| frozen_numbers-6901 | 924.56 | 62,329 |

[First-fit completion](../output/dialogue-observation-learning-v1/scientific-run-01/frozen_original-6901/completed.json)
has SHA-256 `737309a4092aed7f5a2f4d43d17f106103dbca849eac22bcd36d001bf66f2e61`.
The [second-fit completion](../output/dialogue-observation-learning-v1/scientific-run-01/frozen_numbers-6901/completed.json)
has SHA-256 `d591f17465ddd6df7a9ff6970dfbd33bc85b66abe26b79450ca1be6fc695ed14`.
The [current progress snapshot](../output/dialogue-observation-learning-v1/scientific-progress-02.json)
records the live process and initial gradient witnesses separately from final
scientific validity. The [first snapshot](../output/dialogue-observation-learning-v1/scientific-progress-01.json)
is retained.

- [Actual supervisor launch](../output/dialogue-observation-learning-v1/scientific-process-01.launch.json):
  `87034aabfbf9862a54b34076d37794a1578b9ad254580b13bf742fa43952b1fd`.
- [Worker start](../output/dialogue-observation-learning-v1/scientific-run-01/started.json):
  `e76773d8f7de3588b61cca50d8dc7aa73582ff8969a46893ab2c97db2e774afd`.

The shared compute slot was released before startup. Other explicitly recorded
CPU-only work may overlap; it cannot support clean standalone latency claims.
The separately owned BLACKOUT [brief qualification](../output/dialogue-observation-learning-v1/external-overlap-01.json)
and [learning-study startup](../output/dialogue-observation-learning-v1/external-overlap-02-start.json)
are recorded as owner-reported CPU-only work, with independent process-group
liveness checks. No partial quality results from that study were opened.
The BLACKOUT worker subsequently [released its CPU slot](../output/dialogue-observation-learning-v1/external-overlap-02-terminal.json):
the owner reported exit 0 after 1,553.723 seconds, and its process group was
independently confirmed absent while OpenJev remained live. Its quality audit
was still pending; the release supplies execution provenance only.
All training journals, partial outputs and terminal evidence will be preserved.
Scoring waits for the entire twelve-fit attempt and its actual terminal state.

An [independent saved-results checker](../output/dialogue-observation-learning-v1/result-audit-01/README.md)
is ready, with nine synthetic tests passed. It independently recomputes the
primary scores and all seven conditions. The [figure generator](../output/dialogue-observation-learning-v1/figure-01/README.md)
has also passed synthetic presentation checks. Neither has read actual study
predictions. Both were prepared without changing the frozen scientific sources.
Their short CPU-only checks overlapped training and are recorded separately.

Official DEV is exposed development evidence. Official TEST remains sealed.
An eventual representation-learning gain would strengthen the baseline for a
later recurrent or connectome comparison, not establish such a contribution.

The [artifact publication note](../output/dialogue-observation-learning-v1/README.md)
distinguishes published metadata from local reversible tokens, labels, lexical
arrays and future individual predictions. The underlying Schema-Guided Dialogue
data retain CC BY-SA 4.0 terms.
