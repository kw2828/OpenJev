# Observation learning: complete inputs ready, training pending

The corrected preparation completed for **4,380 dialogues and 114,070 scored
endpoints**. Independent saved-input auditing passed. This prepares the matched
encoder-learning comparison; it produces no accuracy or architecture result.

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

## Next execution

The twelve fits require 15,360 optimizer updates, 484,080 training dialogue
visits and 12,417,840 supervised endpoint presentations. A separate cost pilot
will test three largest full batches, three training extremes, a one-dialogue
tail and three evaluation extremes under all four arms. It will use synthetic
targets and include backward, effective-batch weighting and checkpoint I/O.
Full-study cost and a hard allocation remain to be established before training.

Official DEV is exposed development evidence. Official TEST remains sealed.
An eventual representation-learning gain would strengthen the baseline for a
later recurrent or connectome comparison, not establish such a contribution.

The [artifact publication note](../output/dialogue-observation-learning-v1/README.md)
distinguishes published metadata from local reversible tokens, labels, lexical
arrays and future individual predictions. The underlying Schema-Guided Dialogue
data retain CC BY-SA 4.0 terms.
