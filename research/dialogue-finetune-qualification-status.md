# Observation learning qualification: execution status

The [protocol](dialogue-finetune-qualification-protocol.md), implementation and
synthetic tests were published in `b111e5d` before preparation. All 53 synthetic
tests passed; test-formatting lint issues were corrected and the eight affected
tests passed again. Independent source review found no remaining material
blocker. No task quality is measured in this qualification.

The single preparation completed in **6.033 seconds**, with **982,876,160 bytes**
peak process RSS and zero encoder calls. All **2,017 original TRAIN dialogues**
were prepared, with exact agreement against their historical lexical inputs.
There are **20,741 unique tokenized texts** across the cohort.

| Fixed workload rank | Dialogue | USER exchanges | Queries | Unique texts | Encoder batches per pass |
| --- | --- | ---: | ---: | ---: | ---: |
| 10th percentile, index 201 | 25_00108 | 4 | 2 | 18 | 1 |
| 50th percentile, index 1008 | 116_00015 | 11 | 6 | 50 | 2 |
| 90th percentile, index 1814 | 85_00039 | 13 | 9 | 80 | 3 |

These are ranks of a padded-attention workload proxy, not performance or elapsed
time percentiles. The [complete plan](../output/dialogue-finetune-qualification-v1/preparation-01/plan.json)
is SHA256 `f0e35e69f5c614e02a516edfa5bd049f3fda3beff48ed8c221b75907a0adee83`.
The [completion receipt](../output/dialogue-finetune-qualification-v1/preparation-01/completed.json)
is SHA256 `821275e9faf0f3e8ef72c95674da5b0866e1dd25f00cb3b88df968cd1b8ca748`.
The reversible token payload remains local; its identity and publication scope
are documented in the [artifact README](../output/dialogue-finetune-qualification-v1/README.md).

The parent supervisor observed exit **0** and process group **6014** absent in
**6.499 seconds**, including startup and shutdown. The single model pilot is
prepared but has not executed. It requires the shared MPS/CPU allocation and
must complete all six cells within 300 seconds. Preparation success is not
numerical qualification or an accuracy result.
