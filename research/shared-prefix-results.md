# Shared context makes multi-question scoring faster

**A shared-prefix prototype improved four-question request latency by 2.08x in the median paired comparison, while passing the fixed output checks on this synthetic workload.** Single-question caching was slower. This is a systems result using the existing Qwen model, not a new architecture, training result, calibration result, or production rollout.

![All four methods, all eighteen workload cells, median and p95 timings](../output/shared-prefix-v1/report-01/warm-latency.png)

## What was measured

On Apple M5 Max, the pinned Qwen3-4B-Instruct-2507 4-bit model scored 54 fixed English requests: 1/2/4 questions, short/long context, 2/4/12 candidates, and three requests per cell. Complete prompts ranged from 130 to 711 tokens. Each method made one allocator-cold call and five warm calls per request: **216 cold and 1,080 warm evaluations**. Model loading took 3.35 seconds; the complete execution took 253.91 seconds.

The four methods use identical full prompt tokens, candidate order, full-vocabulary readout and probability definitions. Batched suffixes share prefix computation but allocate their own KV storage. Timing includes tokenization, cache work, synchronization, readback and answer formatting. Request validation, artifact writing and comparison calculations are excluded.

## All workload cells

Times are median warm request milliseconds. Speed is the median of the three paired request ratios in each cell: each request's serial median divided by its shared-batch median. It need not equal the ratio of the table's pooled medians. Each cell contains fifteen warm timings per method.

| Questions | Context | Candidates | Serial | Batch full prompts | Shared + serial | Shared + batch | Shared-batch speed |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | short | 2 | 41.1 | 40.8 | 50.5 | 51.4 | 0.80x |
| 1 | short | 4 | 42.3 | 43.3 | 53.0 | 52.7 | 0.81x |
| 1 | short | 12 | 53.4 | 53.5 | 63.0 | 62.5 | 0.85x |
| 1 | long | 2 | 122.6 | 123.3 | 133.1 | 131.8 | 0.93x |
| 1 | long | 4 | 132.5 | 133.4 | 144.8 | 143.7 | 0.92x |
| 1 | long | 12 | 134.3 | 135.5 | 147.8 | 145.4 | 0.92x |
| 2 | short | 2 | 89.7 | 61.8 | 94.0 | 82.3 | 1.09x |
| 2 | short | 4 | 93.7 | 62.9 | 108.5 | 77.2 | 1.20x |
| 2 | short | 12 | 112.8 | 90.0 | 128.0 | 97.7 | 1.16x |
| 2 | long | 2 | 263.7 | 250.8 | 204.7 | 184.0 | 1.43x |
| 2 | long | 4 | 291.4 | 269.6 | 223.8 | 180.7 | 1.62x |
| 2 | long | 12 | 308.8 | 307.3 | 249.8 | 208.1 | 1.50x |
| 4 | short | 2 | 202.6 | 116.4 | 171.7 | 100.4 | 2.07x |
| 4 | short | 4 | 207.4 | 130.3 | 201.1 | 109.1 | 1.92x |
| 4 | short | 12 | 255.2 | 182.4 | 236.6 | 147.5 | 1.73x |
| 4 | long | 2 | 570.4 | 549.3 | 304.4 | 203.7 | 2.80x |
| 4 | long | 4 | 625.1 | 570.4 | 333.5 | 227.7 | 2.73x |
| 4 | long | 12 | 650.5 | 630.3 | 371.1 | 313.2 | 2.26x |

The shared-batch median paired ratio across all eighteen four-question requests is **2.08x**. For the longer-context four-question groups, it is **2.26-2.80x**. Full-prompt batching alone provides a smaller gain, so repeated context processing is a material cost here. For two short-context questions, full-prompt batching is faster than shared-prefix batching. A general implementation should retain the simple path for workloads where cache setup costs more than it saves.

Across four-question cases, maximum observed active MLX memory rose from **2,847.94 MiB** for serial scoring to **3,211.23 MiB** for shared batching. This is device allocator memory, including the resident model, not process RSS. The [machine-readable report](../output/shared-prefix-v1/report-01/summary.json) includes all p95 timings, initial allocator-cold timings, memory distributions and work counts.

## Output agreement and its limits

All **1,296 records** passed the prospective checks: no selected-ID changes, maximum absolute candidate-probability change at most 0.005, and candidate vocabulary-mass change at most 0.005. The largest measured changes across all methods were **5.15e-10** in candidate probability and **4.90e-10** in candidate mass. Candidate logits were not bit-identical: their maximum absolute difference was **0.75**.

**These are unusually easy, high-margin questions.** All 126 distinct baseline decisions assigned at least **0.9999999989** probability to their chosen label. Consequently, this test is weak evidence about numerical sensitivity near ties. Small score changes could affect uncertain decisions elsewhere. No accuracy, uncertainty calibration, cross-field consistency or general probability-equivalence claim follows. The production scorer remains unchanged; a separate varied-margin quality check is needed before a default rollout.

The independent saved-output auditor reconstructed candidate softmax, entropy and selected IDs from saved candidate logits, checked the exact record set, recomputed output differences and work counts, and verified bound files. Full-vocabulary logits were not retained, so total candidate mass is range-checked and compared, not independently reconstructed. The model was not rerun during reporting.

## Relationship to the supplied diagram

The [public Qwen implementation](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/tree/2af86848be75847ccb3553b0941cc51d6ef7e4e9) suggests shared context and parallel bounded decisions. Its architecture and latency claims are separate from this experiment. Our comparison uses OpenJev's existing zero-generation scorer rather than autoregressive JSON generation, and we did not copy its heuristic confidence handling. See the [source review](qwen-parallel-source-review.md).

Caching is useful engineering, but supplies no evidence for proprietary RLCD training, biological wiring, JEPA or recurrent world-model learning. The next model research decision still needs a task where better learned state or dynamics improves outcomes against strong controls.

## Reproduce and inspect

- [Frozen protocol](shared-prefix-protocol.md) and [machine-readable protocol](../output/shared-prefix-v1/protocol.json).
- [Requests and exact tokens](../output/shared-prefix-v1/requests.json), [all outputs](../output/shared-prefix-v1/run-01/records.jsonl), [runtime](../output/shared-prefix-v1/run-01/runtime.json) and [completion](../output/shared-prefix-v1/run-01/completed.json).
- [Independent audit receipt](../output/shared-prefix-v1/report-01/receipt.json).
- [Experiment implementation](../src/openjev/research/shared_prefix.py), [runner](../scripts/benchmark_shared_prefix.py), [auditor and plotter](../scripts/report_shared_prefix.py).

Fourteen experiment/runner tests and twenty-four independent report tests passed. The tiny mathematical fixture runs on CPU float32; the measured model runs on GPU with the original quantized weights. The initial preparation check requested uncached repository documentation; the local file filter was corrected before the protocol was frozen. [Preparation receipt](../output/shared-prefix-preparation-attempt-01.json). No weights were downloaded.

```sh
PYTHONPATH=src .venv/bin/python scripts/benchmark_shared_prefix.py prepare --output output/shared-prefix-reproduction
PYTHONPATH=src .venv/bin/python scripts/benchmark_shared_prefix.py run --output output/shared-prefix-reproduction
```

Use the installed language dependencies and the already cached pinned model. Reproduction creates fresh outputs; do not overwrite the published execution. These results are descriptive for this machine and workload, with no confidence interval or claim that the upstream 5.6-7x figure was reproduced.
