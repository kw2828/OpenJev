# Paired query-value pilot: incomplete at the time limit

**The single run reached its fixed 900-second deadline with 210 of 256 available label panels complete.** All 72 TRAIN trajectories finished, but the paired-label collection did not. The scientific signal rule was not evaluated. This attempt supplies no conclusion about whether local query value is predictable, and admits no model fitting.

The original supervisor terminated the worker, reaped it and confirmed that its process group was absent. Its elapsed time was **901.011802 seconds**, including termination and cleanup after the 900-second deadline. The worker preserved its partial records, pending operation and failure receipt. There was no extension, restart, replacement seed or resumed label generation.

## What this experiment tests

The [previous learned gates](otto-query-gate-learning-results.md) queried on every decision. Their disagreement target did not measure whether paying for a neural query would help. The [prospective protocol](otto-query-advantage-protocol.md) instead asks whether one neural-selected action has a repeatable local benefit over the analytic action.

It fixes two sensing settings, twelve originating cases per setting and three actual query schedules per case. Five predeclared anchor slots per episode retain public beliefs and query history. At each available anchor, sixteen paired posterior-source samples compare the two first actions, followed by the same analytic policy for at most 32 total moves. Identical first actions reuse one physical continuation and have exactly zero advantage. Native hidden truth is not a label input.

The proposed analysis compares two independent eight-sample halves, centers within settings, retains different-action controls, and uses 2,000 case-cluster bootstrap draws. All eleven conditions, including technical completion, must pass before a separately frozen fitting study. Neither the whole-sample signal reduction nor the independent scientific audit ran here. The success-only renderer was also not invoked.

## Complete accounting of the attempt

| Item | Recorded work |
|---|---:|
| Full TRAIN trajectories | 72 / 72 |
| Native moves and final public updates | 1,963 |
| Fixed anchor slots | 360 |
| Available, validated anchors saved before labeling | 256 |
| Slots unavailable because the source was already found | 104 |
| Completed label panels | 210 / 256 |
| Interrupted panel | 1 |
| Available panels not started | 45 |
| Completed TensorFlow scores | 1,051 |
| Of those, separate annotation scores | 87 |
| Completed posterior source draws | 3,373 |
| Completed physical continuation records | 4,185 |
| Recorded continuation moves and public updates | 79,123 |
| Training updates / evaluation episodes | 0 / 0 |

The interruption occurred in panel anchor 289, episode `train:shift:19200008:never`, with one `teacher_choose` operation pending. Counts include work in that unfinished panel; they must not be interpreted as a complete label set. The 4,186 constructed branch snapshots and 4,185 completed records preserve the interrupted branch rather than rounding it into a completed result. All partial payloads remain in the release.

The worker recorded 658,091 sampler events and peak resident memory of 738,656,256 bytes. The deadline, rather than the memory or output ceiling, ended the attempt. The earlier failed import-only qualification is also retained: supplying the repository source path corrected the test invocation without modifying the native runtime or scientific sources. Subsequent qualification passed **42 fabricated tests** across the pair adapter/reducer, producer and auditor, with static checks. These are engineering checks, not evidence of learning efficacy.

## Runtime diagnosis and next step

The [independent failure review](../output/otto-query-advantage-v1/failure-review-01.json) verified all fifteen failed payload hashes and all 125 frozen sources, then checked operation counts and timing metadata without decoding outcomes or arrays. Across the 210 completed, non-overlapping panel intervals, **452.682515 of 692.565681 seconds (65.36%)** fell inside the journal timer. That timer includes serialization, write guards and file handling; it does not isolate disk sync time. The remaining 239.883166 seconds include analytic work, checks and other overhead. These totals exclude the interrupted panel and other phases and are not a prediction of an achievable speedup.

Source review identified substantial instrumentation work: every event is serialized twice, appended through a separately opened stream, flushed and synced, with repeated output-directory size scans. A typical continuing nonterminal move triggers eight event syncs and about twenty directory scans, in addition to analytic teacher work. Source inspection alone does not identify how much time each of those operations consumed.

The next engineering step is a separately qualified logger using fabricated traces: encode each event once, keep streams open, track exact reserved bytes incrementally, and reconcile sizes at declared boundaries while preserving the per-event durability contract. Qualification must cover byte-identical output, write/sync/close failures, and rejection before exceeding caps. A buffered commit scheme would change crash guarantees and needs its own specification.

This proposal does not reopen the failed scientific allocation. The original incomplete records and frozen sources stay unchanged. No threshold, sample count, horizon or architecture has been selected from partial outcomes. Any further scientific study needs a new prospective protocol and a justified resource allocation; this pilot has not passed its continuation rule.

The metadata review also found a separate qualification gap: the frozen success auditor expects an absolute producer script path, while the actual command used its equivalent repository-relative path. It would reject that invocation even after successful collection. It was not invoked here, and this mismatch did not cause the timeout. Any future auditor must qualify the exact launch form in a separately reviewed revision; the original frozen auditor remains unchanged.

## Evidence and limits

[Evidence release](https://github.com/kw2828/OpenJev/releases/tag/otto-query-advantage-v1) · [Frozen plan](../output/otto-query-advantage-v1/plan-01.json) · [Failure receipt](../output/otto-query-advantage-v1/run-01/receipt.json) · [Original terminal](../output/otto-query-advantage-v1/supervision-01.terminal.json) · [Engineering qualification](../output/otto-query-advantage-v1/engineering-01/receipt.json).

| Record | SHA256 |
|---|---|
| Frozen plan | `d3ccfc6990297491f79253640cecee41cab2f8d38c1a8bfbb226f67721139ac6` |
| Worker failure receipt | `a74b825c5fa67d2fb32074a23b068b27d98deaf64cd7cb7246674ba7b1c0b963` |
| Original supervisor terminal | `be01ecc100f517dde0cb88fd89b520461d5a8a382e8912def2240a1db7fa7dc4` |
| Metadata-only failure review | `10a3d7e3929cf10749491cf9ed2873a2773943234c3d228e1bd9640aad2fc0a3` |

This is a resource-limited execution result, not a negative signal estimate. The partial labels have not been scientifically audited or used to compute a whole-cohort signal statistic. No recurrent-model, biological-wiring, RL or controller-quality improvement is established. The prior completed query-gate result remains unchanged.
