# Lexical-input ablation: execution status

The [protocol](dialogue-qwen-lexical-ablation-protocol.md) and exact subtraction
implementation were published before preparation. This is a development control
motivated by the [retention diagnosis](dialogue-qwen-retention-results.md), not a
new architecture result. The complete comparison and independent result audit
are finished: **5/16 conditions passed and both arms failed**. See the
[paired results](dialogue-qwen-lexical-ablation-results.md).

## Source and synthetic qualification

The protocol was published in `1575bb3`, SHA256
`c04a1a0d77bedd028465f954de4261828dd23a75b83ea069da173455f787a329`.
Implementation and tests were published in `187c5f9`. Independent source review
found no material issue. The single
[synthetic preflight](../output/dialogue-qwen-lexical-ablation-v1/preflight-01/receipt.json)
passed **22 tests in 0.26 seconds** and lint. Its receipt is
`1df4b3acbb4249b090d5044932f1fb5a71e8012d5c89fa83c86773592ca17efa`.
All nine original study sources remain unchanged.

## Completed preparation

The sole preparation exited zero in **18.256 seconds**, with **886,931,456 bytes**
peak process RSS. It made zero model calls, used only the local pinned tokenizer,
and copied evaluation labels without decoding them. Both arms retain all
**7,444 batches / 15,638 decisions** and the original candidate-to-label mappings.

| Arm | Questions | Prompt tokens | Padded token positions | Longest prompt |
| --- | ---: | ---: | ---: | ---: |
| Current | 7,819 | 4,006,672 | 4,446,826 | 779 |
| History4 | 7,819 | 4,795,015 | 5,235,169 | 932 |

The new pilot selection contains 28 batches, chosen from prompt metadata using
the frozen first-twelve/longest-group rule. It has not supplied quality evidence.

- [Preparation plan](../output/dialogue-qwen-lexical-ablation-v1/preparation-01/plan.json):
  `89b90d4a7decddf35e2dfbfacd53ce15a61e455cb32be9a719092839367caa36`.
- [Completion receipt](../output/dialogue-qwen-lexical-ablation-v1/preparation-01/completed.json):
  `868f8306f98091e192a6d88a9f2660fdf22f89c18165471162cd6746002dea7f`.
- Exact request payload: **89,295,705 bytes**, SHA256
  `b08a5bef53a897f240cee73d56324b572da409cbff951d83adedafb4825aec1e`.
  The [lossless compressed publication](../output/dialogue-qwen-lexical-ablation-v1/publication-01/README.md)
  is **1,797,232 bytes** and passed round-trip size/hash verification.

## Admission and reporting boundary

The [independent paired-input audit](../output/dialogue-qwen-lexical-ablation-v1/preparation-audit-01/receipt.json)
passed once in **1.907 seconds** with **31,719,424 bytes** peak RSS. It checked all
7,444 paired requests, 15,638 questions and 87,902 candidate descriptions,
reconstructing the permitted string removals without importing the producer.
Opaque label bytes, mappings, ordering, workloads and the 28-request pilot
selection agree. Its receipt is
`ec9d86ba7d2bbb0c6da781cc037f36c8254b218bc4c167c65300ac4ca95ea44d`.
Tokenization and original public-input provenance remain inherited from the
authenticated preparation chain; this is not a model or quality check.

The fresh pilot completed **28 batches / 56 decisions** in **13.576 seconds**,
using **2,762,457,088 bytes** peak process RSS. All 28 attempted calls returned;
no choices or probabilities were saved and no labels were accessed. Its fixed
projection is **4,171 seconds**, below the **7,200-second** admission ceiling.
The descriptive request-rate alternative is 5,315 seconds and is not used for
admission. This projection is a heuristic, not measured complete-run latency.

The [pilot completion](../output/dialogue-qwen-lexical-ablation-v1/pilot-01/completed.json)
is SHA256 `08498d59ab5e3129213f3a3d82b9d87b56c0e0a5b428b95c421daa5fd99d5bbd`.
A [saved-only admission check](../output/dialogue-qwen-lexical-ablation-v1/pilot-admission-01.json)
independently recomputed the timing formula and checked manifests, all fourteen
source hashes, coverage and paid work. Session **53911 exited 0**; PID/process
group **90676** was absent on the
[terminal check](../output/dialogue-qwen-lexical-ablation-v1/pilot-terminal-01.json).

The full allocation started at **2026-09-20 14:24:45 UTC**, after the other
research task explicitly returned the numerical slot with terminal status and
process-group cleanup. Session **46815** exited **0** after all 7,444 requests
and 15,638 decisions completed in **2,479.636 seconds (41.327 minutes)**, with
**3,491,479,552 bytes** peak process RSS. All attempted calls returned. The
completion receipt is
`165552f6e7ca391145d6beade603829e2330e7d66511670d567af122109f9045`.
PID/process group **91127** was absent at the
[terminal check](../output/dialogue-qwen-lexical-ablation-v1/run-terminal-01.json)
at **15:06:34 UTC**, and the shared model/plant allocation was released. The
[launch record](../output/dialogue-qwen-lexical-ablation-v1/run-launch-01.json)
binds the prepared plan and admitted pilot. No partial quality was inspected,
and there was no retry, resume or scope reduction.

After successful complete inference, a separate saved-only report compared the
new predictions to the original corresponding-arm predictions. All **16**
prospective conditions were required: each arm had to reduce retained error by at
least two percentage points, lose no more than one point of changed accuracy,
and avoid increasing overall NLL or Brier, under both row and equal-service
weighting. Only **5/16** passed; both per-arm conjunctions failed. The no-flags
representation was not promoted. No autonomous-memory or architecture claim
follows.

The [separate reporter](../scripts/report_dialogue_qwen_lexical_ablation.py) passes
[15 synthetic checks](../output/dialogue-qwen-lexical-ablation-v1/report-preflight-01/receipt.json)
in 0.72 seconds and lint. It binds both completed campaigns, reconstructs their
paired choices/probabilities and reports all sixteen conditions. Independent
source review found no material issue at
reporter hash `c25ea0f6727fa96ee0af17eca4a3530ad379e70ae58f37ac9e7de8a028516928`
and test hash `0dbde93d198845cc1da14684cf4e4d885a4b69e2aff0ee05f6d001b7d03d3a3d`.
Its reconstruction inherits the pinned original
reporter and is not an additional independent result audit.

The actual report exited **0**, taking **7.742 seconds** and **609,746,944 bytes**
peak RSS, with zero model calls. Its
[receipt](../output/dialogue-qwen-lexical-ablation-v1/report-01/receipt.json) is
`37a947e727099b05df00c64458e1e9218cc4de02931845eaee15e1e03d84fb0d`,
and its summary is
`3668414c4785d1d35bedb186b5a560550edb5612b97e8c8b669aae08934d583b`.

The [independent saved-result checker](../scripts/audit_dialogue_qwen_lexical_result.py)
was qualified by its single
[synthetic preflight](../output/dialogue-qwen-lexical-ablation-v1/result-audit-preflight-01/receipt.json)
passed **14 tests in 2.33 seconds** and lint; independent source review found no
material issue at source hash
`f0067b1a71902e8388f127cda0b93eaa96b8cc79db0a55a143936dd683413a29`.
It reconstructs 84 primary metric cells and all 16 decisions through the pinned
original independent auditor, without importing reporter metric code. Exact
prompt subtraction, tokenization, actual inference, timing and secondary metrics
remain outside this arithmetic audit. The actual audit exited **0** and agreed
on **1,455 scalar comparisons**, all 84 primary cells and all 16 decisions. It
took **3.440 seconds** with **134,053,888 bytes** peak RSS and zero model calls.
Its [receipt](../output/dialogue-qwen-lexical-ablation-v1/result-audit-01/receipt.json)
is `f0e8d66984fb0cb68d9190803d74fa2a547d1d8f4480f87a7cd52023d8edbec6`.

The [comparison plotter](../scripts/plot_dialogue_qwen_lexical_ablation.py) is
prepared and visually checked on a clearly watermarked synthetic fixture.
Its [preflight receipt](../output/dialogue-qwen-lexical-ablation-v1/figure-preflight-01/preflight-receipt.json)
also records rejection of incorrect pins and extra directory members before
quality-summary decoding. The actual figure displays both arms, both weightings,
32 metric values and all 16 decisions, requiring an agreeing completed audit.
The [real comparison](../output/dialogue-qwen-lexical-ablation-v1/figure-01/comparison.png)
was generated after the agreeing audit in **0.486 seconds**, with zero model
calls. Its receipt is
`9f052ae1cf41249e208ae2c8c079c2a70247d5f595e6632d7f4e6dcee871e846`.
The rendered labels, all sixteen markers and scope notes were visually checked.

A subsequent authentication-only refinement accepts the independent auditor's
existing `2e-12` floating-point tolerance for metric deltas while keeping all
decision flags, thresholds, relations and 16 component identities exact. Its
[focused qualification](../output/dialogue-qwen-lexical-ablation-v1/figure-auth-preflight-02/receipt.json)
passes both valid and invalid cases and lint at plotter hash
`3e1588bcc8e282c42a59f8cb7bd894c6df8c2025ef95f3b0554738290c72a424`.
The rendering code and original synthetic figure are unchanged.
