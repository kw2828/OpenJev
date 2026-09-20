# A pretrained recurrent control for the Qwen observation baseline

Source review, 20 September 2026. The live Qwen comparison remains unchanged.
This review read public papers, source and configuration metadata only. It did
not download weights, import upstream code, run a model or inspect the live
experiment's predictions.

**Delta-mem is a useful missing control, not a new OpenJev architecture.** Our
[earlier comparison](dialogue-memory-results.md) tested small recurrent heads
after frozen MiniLM features. The [delta-mem paper](https://arxiv.org/html/2605.12357v1)
instead places associative state inside frozen Transformer attention, with
learned low-rank query/output corrections. Its TSW/SSW variants have 4.87 million
trained parameters, beyond the small state matrix. Appendix A reports one epoch
on 2,219 QASPER examples using eight A800 GPUs. Appendix B reports slower
decoding than vanilla. These findings do not establish OpenJev accuracy, local
speed or calibrated confidence.

## Verified release and implementation

The [official adapter](https://huggingface.co/declare-lab/delta-mem_qwen3_4b-instruct)
targets `Qwen/Qwen3-4B-Instruct-2507`, the backbone family underlying our current
four-bit MLX model. It is specifically **token-state-write, rank 8, query/output
corrections**, trained with write length 8,192. It is not a mergeable LoRA
checkpoint. The strongest MSW results in the paper belong to another variant
and must not be attributed to this released TSW adapter.

- Official code commit: `5cd5d9153c7f408764728d953565201e198c39e2`.
- Adapter revision: `c46dc31155608e412d44bf56638d5a6f856f2e7e`.
- Published weight metadata: 11,017,893 bytes, SHA256
  `a7c346c0166698070cc32dbe3bf2e7450c7b9aae86667f28b64d2d020ed039ec`.
  This is upstream metadata, not a locally verified weight digest.
- [Retrieved configuration and source descriptors](../output/dialogue-delta-mem-review-v1/source-01/receipt.json).
  Source snapshots stay local; descriptors identify the exact upstream URLs and
  retrieved hashes. Configuration and file-list metadata are included.

The [pinned core](https://github.com/declare-lab/delta-Mem/blob/5cd5d9153c7f408764728d953565201e198c39e2/deltamem/core/delta_impl.py)
implements row-wise coupled retention/write gates, pre-write reads, padding
masks, explicit write enable/disable and cloned state export. Its adapter weight
loader uses `weights_only=True`. Empty target-layer selection applies attention
adapters throughout the supported model. These are source observations, not
numerically validated compatibility with MLX.

The [chat runtime](https://github.com/declare-lab/delta-Mem/blob/5cd5d9153c7f408764728d953565201e198c39e2/deltamem/runtime/session.py)
also retains attention KV caches and processed token history. **The small delta
state is not the chat runtime's total memory footprint.** Carry/reset controls
must match the KV/text retention policy and count ingestion as well as query
cost. A changed streaming schedule would require separate qualification.

The [repository](https://github.com/declare-lab/delta-Mem) recommends CUDA/PyTorch.
Its public tree contains no detected license file and GitHub reports no license;
the adapter card declares CC BY 4.0. We have not copied its implementation into
OpenJev or established an executable Apple Silicon adapter.

## What comparison would answer a new question

First finish the [current semantic comparison](dialogue-qwen-observation-status.md).
If Qwen establishes useful observation quality, qualify the released adapter's
runtime and then compare carried state against the **same trained adapter reset
before each exchange**, using identical context, precision and readout. Vanilla
Qwen and an explicit candidate ledger remain practical controls. Match public
inputs and the cache/text retention policy, and account for every retained cache
and recurrent tensor. Separately evolved caches measure the whole trajectory
effect; identical KV tensors would isolate an immediate delta-state intervention.
A pretrained
adapter comparison is not training-data-matched to our small supervised heads.

Persist only public exchanges. Each question must start from the same saved
public state; discard its temporary writes and caches afterward. Question order,
candidate wording and privileged previous labels must not alter future memory.
An autonomous study must replace supplied previous gold with the system's own
state and charge every intermediate update.

The mechanism earns further work only if carrying state improves changed-value
decisions without creating extra retention errors beyond reset and explicit
context controls. A reset-adapter gain alone would not demonstrate a benefit
from carrying delta state. This is a conditional research direction; no new efficacy
claim or numerical run follows from this source review.

## A separate benchmark lead

[StateMemBench](https://arxiv.org/html/2608.19652v1) separates current, superseded
and other answers, with anti-update controls. It could expose failures missing
from short SGD dialogues. However, its answer pool is used for grading and is
**not shown to the original answering system**. Showing those alternatives to
OpenJev would create a distinct candidate-supplied task, not a directly
comparable reproduction. Dataset/code availability and licensing have not yet
been verified, and no examples were used here.
