# A pretrained recurrent control for the Qwen observation baseline

Source and checkpoint review, 20 September 2026. This work left the Qwen
comparison unchanged. The initial phase read public papers, source and configuration only.
A subsequent bounded inspection downloaded the pinned adapter once and read its
tensor metadata on the meta device. Neither phase imported upstream code, built
or ran a model, or inspected the live experiment's predictions.

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
- Locally verified weight file: 11,017,893 bytes, SHA256
  `a7c346c0166698070cc32dbe3bf2e7450c7b9aae86667f28b64d2d020ed039ec`.
  The subsequent download matches the upstream size and digest.
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

## Checkpoint dimensions match, numerical compatibility remains untested

The [bounded inspection receipt](../output/dialogue-delta-mem-review-v1/weights-inspection-01/receipt.json)
records one download and one `torch.load(map_location="meta", weights_only=True)`
in **2.36 seconds**, under a prospective 60-second limit. No tensor values were
evaluated. The downloaded checkpoint stays local; the published receipt includes
its official pinned URL through `started.json`, size and verified digest.

The [shape comparison](../output/dialogue-delta-mem-review-v1/shape-check-01.json)
matches all **324 tensor names and shapes** against the pinned source and our
cached Qwen configuration. Every stored tensor is bfloat16. The adapter contains
**5,456,160 stored scalars**, including inactive key/value corrections; its
released query/output configuration activates **4,866,336 scalars in 252 tensors**.

There is one **8 by 8 recurrent matrix per layer**, across 36 layers: 2,304 state
scalars per stream, or 4,608 bytes if stored in bfloat16. This excludes the base
model, adapter weights, KV cache and runtime workspace.

The source establishes several requirements for a future port:

- Apply the 4,096-wide query correction before query normalization and RoPE, and
  the 2,560-wide output correction after the attention output projection.
- Preserve pre-write reads and the released row-wise update
  `S_next = diag(lambda) S - diag(beta) (S k) k^T + diag(beta) v k^T`,
  with `lambda = 1 - beta`.
- Use correction scale `alpha / rank = 2`. The configuration's
  `online_gain = 0.05` governs initialization, not an extra inference multiplier.
- Preserve the quantized base projections and distinguish token-validity masks
  from causal attention masks, including padding, offsets and dialogue resets.

The installed MLX Qwen implementation has matching mathematical insertion
points. This is static structural agreement only: no numerical parity, memory
measurement, speed result or task-quality improvement has been established.

### The recurrence needs its own runtime qualification

The [pinned upstream scan](https://github.com/declare-lab/delta-Mem/blob/5cd5d9153c7f408764728d953565201e198c39e2/deltamem/kernels/affine_scan.py#L61)
uses CUDA/Triton, with one program per batch/state row and a sequential token
loop inside the kernel. It holds state in float32 internally, reads before
writing, and casts at returned boundaries. Its forward wrapper also allocates
state history. The non-Triton fallback uses per-token Python tensor operations.

The installed MLX `gated_delta.py` is not a direct substitute: it predicts after
decay, reads after writing, uses a head-scalar write gate, and its Metal indexing
assumes a key width divisible by 32. The released adapter has rank 8 and row-wise
gates. [Inspected local source identities](../output/dialogue-delta-mem-review-v1/kernel-source-01.json).

A future port should first qualify an inference-only rank-8 Metal recurrence
against an independent array reference. Check every read and final state with
nonzero initial state, unequal gates, padding holes and write-disabled reads.
Streaming comparisons must use identical chunk boundaries: bfloat16 rounding
at returned boundaries can make an unsegmented pass differ. Only then measure
single-token and prefill costs on a fixed shape grid, separating compilation
and including projections as well as the recurrence. Tiny kernels may
underutilize the GPU; a speedup is not established. Full-model integration and
any training gradients require separate qualification. No kernel was executed
in this source review.

## What comparison would answer a new question

The [completed semantic comparison](dialogue-qwen-observation-results.md) now
fails both behavioral and proper-score requirements: improved changed-value
recognition is outweighed by severe retention errors, and extra history worsens
the aggregate result. **This result does not promote the adapter to a quality
study.** Diagnose the observation interface and retained-value failures first.

If a later controlled observation baseline earns continuation, qualify the
released adapter's runtime and then compare carried state against the **same trained adapter reset
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
