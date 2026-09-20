# Trainable observations: numerical and cost qualification

Prospective protocol, 20 September 2026. This qualifies the implementation for
the [observation-learning control](dialogue-observation-finetune-design.md).
It is not a task-quality study and admits no scientific training automatically.

## Fixed inputs and preparation

Use the pinned MiniLM revision, exact historical texts/pooling and normalized
scalar memory described in the design. Authenticate the original feature
packet, public TRAIN streams, schema catalog, lexical layout/cache and source
closure. Construct token-ID inputs for all 2,017 original TRAIN dialogues;
every supplied query receives every public USER exchange. Gold values, bins
and annotation availability do not select training targets or update masks.
The source packet contains previously exposed DEV records, but only TRAIN is
prepared. No DEV dialogue file or official TEST content is opened.

Keep 254 content tokens per encoder chunk, add CLS/SEP, use CLS/SEP-inclusive
masked means, weight chunk means by content length, combine then normalize.
Rank dialogue workloads by sum of batch size times maximum chunk length
squared, with chunk batch size 32 and dialogue-ID tie breaking. Select ranks
`floor(p*(N-1))` for p=0.10, 0.50 and 0.90 before any encoder execution.
Record all workload metadata and all three selected identities. Preparation
has a 300-second wall cap, 8-GiB RSS cap and 256-MiB output cap, with no model
forward. Its tokenization/authentication/storage time is separate paid work.

## One model pilot

Use MPS float32 MiniLM with eager attention, CPU float32 scalar memory, one
CPU thread and disabled encoder dropout. Pooling, gathering and transfer to
CPU remain differentiable. Frozen encoder uses no-grad tensors that the
trainable memory can safely consume. Re-encode schemas and dialogue text on
every microbatch in both arms; this matched path is not the cheapest cached
frozen baseline.

There are six cells: three selected workloads, each frozen then trainable.
Each cell starts from the same pretrained encoder and scalar initialization
seed 6801. Record full initial tensor digests and assert equality. Use separate
AdamW optimizers, scalar learning rate 0.001, encoder rate 0.00002, weight decay
0.0001 and global gradient clipping at 1. The pilot optimizes synthetic losses
only and discards all resulting weights.

Each cell executes one warm and two measured optimizer updates. Each update
accumulates four independent full-dialogue forward/backward microbatches of
its representative, with fresh encoder computation each time. This is
**18 updates and 72 dialogue visits**, not 72 independent dialogues. Targets
cycle deterministically through supported candidate indices using time, query
position and update/microbatch offset, covering every real T×Q endpoint.
Average each dialogue loss over T×Q, then divide by four before backward.
No official target, scored-row mask or task prediction is saved or evaluated.

Before updates in each cell, encode every selected context/query/candidate
text and compare with its authenticated historical pooled vector. Maximum
absolute error must be ≤2e-5, without a changed tolerance after outcomes.
After the three updates, perform one separately timed full-dialogue evaluation
forward with no backward. Count these parity/evaluation encoder calls too.

Require finite nonzero gradients for the trainable encoder's word embeddings
and first attention-query weight, finite nonzero scalar gradients in both
arms, and finite updated parameters. Frozen encoder gradients must remain
absent and its tensor digest unchanged. Monitor the scalar's actual incoming,
feature and outgoing beliefs and released mass with the existing V2 monitor
and its 2e-6 normalization tolerance. Initial state is NONE; no reset or detach
occurs inside a dialogue. Synthetic pooling and permutation tests precede this
pilot; they do not replace real-checkpoint parity or gradient checks.

The complete pilot has a **300-second wall cap**, **8-GiB process RSS cap**,
**8-GiB sampled MPS driver-allocation cap** and **256-MiB output cap**. Timing
includes input/source/model authentication, imports, loading, transfer and
synchronization, checks, optimizer work and output closing. MPS memory is
reported separately because it is not additive to process RSS; sampled values
do not prove a transient accelerator peak. A parent watchdog terminates the
entire process group on timeout. Coordinate the shared model/plant slot before
launch and record actual termination and group cleanup.

## Decision and failure behavior

Publish source/tests/protocol before preparation, then publish the complete
prepared plan before the single model execution. Source/input changes reject
execution. Use exclusive directories, retain partial work and failure receipts,
and do not resume/retry/shrink a failed frozen attempt. Report every cell and
all paid work, including warm updates and extra parity/evaluation passes.

Passing establishes the tested gradient path and measured cost on three
specific workloads. It does not establish accuracy, full-training time, MPS
peak memory, a novel architecture or superiority over cached inference.
Whole-study cost projection, if later produced, remains a heuristic requiring
its own complete allocation and matched scientific protocol.
