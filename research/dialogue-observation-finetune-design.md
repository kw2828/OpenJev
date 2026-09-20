# Next control: learn observations while keeping autonomous scalar memory

Source and runtime-metadata review, 20 September 2026. This is an implementation
design, not a frozen training protocol or an admitted run. The completed
[history-support audit](dialogue-history-support-results.md) does not justify
the proposed delayed-proposal mechanism on this cohort. It points back to
interpretation of public text and caller-supplied values.

## The comparison

Fit fresh instances of the same normalized scalar memory with either a frozen
MiniLM encoder or that encoder trained end to end. Match public streams,
schemas, candidates, lexical observations, head initialization, dialogue orders
and loss. Initialize at NOT_MENTIONED once per independent dialogue-query
stream. Process every public USER exchange, including unscored ones; labels
select losses only. Keep predicted state throughout, without gold resets or
detaching between turns.

Include a separately identified number-normalization observation control in
the eventual comparison. It tests whether interpreting number words accounts
for improvements; it is not a semantic oracle, and incidental number matches
can still be wrong. Do not quietly add those features only to the proposed arm.
The exact normalization, loss, seeds and study margins still need to be frozen.

This tests representation learning. A gain would improve the practical
baseline, but would not establish a new recurrent architecture. A later novel
operator must beat this stronger observation-matched scalar and established
ledger/capacity controls on an untouched evaluation.

## Minimal implementation path

Use the existing `sentence-transformers/all-MiniLM-L6-v2`, revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Metadata identifies six layers,
width 384 and 12 attention heads. Keep its current pooling contract: 254-token
content chunks, CLS/SEP-inclusive masked chunk means, content-length-weighted
chunk combination, then L2 normalization.

Reuse the exact text construction in
[`build_contexts`](../scripts/prepare_dialogue_tokens.py) and the schema/candidate
construction in [`build_layout`](../scripts/prepare_dialogue_schema_tokens.py).
Cache token IDs and immutable identity maps. Encode each public turn once,
sharing its differentiable vector across the supplied independent queries.
Encode needed unique schemas/candidates in each dialogue microbatch and gather
their vectors into actor shapes without NumPy conversion. Recompute embeddings
after every optimizer update. Candidate-plus-turn joint encoding is a different,
more expensive observation intervention and is not part of this minimal control.

Reuse [`DialogueCopyMemoryV2`](../src/openjev/research/dialogue_copy_memory_v2.py)
with `method="scalar"`, without editing the source frozen in prior experiments.
Existing cache-based encoding uses inference mode, frozen parameters and NumPy
outputs; existing batch assembly copies those arrays, so neither is a training
path for the encoder. Add a differentiable wrapper and a separate runner.

Keep encoder dropout disabled in both arms (`eval()` still permits gradients)
so the comparison does not also add stochastic observations. Verify gradients
reach encoder embeddings and attention parameters; an unused BERT pooler need
not receive gradients. Check frozen-wrapper outputs against the current
pooling implementation on identical synthetic token sequences.

The old state monitor uses CPU masks and float64 reductions. A candidate local
path is an MPS encoder, differentiable transfer of pooled vectors to CPU, and
the unchanged monitored CPU scalar. That cross-device gradient path is not yet
qualified. Include transfer and synchronization in measured cost; do not claim
that accelerator availability proves numerical compatibility or speed.

## Bounded qualification before training

First test gradient flow, full-stream state continuity, masking, schema
permutation and exact pooling semantics using synthetic inputs. Then freeze a
single cost pilot with actual token workload selected by metadata, without
looking at task quality: dialogue microbatch one, accumulation over four full
dialogues, and three fixed representative workloads. Measure frozen and
trainable paths, including context/schema encoding, transfer, recurrence, loss,
backward, clipping, optimizer, validation and evaluation. A prospective
300-second cap is proposed; publish the exact selection, repetitions and
resource limits before execution. No pilot has run.

Any projected full-study cost is a heuristic, not measured training time.
Do not reuse frozen-forward timings to estimate encoder backward cost. Prepare
the complete training allocation and matched scientific comparisons only after
this implementation qualification, and coordinate the shared numerical slot
before model execution. Official TEST remains outside development.
