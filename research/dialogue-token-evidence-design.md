# Shared token evidence for dialogue state tracking

Status: source-only proposal, not a frozen protocol or an executed experiment.
No corpus, encoder, model or training calls were made for this note. The
candidate-concatenation study remains stopped at its failed capacity screen.

Its [capacity receipt](../output/dialogue-joint-v1/capacity-01/completed.json)
has SHA256 `7f8174f3afb7c81e29740e373f83224d0ca723a14d847f667e12c59aa1d779b3`.
The fixed first 512 strings completed in 25.4443 seconds, including 0.5582
seconds of synchronized encoder work. Extrapolation to 1,202,932 unique
candidate/context strings gave **1,254.4835 seconds**, above the 720-second
admission limit. Projected cache bytes were 1,868,030,878, below 4 GiB. The
time projection is heuristic, not measured full encoding time. No full cache
or scientific fit was run. This proposal changes the representation and needs
its own prospective protocol; it cannot convert that rejection into a pass.

## Relevant established method

[SUMBT, ACL 2019, sections 2.1–2.4](https://aclanthology.org/P19-1546.pdf)
encodes the system/user pair into contextual token vectors once per turn.
The slot embedding queries these vectors through learned multi-head attention;
an RNN tracks the resulting context, and distances to value embeddings define
the output distribution. The utterance BERT is fine-tuned, while the separate
slot/value BERT is fixed. Its controls include pooled BERT+RNN and
BERT+RNN+Ontology; evaluation uses WOZ and MultiWOZ. Consequently, retaining
frozen MiniLM and adding candidate-conditioned token pooling would adapt an
established late-interaction pattern, not reproduce SUMBT. The
[authors' implementation](https://github.com/SKTBrain/SUMBT) is a reference;
this proposal requires no imported upstream implementation or weights.

## One bounded representation proposal

Keep the frozen encoder revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, current public
`System: ...\nUser: ...` context, existing schema vectors and ten lexical
features. The pinned [MiniLM configuration](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/config.json)
has six layers, 12 attention heads, hidden width 384 and intermediate width
1,536. The locally authenticated configuration SHA256 is
`953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41`.
The existing encoder uses last-layer token states, mask-mean pooling and final
L2 normalization, as recorded in the [preparation note](dialogue-memory-preparation.md).
Retain its Apache-2.0 model attribution and the dataset's CC BY-SA attribution.

Cache the **raw float32 contextual token states** for each exact public context,
with offsets, chunk lengths and valid-token masks. Encode no candidate inside
the Transformer context. The encoder runs once per unique context, or once per
254-content-token chunk for long contexts; it never runs once per question or
candidate. Existing sentence-vector files cannot reconstruct discarded token
states, so this requires a new cache pass under a separately frozen protocol.

For token states `H[i]` of width 384, schema query `q` and candidate vector `c`,
use a small shared trainable attention adapter:

```text
z = Wq [q; c]                 # 768 -> 64, no bias
k[i] = Wk H[i]                # 384 -> 64, no bias
a = softmax_i(log(pi[i]) + dot(z, k[i]) / sqrt(64))
e = L2_normalize(sum_i a[i] H[i])   # 384-dimensional evidence
```

Pass `e` through the existing V2 turn projection and unchanged readout or
normalized scalar-memory head. Train the adapter with the head; keep the
Transformer and schema vectors frozen. The adapter adds 73,728 parameters per
fit, so the system has 173,186 registered trainable parameters rather than the
old head's 99,458. This is an observation-adapter comparison, not equal-capacity
evidence against the old model.

`pi` preserves the old pooling convention: for a chunk with `n` content tokens,
each of its `n+2` valid tokens receives prior weight
`n / (sum_chunk_content_tokens * (n+2))`. Include CLS/SEP as before. Zero
attention logits then recover the original chunk-weighted mean, up to declared
rounding. Normalize across all valid tokens of this context, never across
candidates, questions, dialogue turns or batch examples. No cross-chunk
Transformer attention is added.

## One falsifiable control

Compare **candidate-conditioned attention** above with **slot-only attention**,
which replaces `[q;c]` by `[q;q]`. Use the same two projection matrices,
initial tensors, token cache, operations, parameter count, optimizer schedule
and training cases; cross this one contrast with the existing readout and
scalar heads and all three paired seeds. Slot-only pooling is a stronger
control than uniform pooling because it can already retrieve slot-relevant
text. It does not have the same effective parameterization as the conditioned
arm, so equal registered counts are not a claim of identical expressivity.

Proposed quality requirement: preserve the seven per-head thresholds from the
[stopped joint protocol](dialogue-joint-protocol.md), now comparing conditioned
against slot-only attention. Both heads must pass, including +10 points each
on unseen TRUE and DONTCARE, +2 macro points and the NLL/seen/revision/paired
guards. Do not select the better head. Historical mean-pooling V2 remains a
descriptive reference. A gain isolates useful candidate conditioning within
this adapter recipe, not a new memory law or a world model.

Proposed compute requirement: independently qualify the shared token cache
under the same 720-second admission, 900-second execution and 4-GiB cache
limits, then retain the 3,600-second paired-fit cap. Reject rather than silently
change precision, truncate text or add a larger budget. These are design
recommendations, not an already frozen continuation rule.

## Cost and causal limits

The prior completed feature receipt recorded 44,763 encoded strings/chunks and
1,337,813 valid tokens, including schema strings. Reusing its exact templates
gives a conservative context-token payload bound of **2,054,880,768 bytes** at
384 float32 values per token, before masks, offsets, metadata and serialization.
There are 44,410 public context occurrences across the two cohorts. These are
saved-count bounds, not a newly tokenized workload or measured token-cache
cost. The old 15.2568-second sentence-cache timing excluded retaining token
states and cannot be advertised as the new preparation time.

The Transformer avoids candidate multiplication, but pooling still costs
roughly `O(tokens * 384 * 64 + questions * candidates * tokens * (64+384))`
per turn, plus projection of each schema pair. Trainable projected keys cannot
be cached permanently across optimizer updates. Gather ragged token blocks
lazily and batch the attention without materializing a full token tensor
repeated along every candidate dimension. Record actual peak memory, bytes
read, padding, pooling work and elapsed time. Candidate/query additions and
permutations must not change another query's results.

Token hidden states and normalized mean-pooled schema vectors have the same
width but are not guaranteed to share a calibrated dot-product space. Learned
query/key projections address that modeling problem without claiming it is
solved. A raw cosine score or attention weight is not a probability that the
utterance supports a candidate; negation, corrections and DONTCARE still need
supervised interpretation. Inspect finite masks and empty-context handling;
padding must never enter softmax, and an all-masked row must be rejected or
handled by an explicitly fixed sentinel rather than produce NaNs.

Persist all supplied queries' public prefixes, including unscored turns.
Neither target values nor frame eligibility may choose which tokens are
encoded, pooled or updated. No future turn may enter a past context. A lazy
cache must charge first-access encoding and avoid using gold labels to choose
cache misses. Serving cost includes tokenization, encoder misses, pooling,
schema lookup and memory update; cached-head latency alone is not end-to-end
latency. The exposed development set, supplied service/slot routing, zero
unseen-FALSE support and frozen pretrained Transformer all remain limitations.
