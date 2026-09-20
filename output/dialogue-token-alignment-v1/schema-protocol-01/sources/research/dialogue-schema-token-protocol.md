# Exact schema token cache

This is one new frozen-feature preparation, not a fitted-model experiment.
It supports the [token alignment design](dialogue-token-alignment-design.md).
The earlier typed study and its failed continuation rule remain unchanged.

Use the original typed study's exact 29,211 fit and 13,599 evaluation row
membership, all within official TRAIN. Select their 53 public query schemas,
including every supplied candidate, irrespective of current labels. Include
the standalone query strings for provenance as well. Candidate strings already
include the full service/slot question followed by the original value text.
Do not paraphrase, concatenate dialogue text, or select only correct candidates.

Deduplicate exact strings and retain their original pooled-feature indices.
Reuse the frozen MiniLM revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`,
float32, MPS, 254 content tokens per chunk, CLS/SEP, and 128 chunks per encoder
batch. Preserve all tokens. Reuse the pinned raw-token encoding primitive.
Chunk priors are the original content-length weighting spread over the
chunk's content and special tokens. The normalized prior-weighted pooled
vector must agree with the original cached schema vector within 2e-5 maximum
absolute error for every string.

Create one exclusive metadata freeze and one exclusive encoding attempt.
Each has a 120-second whole-attempt limit and a 256-MiB output limit. Check
actual tokenized output size before the first encoder forward. Preserve any
failure, partial files, and call counts; do not resume or silently retry.
Authenticate inherited input receipts, implementation files, encoder weights,
runtime, and all new outputs. Freeze/publish this source and protocol before
encoding. Record setup, tokenization, encoding, parity, hashing, total time,
process peak memory, and byte counts. This is preparation cost, not model
quality or an inference speed comparison.

The output is flat float32 token vectors and priors, integer text/chunk
offsets, and query/candidate-to-text mapping. A metadata-only reader hashes
float payloads opaquely and reads only the index and integer geometry. The
full loader additionally recomputes pooled parity. No official DEV inference,
official TEST access, parameter training, model selection, or efficacy metric
is part of this preparation.
