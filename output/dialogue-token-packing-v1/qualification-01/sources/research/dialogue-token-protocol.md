# Shared token evidence: prospective dialogue control

Status: specified before new cache encoding or scientific fitting. Freeze and
publish this document with preparation sources before the fixed cost probe;
freeze the full study sources before fitting. The previous goal turn made
progress: a real candidate-concatenation cost probe and independent audit
established failed admission. That attempt remains stopped and unchanged.

## Question and scope

Does candidate-conditioned attention over shared contextual token vectors
improve unseen-service decisions compared with slot-only attention, under
both the existing readout and normalized scalar-memory heads? This tests
evidence extraction before introducing a different recurrent mechanism.
It follows the [shared-token design](dialogue-token-evidence-design.md).

Attention for slot-utterance matching is established prior work, including
[SUMBT](https://aclanthology.org/P19-1546/). This frozen-MiniLM adaptation is
neither a SUMBT reproduction nor a new world model. A pass qualifies a stronger
observation baseline for later architecture work; it does not establish
novelty, calibrated confidence or generalization to an untouched benchmark.

Reuse the exact SGD train/development subset, supplied service/slot/candidates,
lexical cache and model revision from the corrected memory study. The official
test set remains untouched. No new labels, teacher/API calls, encoder
fine-tuning, gold previous states, annotation spans, system acts or future
utterances enter the actor. Neither labels nor scored-time eligibility may
select contexts, tokens, query updates or cache misses.

The training subset has 2,017 dialogues and 51,741 scored questions, with
32,743 NONE, 357 DONTCARE and 18,641 assigned values. Development has 2,363
dialogues and 62,329 scored questions. Unseen support includes 524 assigned
TRUE, zero FALSE, 179 DONTCARE and 203 revisions. Seen support includes 1,938
TRUE, zero FALSE, 109 DONTCARE and 241 revisions. These are question counts,
not independent dialogue counts. The exposed development set and missing
FALSE/clear support constrain any result.

## One frozen representation and two attention modes

Use `sentence-transformers/all-MiniLM-L6-v2`, revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, with the same six authenticated
model/tokenizer files, float32, eager attention, inference mode and batches of
128 on MPS. Each text is exactly the existing
`System: <preceding system>\nUser: <current user>` template, with an actual
newline. Cache each exact unique public context once, including unscored
turns. Candidate descriptions are never concatenated inside the Transformer.

Tokenize without truncation. Split into nonoverlapping chunks of at most 254
content tokens, add CLS/SEP and keep every valid last-layer token vector.
Store raw float32 vectors, float32 pooling priors, int64 text offsets,
explicit chunk lengths, public context indices and source identities. No
padding token is retained as a valid token. No cross-chunk Transformer
attention is introduced. Original packet text indices must agree with the
reconstructed public strings.

For a chunk with `n` content tokens, each of its `n+2` valid tokens receives
prior weight `n / (total_context_content_tokens * (n+2))`. Weighted pooling
followed by L2 normalization must recover the original sentence vector with
maximum absolute coordinate error at most **2e-5** for every encoded context,
both in the sample and full cache. This tolerance permits floating-point
batching differences, not a different representation or omitted content.
Failure stops the attempt; do not switch pooling or relax the tolerance.

For query vector `q`, candidate vector `c` and token vector `H[i]`, candidate
attention uses bias-free projections `z = Wq [q;c]` (768 to 64) and
`k[i] = Wk H[i]` (384 to 64). Token weights are
`softmax_i(log(prior[i]) + dot(z,k[i])/sqrt(64))`. The evidence vector is the
L2-normalized weighted sum of raw token states. Slot-only attention uses
`[q;q]` in place of `[q;c]`; every other operation and registered parameter
remains the same. Keep token masking and prior weighting identical.

Pass evidence into the existing candidate-specific turn projection and V2
readout/scalar heads. Preserve query/candidate vectors, ten lexical features,
feature order, normalization and recurrent transition. The two adapter
matrices add **73,728** parameters, for **173,186** registered parameters per
fit. Equal registered counts do not imply identical effective expressivity:
the slot-only query input repeats `q`. The old 99,458-parameter mean-pooling
models are historical references, not equal-capacity controls.

Initialize the parent head first, then the adapter. Each corresponding parent
substate must match the V2 seed's initial tensor digest; all four arms within
a seed must share the same full initial tensor digest. There is no warm start
from trained weights. Padding must leave recurrent state exactly unchanged;
all-masked real contexts fail, while padded turns receive a fixed zero
evidence vector. Query and candidate permutations may only permute their
outputs; unrelated questions and future turns cannot change a past prediction.

## Prospective preparation cost rule

Preparation, capacity and full encoding use separate exclusive destinations.
Freeze original input completions, runtime, device, six encoder identities,
public layout, protocol and complete source closure before neural encoding.
The preparation and capacity phases each have a **120-second** total cap.
The capacity probe encodes exactly the first **512 unique contexts** in fixed
construction order. Full-workload tokenization may determine chunk lengths
and padding costs, but only these sample contexts receive neural encoding.

Record three disjoint time buckets. Synchronized neural forward plus transfer
of raw hidden states to CPU is scaled by full/sample **padded token positions**.
Retention of token/prior arrays, flush, weighted-mean equivalence validation
and payload hashing is scaled by full/sample **valid token rows**. Add observed
remaining capacity overhead, including full tokenization, authentication,
setup and metadata work. This sums to the projected full preparation time;
do not substitute the old sentence-vector timing or double count a bucket.
File flushing is not a measurement of guaranteed durable disk throughput.

Permit full encoding only if projected total is at most **720 seconds** and
projected total cache size is at most **4 GiB**, including vectors, indices,
priors, chunk/layout metadata, source copies and headers. Full encoding has a
separate **900-second** execution cap and records actual time, bytes, calls,
tokens/chunks, padding, pooling equivalence and memory. The sample is additional
work even if later recomputed. Its cold deterministic prefix is a heuristic
cost estimate, not a representative latency sample or measured full latency.

All phases preserve partial and failed work, check time and input identities
before terminal success and prohibit retries, resume or overwriting an
attempt. A rejection cannot silently change hardware, precision, prompts,
sample, data, caps or thresholds. Do not run other corpus/model experiments
during capacity, full encoding or fitting. Source review and saved-artifact
analysis may proceed independently.

## Paired scientific comparison

Only a completed admitted cache permits study freezing and training. Publish
the complete study source/input/runtime manifest before the first fit. Within
each seed, run `slot_readout`, `slot_scalar`, `candidate_readout`, then
`candidate_scalar`. Seeds are **4101, 4102, 4103**. Preserve every fit; do not
select a head, checkpoint, seed, schedule or temperature using development.

Use the unchanged V2 recipe: **20 epochs**, batch 32, AdamW learning rate
.001, weight decay .0001, gradient clipping 1, four CPU threads and the same
three-stratum loss weights. Match epoch permutations across all four arms.
Train through predicted recurrent states and evaluate only final weights.
There are 1,280 optimizer updates and 1,034,820 supervised presentations per
fit; the twelve-fit total is 15,360 updates and 12,417,840 presentations.
The whole train/evaluate command has a **3,600-second** cap, including input
authentication, evaluation and output. A stopped run is not a complete study.

Retain the existing read-only monitor and **2e-6** tolerance for incoming,
actual feature-prior and outgoing probability normalization, and released
mass in `[0,1+2e-6]`. Monitor real public turns including unscored updates and
executed dummy question slots. Tests must establish that monitoring preserves
outputs and gradients, and that padding, causality and permutation behavior
hold. Technical validity is separate from task effectiveness.

## Fixed continuation rule

Require all twelve technically valid, authenticated fits. For each head,
candidate attention must meet all seven conditions against slot-only
attention, averaging all three seeds:

1. Unseen assigned-TRUE accuracy improves by at least 10 percentage points.
2. Unseen DONTCARE accuracy improves by at least 10 percentage points.
3. Unseen three-stratum macro accuracy improves by at least 2 percentage points.
4. Seen three-stratum macro accuracy is no more than 1 point worse.
5. Unseen micro NLL is no worse.
6. Unseen macro strictly improves in at least two of the three paired seeds.
7. Unseen revision accuracy is no more than 1 point worse.

Both heads must pass, giving **15 checks** including complete membership.
Accuracy margins use exact integer-ratio comparisons, inclusive boundaries
and no epsilon. NLL uses float64, no floor or favorable tolerance; zero target
probability gives infinite NLL. Missing support or undefined required metrics
fail. Report zero-FALSE support as undefined. Literal carry, always-NONE and
historical V2 are descriptive references outside this rule. All-subgroup and
per-seed results remain visible even when the aggregate rule fails.

## Cost, reporting and provenance

The token cache is shared by all arms. Assemble padded token blocks once per
batch, without repeating raw vectors along question/candidate axes. Record
cache bytes gathered, valid/padded token positions, emitted tensor bytes,
key/query projections, attention score and evidence positions, real/public
candidate updates, optimizer work, elapsed phases and process memory. Learned
keys cannot be cached across optimizer updates. Count pooling and recurrence;
cached-head latency alone is not end-to-end serving latency. No speed-quality
frontier claim follows without a separately specified serving benchmark.

Keep the 52-file full training layout (four root files and four per fit).
The saved-output reporter authenticates the exact source/input/cache closure,
initialization and paired batch orders, training invariants and all metrics
without loading model weights or executing neural models. Publish aggregate
results, all-fit figures, metadata receipts and source. Raw dialogue strings,
token vectors and individual predictions remain local with published hashes.
No result from an unfinished run or synthetic fixture is empirical efficacy.

## Parent identities

- V2 plan: `9c39c3b27c7219190bc3f45fc342bc4da4eb408e622402a92ce51efb68ed3905`.
- V2 completion: `768376e24824523b9a5c26923c14206e567fda4f3578ed1babdf7cbdfb1d89f4`.
- Feature packet: `e4503c3dd63d28b74e91877dfc9c69b81f4b880e4f07d240cf8c08331fa0b308`.
- Lexical cache: `2e3e528ae05e55a32aa812d25a9dd609d7878e4daff77130994b0a63c32dad54`.
- Prepared data: `677d37ea581b3165b0adc96a0a0daab74271e0434df2ad23ace9568537e2cd12`.
- Earlier rejected candidate-encoding capacity:
  `7f8174f3afb7c81e29740e373f83224d0ca723a14d847f667e12c59aa1d779b3`.

Retain MiniLM's Apache-2.0 attribution and SGD's CC BY-SA-4.0 dataset terms.
