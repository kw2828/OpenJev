# Candidate-conditioned evidence: prospective development control

Status: fixed before capacity encoding, cache construction or any new fit.
The previous turn made empirical progress: 15 normalized fits completed and
passed numerical checks, but selective retention still failed its original
rule. Preserve that failure. This study tests the representation bottleneck
identified in the [observation design](dialogue-observation-design.md), not a
new memory mechanism.

## Question and boundary

Does letting the same frozen encoder read the supplied candidate together with
the current dialogue improve unseen-service TRUE and DONTCARE recognition
under both a readout head and scalar recurrent memory? A gain under both heads
supports a useful representation change in this recipe. A scalar-only gain
indicates an interaction, not proof that a memory operator is superior. A
failed recipe remains failed if one head, seed or subgroup looks better.

Schema-conditioned evidence extraction is established work, including
[SUMBT](https://aclanthology.org/P19-1546/) and
[TripPy](https://aclanthology.org/2020.sigdial-1.4/). The proposed concatenation
with a frozen sentence encoder does not reproduce those systems or introduce
architectural novelty. Its purpose is to qualify a stronger observation
baseline before further recurrent architecture work.

Reuse the authenticated SGD train/development subset, source revision,
supplied service/slot/candidates, lexical cache and MiniLM weights from the
[normalized study](dialogue-copy-v2-protocol.md). Official test contents stay
untouched. No teacher, external API, new labels, encoder fine-tuning, gold
previous state, annotation spans, system acts or future utterances are allowed.
The development set is heavily exposed; even a pass is not confirmation.

The retained training set has 2,017 dialogues / 51,741 scored questions:
32,743 NONE, 357 DONTCARE and 18,641 assigned values, including 2,316 TRUE
and 136 FALSE targets for boolean ontologies. Development has 2,363 dialogues
/ 62,329 questions. Unseen support is 524 assigned TRUE, zero assigned FALSE,
179 DONTCARE and 203 revisions; seen support is 1,938 TRUE, zero FALSE,
109 DONTCARE and 241 revisions. No development clear examples exist. These
are scored-question counts, not independent dialogues. This panel cannot
establish general boolean polarity recognition.

## One representation change

The independent arm keeps the existing 384-dimensional normalized observation
vector for `System: <preceding system>\nUser: <current user>`, broadcast to every
supplied question/candidate. The joint arm replaces only that vector with the
frozen encoding of this exact template:

```text
<existing candidate description>\nSystem: <preceding system>\nUser: <current user>
```

Here each `\n` means an actual newline. The existing candidate description
already contains service, slot and value semantics, including explanatory
NONE/DONTCARE text. Do not add prompts, examples, metadata, labels or a new
negation rule. Preserve the separate query and candidate vectors, ten lexical
features, projection dimensions, feature order, parameter names/counts, head
initialization and normalized V2 recurrence.

Use `sentence-transformers/all-MiniLM-L6-v2` revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, float32, eval/inference mode,
eager attention, batch 128 and the same six authenticated model/tokenizer
files. Tokenize without truncation. Split content into nonoverlapping chunks
of 254 tokens, add CLS/SEP, mask-mean-pool each chunk including those special
tokens, combine chunk vectors weighted by content-token count and L2 normalize.
No cross-chunk attention is introduced. Record all chunk/token counts and
overlength text counts; later chunks may lack schema text.

Encode every real public turn for every supplied query and valid candidate,
including NONE and DONTCARE. Exact-string deduplication may share work. Query
membership may use the pre-existing supplied-query layout, but scored-time
eligibility, target values and transitions cannot enter actor features, update
masks or encoding selection. Unrelated queries cannot affect each other.
Future text changes cannot change earlier features. Candidate permutations
must only permute candidate outputs. Zero-padded cache indices are not real
encoding work and must be counted separately.

## Capacity screen and bounded encoding

Freeze and publish the preparation plan, this protocol, preparation source
and tests, original encoding source, exact inputs, six model files, device and
runtime before encoding. Use MPS on the current machine. A capacity attempt
has a fixed **120-second** total cap, including authentication/loading and
output work. It encodes only the first **512 unique strings in deterministic
construction order**. No labels, model-head fitting or effectiveness metrics
are used to choose the sample or decide capacity.

Record sample and full-corpus token/chunk counts, synchronized encoder time,
loading/authentication and other elapsed overhead, measured device memory
where available, process peak resident memory, exact cache-byte estimate and
output size. Estimate full encoder time by the sample's elapsed encoding time
per padded token position multiplied by all full-workload padded token
positions under the same sequential batching. Add the capacity attempt's
observed non-encoding time. This is a cost projection, not measured full time
or a statistically representative latency estimate.

Proceed only if the projection is at most **720 seconds**, leaving headroom
under a fixed **900-second** full-encoding cap, and the projected cache is at
most **4 GiB**. The full attempt records actual costs and checks its limits
between batches and before terminal success. The capacity work is additional
cost, even if full encoding recomputes its sample. No cheaper backend, sample,
prompt, cap or dataset may silently replace a rejected attempt.

Every phase uses an exclusive destination. Failure or timeout preserves
completed and partial work with receipts and stops that attempt. No retry,
resume or overwrite is permitted. Input/source drift rejects execution.
Do not run other corpus/model experiments during capacity, encoding or fitting.
Source review and saved-output analysis may run independently.

## Paired two-by-two fit

Only after capacity and full cache completion, freeze and publish all study
sources, tests, input receipts, runtime and output identity. Train four arms
in this fixed order inside each seed: `independent_readout`,
`independent_scalar`, `joint_readout`, `joint_scalar`. Seeds are 4101, 4102,
4103. Every arm has the original 99,458 registered parameters; unused departure
parameters in readout remain disclosed. No trainable encoder parameters count
toward that head total.

Use the V2 recipe: 20 epochs, batch 32, AdamW learning rate .001, weight decay
.0001, gradient clipping 1, four CPU threads and the same three-stratum loss
weights. Match full initial tensor hashes to the corresponding completed V2
head/seed and across both representations; match epoch permutations across
all four arms. Train through predicted states, evaluate final weights only,
and preserve all 12 fits. No checkpoint, seed, head or hyperparameter search.

The full train/evaluate command has a **3,600-second** cap from before input
authentication through final validation/output, checked between batches.
There are 1,280 optimizer steps and 1,034,820 supervised presentations per fit:
15,360 steps and 12,417,840 presentations total. Record every batch's indices,
loss, public-layout work, observation tensor work and internal invariant
witnesses. Use the existing read-only V2 monitoring contract with tolerance
`2e-6` for incoming, actual feature-prior and outgoing state, and released
mass in `[0,1+2e-6]`. Check all real public updates, including unscored turns
and executed dummy question positions. Monitoring must preserve computation
and gradients. Technical validity is a prerequisite, separate from efficacy.

## Fixed continuation rule

Require complete, authenticated, technically valid outputs from all 12 fits.
For **each** head independently compare joint against its matched independent
arm, averaging every seed. All seven conditions must pass for both heads:

1. Unseen assigned-TRUE accuracy improves by at least 10 percentage points.
2. Unseen DONTCARE accuracy improves by at least 10 percentage points.
3. Unseen three-stratum macro accuracy improves by at least 2 percentage points.
4. Seen three-stratum macro accuracy is no more than 1 point worse.
5. Unseen micro NLL is no worse.
6. Joint strictly wins unseen macro in at least two of the three paired seeds.
7. Unseen revision accuracy is no more than 1 point worse.

This gives **15 scientific checks**: completed membership plus seven per head.
Use exact integer-ratio accuracy comparisons and inclusive margins, without
rounding or epsilon. NLL uses float64 with no floor or favorable tolerance.
A zero target probability means infinite NLL; missing support or undefined
metrics fail. The zero-FALSE subgroup is reported as undefined, not a success.
Literal carry, always-NONE and historical V2 are references outside this rule.

Report every fit, support, micro accuracy/NLL/Brier, macro metrics, transition
bins, revision accuracy, TRUE/FALSE/DONTCARE counts and paired gains. Keep the
representation-by-head interaction descriptive. Previous V2 output differences
are replication context, not alternative success criteria. A pass justifies
a new confirmation protocol; it does not establish a novel architecture.

## Cost and evidence

Show preparation, fixed capacity work, full encoding, cache bytes, training,
evaluation and reporting separately. Report actual candidate encodings,
unpadded/padded tokens, parameter counts, optimizer work and public/padded
observation tensors. Joint context cannot claim the old shared-turn encoder's
cost. Cached head timing is not end-to-end request latency; do not claim a
speed-quality frontier without a separately measured request benchmark.

The saved-output reporter must authenticate the exact 52-file training run,
29 scientific source files, initialization and batch orders, input/cache
identities and recorded invariants without neural execution or deserializing
weights. Peer review the sources and test causal/permutation/gradient and
metric boundaries before fitting. Preserve raw text and individual predictions
locally; publish aggregate results, receipts, source and figures. Retain
repeated-development, supplied-schema, pretrained-transformer and licensing
limits from the parent study.

## Fixed parent identities

- V2 plan: `9c39c3b27c7219190bc3f45fc342bc4da4eb408e622402a92ce51efb68ed3905`.
- V2 completion: `768376e24824523b9a5c26923c14206e567fda4f3578ed1babdf7cbdfb1d89f4`.
- Feature packet: `e4503c3dd63d28b74e91877dfc9c69b81f4b880e4f07d240cf8c08331fa0b308`.
- Lexical cache: `2e3e528ae05e55a32aa812d25a9dd609d7878e4daff77130994b0a63c32dad54`.
- Prepared data: `677d37ea581b3165b0adc96a0a0daab74271e0434df2ad23ace9568537e2cd12`.
