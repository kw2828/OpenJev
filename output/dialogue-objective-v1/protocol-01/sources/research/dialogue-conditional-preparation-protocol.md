# Conditional observation probe: metadata preparation protocol

This protocol prepares the exact cohort and cache addresses for the
[conditional observation design](dialogue-conditional-observation-design.md).
It does not train, select or evaluate a model. The earlier V2, token-training
and execution-qualification decisions remain unchanged.

## Fixed inputs

Paths below are repository-relative. Each external SHA-256 must match before
its receipt is trusted. Never derive the expected digest from the current file.

| Input | Completion SHA-256 |
|---|---|
| `runs/sgd-state-v1/data` | `677d37ea581b3165b0adc96a0a0daab74271e0434df2ad23ace9568537e2cd12` |
| `runs/sgd-state-v1/features-02` | `e4503c3dd63d28b74e91877dfc9c69b81f4b880e4f07d240cf8c08331fa0b308` |
| `runs/dialogue-copy-v1/lexical-01` | `2e3e528ae05e55a32aa812d25a9dd609d7878e4daff77130994b0a63c32dad54` |
| `runs/dialogue-token-v1/preparation-01` | `094b31d5cf7895ad700d24caa7e3f449dc5781a0f2a5a997b1a4d1262a3d8912` |
| `runs/dialogue-token-v1/capacity-01` | `3de6725bae68443f1326e9fdd1a3506ef355d936bfa98e9ef377aedf6e6f9dbc` |
| `runs/dialogue-token-v1/features-01` | `45d63e97698d1d549ad68c9c331cad2154bb17b9c70535392d00bd78edcd56ee` |

Also bind the token-preparation plan to
`c2120622a0b11f0ffb1a9e4925ca1f2619f193a9f228a6d0728759f62dceadf4`.
The independent token-cache audit at
`output/dialogue-token-v1/cache-audit-01/result-01` has receipt hash
`3752c520ab58ee72e47f8a74fdd0ae99a58670c2bdd4b4803451100514c44808`
and summary hash
`d6053c9cf552c20fb8bdc1400e2d27e1c99a43f47f31777b2d6a7fe5132b0d0c`.
Require the recorded successful cache audit and cross-receipt data, packet,
lexical, preparation-plan, encoder-revision and encoder-file-map identities.
The encoder itself is not loaded or executed.

Use a small independent metadata loader. Existing experiment loaders that
scan floating-point arrays or construct models are outside this protocol.
Authenticate recorded input payload bytes and sizes. Float payloads may be
read as opaque bytes for hashing, and their array headers may be inspected;
do not decode their numerical values. Read only integer index/offset/length
arrays needed to validate cache addressing. Validate complete shape, dtype,
offset bounds, contiguity and closure, not just the selected rows.

Mixed JSON containers can be decoded with the standard JSON parser. Only
whitelisted identity, index, label and annotation fields may affect this
preparation. Embedded text strings are not interpreted, emitted, tokenized or
used to derive features or selection. This is not a claim that mixed-container
bytes containing text were never read. The existing raw train/dev data files
are authenticated as opaque payloads; only their catalog is decoded. Official
test contents and all closed-study checkpoints/predictions remain untouched.

## Whitelist and deterministic admission

Allowed packet schema fields are `id`, `split`, `service`, `slot`, integer
`text` and `candidates` embedding indices, `candidate_ids` and supplied
`candidate_values`. Candidate values may establish the existing exact boolean
ontology classification, never a new text feature. Allowed dialogue fields
are `id`, integer `turns` embedding indices and scored records containing
`query`, `time`, `label`, `bin`, `unseen`, `dontcare`. Ignore `user_text`.
Validate exact schema/candidate identities against the original catalog.
Read lexical and token indexes only for their schema, dialogue order, query
order, offsets, shapes and original pooled-feature indices.

Process train, then dev, in the existing packet dialogue order; within a
dialogue sort scored rows by public USER time and query index. Retain each
row's original scored-record index for provenance. Validate every scored row,
including excluded rows. A row at public USER time
zero is excluded as first-turn. Otherwise require a unique scored predecessor
at exactly time `t-1` for the exact same service and slot in the same dialogue
and split. A gap is excluded, never converted to previous NONE. Duplicates,
malformed labels, invalid IDs or inconsistent metadata fail the attempt.

Resolve the previous label through its own schema to a canonical candidate
ID. Map that ID into the current candidate list. Exclude absent previous IDs
and retain every other eligible row. Never transfer a numeric target index
between schemas. Validate the current label against the current schema. No
model correctness, mention flag, target category or transition group selects
rows. These label-assisted rows define a privileged diagnostic cohort.

For admitted rows, derive unchanged/changed and the five transition groups
from the two canonical IDs. Validate agreement with the inherited annotation
where it uses the same adjacent-state definition. NONE and DONTCARE remain
distinct reserved identities. Record current TRUE/FALSE/DONTCARE membership,
and seen/unseen, for later evaluation only. Previous gold is the sole label
designated as a future actor covariate; current gold and group flags remain
supervision/evaluator fields.

Align every dialogue across pooled, lexical and token indexes. For an admitted
row, resolve its exact current context, token start/stop and length, pooled
context index, query/candidate embedding indices and real-candidate lexical
slice. The original pooled context index must match the token cache's mapping.
Candidate mask construction uses the real supplied candidate count. No schema
pruning, truncation, context subsampling or new encoding occurs.

## Evidence and bounded execution

First freeze the source/runtime closure and pinned receipt metadata into
`output/dialogue-conditional-v1/preparation-protocol-01`. Freeze reads no
packet rows or feature arrays. Bind this protocol, the conditional design,
the new preparation script and tests, and the source-only schema references.
Publish that freeze before the one extraction attempt.

Preparation has one exclusive output directory,
`runs/dialogue-conditional-v1/preparation-01`, a **120-second whole-run cap**
and **256 MiB output cap**. These are ceilings, not runtime estimates.
The clock includes authentication, metadata decoding, validation, writing and
payload hashing. Check caps throughout processing. No retry, resume, cap
extension or replacement cohort is permitted under this protocol. Preserve
the partial ledger and a failure receipt if the attempt fails.

Save a start receipt, a whitelisted schema catalog, one row ledger covering
every scored row and its admission/exclusion reason, an aggregate summary and
a completion manifest. Bind every payload by exact bytes and hash. Record
all input, source, runtime and protocol identities. Publish aggregate counts
and hashes; keep per-dialogue label/address ledgers local.

Summary counts must reconcile all scored rows to disjoint first-turn,
missing-adjacent-predecessor, absent-previous-ID and admitted outcomes. Report
the intermediate adjacent-eligible count as well. Break down train/dev and
seen/unseen, with distinct dialogue/query counts and all admitted transition
and TRUE/FALSE/DONTCARE supports. Empty groups remain empty. Include token
length and candidate-count histograms, total and maximum one-row `tokens *
candidates` score positions, and logical input payload counts. They are work
descriptors, not measured allocation, latency or a training-wall forecast.

No model, encoder, optimizer, task scoring or scientific fit occurs here.
Cache finite-value and pooling validity are inherited from the authenticated
prior audit, not newly recomputed. Before fitting, a separate prospective
training protocol must freeze the admitted-row objective, batch construction,
paired initialization and ordering, nine-fit schedule, numerical checks,
analysis/continuation rule and complete compute/storage limits. Preparation
completion alone does not establish scientific efficacy or training readiness.
