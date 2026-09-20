# Qwen observation strength with a known previous value

Prospective protocol. Publish source, prompt, input, runtime and cost bindings
before their corresponding execution. The pilot requires authenticated completed
preparation; full inference additionally requires an admitted completed pilot.
No earlier failure is changed by this study. The allocations below
are chosen before any timing from this experiment, not extensions of a running
or failed attempt.

This is a frozen-model semantic baseline, not a new architecture, calibrated
filter, RL method or autonomous memory experiment. It asks whether the compact
MiniLM observation models were weak practical baselines, and whether a short
public history helps when the immediately previous categorical value is already
supplied correctly. Both positive and negative results remain conditional on
that privileged previous value.

## Fixed cohort and two arms

Use every one of the **7,819** primary held-out-service rows from the existing
official-TRAIN development split: **578 changed**, **7,241 retained**, including
**4,032 unmentioned** and **3,209 assigned** retentions. Preserve exact row IDs,
candidate identities and the six-service membership. These examples have been
repeatedly exposed in earlier development; they are not fresh confirmation.
Do not read official DEV or TEST for this study. Unknown overlap with Qwen's
pretraining also prevents claiming that its service schemas were unseen.

| Arm | Public observation text |
|---|---|
| `current` | The exact preceding SYSTEM/current USER pair used by the existing observation cache |
| `history4` | That pair and up to three preceding public USER exchanges in chronological order |

Select history using the complete public USER timeline, including turns without
scored annotations. Each exchange is that USER utterance and the immediately
preceding raw turn if its role is SYSTEM, otherwise an empty SYSTEM string.
Take indices `max(0,t-3)..t`; do not search backward for a scored or same-service
turn. Preserve exact utterances and role boundaries. Never include a later turn.
Serialize the pairs as the fixed JSON object `exchanges_oldest_first`, with
`SYSTEM` and `USER` strings in each pair. The current arm contains one pair;
history4 contains up to four. This public group context is shared across its
supplied questions, not augmented with other slots' annotations or answers.

Both arms receive the same supplied service/slot query, all current candidates,
public candidate types and ten inherited lexical flags. Candidate types are
NOT_MENTIONED, DONTCARE, TRUE, FALSE and OTHER, determined by the exact public
candidate-ID mapping in `make_question`, never by the current target. Inherited
`literal_current` and `literal_previous` describe a causal full-prefix register;
therefore `current` is not a history-free arm. Append the same fixed, explicitly
named ten flags to each candidate description and put their neutral legend in
the question. Flags are heuristic observations, not verified semantic truth.
Only the public observation text changes between the two arms.

The question also supplies the correct immediately previous value, resolved by
exact canonical ID into the current candidate set. Keep the reserved
NOT_MENTIONED and DONTCARE states distinct from literal ontology values such as
`None`. The exact `TASK` constant in
[`dialogue_qwen_observation.py`](../src/openjev/research/dialogue_qwen_observation.py)
is part of the frozen source and reads:

> What is the user's currently committed value for this slot after the final USER turn? Use the supplied previous value as the state immediately before that turn. Update it only when the dialogue supports a change; a SYSTEM proposal alone is not a user commitment. NOT_MENTIONED means no constraint has been stated; DONTCARE means the user explicitly has no preference. Literal ontology values, including a value named None, remain distinct from these reserved states. Lexical flags are noisy public string-match observations, not answers. The literal register carries the last unique longest USER mention across the public prefix; it does not understand negation or relevance.

Append the exact catalog query text, previous candidate text and fixed lexical
feature-order legend as implemented by `make_question`. The existing scorer's
system instruction treats descriptions and dialogue as evidence rather than
instructions. No examples, per-service prompt variants, paraphrase search or
output-dependent revisions are allowed.

Use the existing local model and revision from `src/openjev/decisions.py`:
`mlx-community/Qwen3-4B-Instruct-2507-4bit`, revision
`50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`.
Use MLX only, with existing cached files. Missing weights or incompatible runtime
fail the attempt; do not substitute the CPU 0.6B model or download new weights.
There is no fitting, sampling seed, generated answer, temperature calibration or
checkpoint selection. The full scientific run makes **15,638 decisions**.

## Actor isolation and model-free preparation

Create a new exclusive preparation directory. The preparation attempt is limited
to **180 seconds**, **2 GiB process-lifetime peak RSS** and **512 MiB output**,
with no model initialization, encoder or model forward call. Local pinned
tokenizer use is allowed. Record whole preparation time, peak RSS, token workload
and all input/source/model/tokenizer hashes.

Authenticate the completed metadata/split chain, original prepared public TRAIN
dialogues, schema catalog, lexical cache and existing comparison summaries.
The published machine-readable plan must supply their exact receipt and payload
pins. The original evaluation-row payload is SHA256
`56e572f6df34cf81ccc38db137d82a699eb3f4f84509af39682d7d963e62a4d0`;
derive the primary subset from its fixed service membership, never from model
predictions, correctness, confidence, loss, literal mentions or rare categories.

The actor assembly API accepts a whitelist of public utterances, supplied query
and candidates, inherited lexical flags and the mapped previous value. It must
not accept current target, transition bin, state frames, slot-value annotations,
dialogue acts, service actions, old predictions or error-diagnosis prose. Keep
scoring targets in a separate evaluator payload. The previous gold value is the
one explicit privileged annotation allowed in the actor.

Use safe local candidate IDs `c00`, `c01`, and so on, with a saved bijection to canonical
candidate IDs. Preserve the scorer's deterministic description sort and verified
unique single-token A-L answer labels. Candidate counts are **4, 5, 6, 7 or 11**;
no cap extension is needed. Verify identical ID-to-letter mappings between arms.
Compute all final prompts and exact token IDs before the pilot. Enforce the
existing 12,000-character context, 2,000-character question and 1,000-character
candidate-description limits, and **4,096 tokens per complete question**, without truncation.
Any invalid or overlength row fails complete coverage; it is not dropped or
replaced. A subsequent template change would require a new prospective version,
not continuation of the same attempt.

Group questions only at the same `(dialogue_id, public_user_time)`. Within each
group sort by original row ID and partition into chunks of at most
four questions. Process groups in dialogue-ID/time order, with current then
history4 for even zero-based group indices and the reverse for odd indices.
The observation context is shared within the group; previous value and lexical
information remain question specific. Freeze every batch's row IDs, candidate
mapping, complete token sequences, lengths, padded input positions and execution
order. Use the existing `SharedPrefixExperiment.logits(method="batch")` path:
full prompts are batched and read at each last real token. No shared-prefix speed
claim, continuation past padding or alternative serving-arm sweep is included.

The preparation plan must bind all eleven local model snapshot files:
`added_tokens.json`, `chat_template.jinja`, `config.json`,
`generation_config.json`, `merges.txt`, `model.safetensors`,
`model.safetensors.index.json`, `special_tokens_map.json`, `tokenizer.json`,
`tokenizer_config.json` and `vocab.json`. Reuse and content-check the existing
shared-prefix model manifest. Before pilot or full-run model loading, verify the
exact file membership, revision, every file hash, source closure, runtime and
prepared request hash. Recheck these identities at phase end. Merely finding a
snapshot directory with the right revision name is insufficient.

The preparation entrypoint is `scripts/prepare_dialogue_qwen_observation.py
--out PATH`. It writes `requests.jsonl` and separate `labels.jsonl`, a `plan.json`
binding both descriptors, and preparation receipts. The runner entrypoint is
`scripts/run_dialogue_qwen_observation.py pilot --prepared PATH
--plan-sha256 PIN --out PATH`; full inference uses the `run` subcommand with
the same arguments plus `--pilot PATH --pilot-sha256 PIN`. The runner never
opens, hashes or decodes `labels.jsonl`; its descriptor is bound transitively
by the pinned plan. Only saved-output reporting accesses that evaluator file.

## Cost pilot and admission

After preparation and source review, select the union of the **first twelve
canonical dialogue/time groups** and **each arm's longest-token group**. Score
both arms and every question chunk for every group in this union. Canonical
group order is dialogue ID ascending then public USER time ascending. For each
arm, find the request with the largest maximum complete-question token length;
break ties by `(dialogue_id, numeric_public_user_time, request_id)`, then include
its entire group.
Preserve prepared execution order. Freeze this selection from token metadata
before any forward call. Do not select by labels, subgroups, old model outputs
or observed pilot behavior.

The pilot cap is **300 seconds**, **12 GiB process-lifetime peak RSS**
and **512 MiB output**. Begin only when
the allocated numerical resource is available. Include loading, authentication,
assembly, synchronization, full-vocabulary readout, finite-logit validation and
receipt I/O. Record every attempted and returned call and batch duration. Do not
load scoring labels, compute quality or retain quality-bearing pilot outputs.
The pilot constructs and JSON-encodes the same candidate readout for matched CPU
cost, checks finite logits and normalization, then discards all choices/logits/
probabilities. Save only identities, work, timings, serialized byte counts and
numerical-validity witnesses. There are no unrecorded warmup calls.

For each arm, let `P_a` be the full run's total padded complete-prompt token slots.
For every pilot batch `b`, record positive slot count `P_ab` and synchronized
batch seconds `t_ab`, including token verification, inference/readback,
float64 readout, validation and JSON encoding. File writes are outside this
request timer but inside whole-phase wall. Define
`r_a = max_b(t_ab / P_ab)`. The admission estimate
in seconds is

```
ceil(2 * sum_a(r_a * P_a) + pilot_whole_wall_seconds + 60)
```

The factor two and extra whole-pilot wall charge provide deliberate headroom;
the 60-second allowance covers additional full-run finalization and output work.
This remains a heuristic, not measured full-run latency or a guaranteed bound:
attention, batching, KV storage, thermal state and file I/O are not linear in
token positions. Publish full and sampled batch counts, token distributions,
maximum lengths and device memory observations alongside the estimate.
The alternative maximum-request-duration times full-request-count extrapolation
may be reported descriptively; it is not substituted for, or maximized with,
the frozen token-rate admission formula.

Admission requires complete valid pilot coverage, resource compliance and an
estimate **at most 7,200 seconds**. Pilot failure or denied admission is terminal
for this version: preserve its partial ledger and receipt; do not retry, shrink
the cohort, select a favorable rate or run only an arm. No quality conclusion
follows from cost admission or rejection.

## Full inference and saved scoring

If admitted, execute both arms once in a separate exclusive output under a
**7,200-second whole-run cap**, **12 GiB process-lifetime peak RSS** and
**512 MiB output**. Include loading, authentication, tokenization/assembly, model
calls, synchronization, validation, saving, hashing and finalization. Record MLX
active/peak/cache memory separately from RSS. Preserve every attempted/returned
batch, all row identities and the exact prepared order. No retry, resume or
replacement call is allowed after failure.

A successful pilot contains `started.json`, `plan.json`, `timings.jsonl` and
`completed.json`. A successful full run adds `scores.jsonl`. Keep logs outside
these exact-manifest directories. Failure receipts preserve the request,
attempted/returned call counts and complete partial prefix; late failure must
not leave a successful terminal receipt. Whole-phase caps remain enforced even
if the projection was optimistic.

The full run deliberately scores the fixed pilot rows again. Charge pilot and
full-run costs separately and combined; this repeated cost measurement is not
an extra quality replicate, a repaired prediction or an output-dependent retry.
Do not release partial quality results. A failed prefix cannot support the
complete-cohort comparisons below.

Save raw supported-label logits, mapped choices, vocabulary mass, input lengths
and work/timing records. For reporting, reconstruct each candidate log probability
in float64 as `z_c - logsumexp(z_supported)`, at fixed temperature one. Select
the first exact maximum in the frozen description order, then map back to the
canonical candidate ID. Padding is excluded. Do not floor target probabilities,
fit corrections or select a readout. Record normalization and finite-value
checks. Candidate vocabulary mass is a separate diagnostic, not a confidence
calibration or rejection rule.

The score target is the saved **current categorical value**, conditional on the
supplied previous gold and each arm's public inputs. NLL and multiclass Brier are
proper scores of this declared categorical forecast; the restricted label
softmax is neither the probability that a real-world action succeeds nor evidence
that confidence is calibrated. Report all, changed, retained, unmentioned and
assigned strata, per-service counts, selected-branch/value errors, rare-type
recall/false positives and exact paired repairs/harms. Preserve undefined rates.

## Two separate scientific conclusions

The fixed external semantic comparator is **historical corrected flat**, averaging
its three optimizer-seed metrics equally. Its pinned summary, receipt and audit
are those already specified in the
[objective protocol](dialogue-objective-protocol.md). Keep all three comparisons
visible. The single Qwen prediction per row is not three independent runs, and
the historical control is not parameter-, compute- or pretraining-matched.
Display other completed frozen MiniLM readouts for context without selecting a
more convenient primary comparator. Carry-previous and the unchanged literal
reference remain descriptive accuracy controls with no fabricated probabilities.

Evaluate two prespecified contrasts: **current minus corrected flat** for semantic
strength, and **history4 minus current** for added public history. Each contrast
has one joint behavioral continuation rule: changed accuracy improves by at least
**2 absolute percentage points** and retained error does not increase, under both
row and equal-service weighting. Equal-service metrics average the supported
services within the specified stratum. For the external comparison, average the
three baseline seed metrics before subtraction. Use full-precision counts and
rates with inclusive boundaries. The two decisions are reported separately;
history cannot retroactively pass the current-input comparison.

Report proper-score evidence separately: for each same contrast, overall NLL and
Brier must both be no worse under row and equal-service weighting to call the
forecast scores non-regressing. Show every component and both changed/retained
score differences, not merely this joint flag. A behavioral pass with a proper-
score regression supports a decision-quality result only. A loss-only gain does
not pass behavioral continuation. Neither flag establishes calibration, and no
combined architecture-success gate is created.

The **29 changed TRUE targets**, **five changed DONTCARE targets**, and absence of
changed FALSE targets are mandatory descriptive limits, not additional gates.
The same dialogues contribute multiple rows; service tables and paired
repair/harm counts are descriptive rather than independent-trial significance
tests. Do not choose prompts, confidence thresholds or future tests from a
favorable rare-category count.

## Required focused tests and interpretation

Before freeze, synthetic tests must show that poisoning current targets, bins,
frames and future utterances leaves prepared actor requests unchanged; changing
the allowed previous value must change its own mapped input. Test full public
time indexing across unscored turns, history cutoff, candidate permutation with
identical description-to-letter/ID mapping, reserved-state distinctions, all
candidate coverage, exact ties and last-real-token selection under unequal
lengths. Invalid identities, overlength input and nonfinite logits must fail
without dropping a row. Test the fixed pilot selection, projection boundaries,
complete-call accounting, resource failure receipts and exclusive outputs.
Numerical model checks use only tiny synthetic fixtures, never this cohort.

A current-input gain makes the frozen Qwen scorer a stronger practical semantic
baseline. A history gain supports additional public evidence beyond the previous
gold and literal register. It does not identify a learned recurrent mechanism.
If both fail, representation, prompting, annotation ambiguity and context
sufficiency remain possible explanations; failure is not proof that recurrence
is required. An autonomous comparison without gold previous state, followed by
fresh evaluation, remains necessary before any memory or architecture claim.
