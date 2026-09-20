# Dialogue copy memory: fixed development comparison

Status: pre-execution protocol. Hash this document, implementation, tests, reporter, and input receipts before any fit. The primary proposal is `selective`, fixed before results. This study follows the negative [dialogue-memory comparison](dialogue-memory-results.md); it tests a different representation and update, not a retune of innovation Kalman. Previously inspected official development data is development evidence only. Official test contents remain untouched.

## Question and provenance

Can explicit candidate identity plus learned selective replacement improve interpretation and correction beyond literal copying and a matched scalar replacement rule? Blind exact matching preserves identities but confuses relevance, negation, and yes/no responses. The saved-only [diagnostic](../output/dialogue-copy-diagnostic-v1/README.md) motivated shared lexical observations. Its target-aware oracle is diagnostic, not achievable performance or a model baseline.

Copying, slot-specific memory, and selective overwrite are established by [TRADE](https://aclanthology.org/P19-1078/), [SOM-DST](https://aclanthology.org/2020.acl-main.53/), and earlier Bayesian dialogue tracking. The [source review](dialogue-copy-source-review.md) also discusses a Bayesian odds-filter option. That option is **not implemented in this experiment**. The actual primary below is a discriminative stochastic transition. No novelty, calibrated Bayesian uncertainty, RL, world-model, connectome, or ICLR-readiness claim follows from implementing it.

Reuse the exact prepared [Schema-Guided Dialogue](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/tree/e852981ae34990f4358979625854259302feaa78) train/development subset and frozen MiniLM embeddings from the [previous protocol](dialogue-memory-protocol.md). These are simulated outlines paraphrased by people, licensed CC BY-SA 4.0. The source commit is `e852981ae34990f4358979625854259302feaa78`; MiniLM revision is `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. This is not transformer-free. Encoder pretraining overlap is unknown.

The retained subset contains 2,017 training dialogues with 51,741 scored categorical questions, and 2,363 development dialogues with 62,329 questions. Development has 29,236 seen-service and 33,093 unseen-service questions, with 241 and 203 revisions respectively. Unseen service is defined by training-schema membership, not a guarantee of a new domain. No development clear examples exist. This supplied-schema task grants service/slot routing and a finite candidate set; it is not the official full dialogue-state-tracking task.

Input completion identities:

- Prepared public data: `677d37ea581b3165b0adc96a0a0daab74271e0434df2ad23ace9568537e2cd12`.
- Frozen feature packet: `e4503c3dd63d28b74e91877dfc9c69b81f4b880e4f07d240cf8c08331fa0b308`.
- New lexical preparation: `2e3e528ae05e55a32aa812d25a9dd609d7878e4daff77130994b0a63c32dad54`.

Lexical preparation performs zero encoder or neural calls. Retain raw dialogue text and individual predictions locally; publish aggregate metrics, source, and receipts. Preserve prior preparation failures and their provenance.

## Actor inputs and independent questions

Each actor step sees only the preceding public SYSTEM utterance, current USER utterance, and supplied question/candidate schema. No current SYSTEM reply, future utterance, gold previous state, operation label, annotation span, active intent, or service result is an input. Gold labels and evaluator-defined transition bins are used only for training loss and reporting. Missing supervision does not reset or pause a question's state.

For efficiency, pack each dialogue's unique supplied questions and replay every question over every real public step. Each question is an independent stream. The union of questions requested at later steps must never influence another question. Query-set invariance tests enforce this. This is equivalent to replaying a public prefix when the caller first supplies that query, not a claim that an online system knows future question schemas. Padding alone pauses updates.

Use these ten deterministic candidate observations, shared by all arms except the explicit ablation:

1. Exact bounded match in current USER text.
2. Exact bounded match in preceding SYSTEM text.
3. Unique longest USER match.
4. Unique longest SYSTEM match.
5. Literal register's current value.
6. Literal register's previous value.
7. Reserved NOT_MENTIONED flag.
8. Reserved DONTCARE flag.
9. USER contains `yes`, `yeah`, `yep`, or `true`, for an ontology value exactly `true` after case folding.
10. USER contains `no`, `nope`, or `false`, for an ontology value exactly `false` after case folding.

Matching uses Unicode word boundaries and case folding. USER matches alone update the literal register; the longest unique candidate wins, while ties and no matches carry. Reserved sentinel names never match literal text. An ontology value named `None` remains distinct from NOT_MENTIONED. The boolean cues are noisy shared observations, not a learned or annotated polarity parser, and do not update the literal register. Matching does not understand relevance or negation.

## Five fixed models

Shared 384-dimensional frozen turn, question, and candidate features are projected separately to 64 dimensions with tanh. For each candidate, concatenate the three projections, their three pairwise elementwise products, the ten lexical features, that candidate's previous probability, and normalized belief entropy. A shared 396-to-64 tanh tower produces write and departure logits. Candidate IDs index state only; no learned ID-specific embeddings are allowed. Initialize belief to the supplied NOT_MENTIONED candidate. Valid candidate order is arbitrary and permutation equivariant.

Write probabilities are `w = softmax(write_logits)`, departure probabilities are `r = sigmoid(departure_logits)`, and total released probability is `m = sum(b * r)`:

| Method | Predicted-state update |
|---|---|
| `readout` | Direct candidate softmax; reset learned belief inputs to initial NOT_MENTIONED at each step. The shared deterministic literal register still provides history. |
| `scalar` | `b_next[i] = (1-m) * b[i] + m * w[i]`. |
| `selective` (primary) | `b_next[i] = (1-r[i]) * b[i] + m * w[i]`. |
| `selective_no_lexical` | Same primary transition, zeroing lexical features 1-6 and 9-10 while retaining the two reserved-type flags. |
| `candidate_gru` | Shared candidate tower followed by a per-candidate GRU of width 16 and a scalar score, normalized over valid candidates. It also sees its own prior categorical belief. |

Scalar, selective, and no-lexical arms have identical parameter shapes and initial tensors. Scalar and selective receive the same public observations and use the same formula for released mass. Given the same belief and scorer outputs, they release exactly the same total mass and differ only in the retention term. Their beliefs and learned parameters can subsequently diverge; no-lexical explicitly masks observations. Departure output starts at zero weights and bias -3. Implement the transition stably in log space without an undocumented probability floor. For a one-hot previous belief and the same scorer outputs, scalar and selective are algebraically identical; meaningful gains would require useful uncertainty, not merely copying a known value.

The four non-GRU arms register 99,458 parameters each. The GRU registers 103,411, including the shared 130-parameter output head that is computed but unused by its readout. The readout's departure output is unused. Report both registered parameter counts and whole parameter tensors with final gradients; the latter does not mean every element has a nonzero gradient. This is not a compute-matched comparison with the GRU. Candidate belief state is O(questions times candidates); GRU adds 16 hidden values per candidate. Shared schema projections are cached within forward computation. All arms retain the same deterministic lexical observations, except the declared no-lexical ablation.

## Training, cost, and analysis

Fit all five methods with seeds 4101, 4102, and 4103. Use 20 epochs, batch size 32, AdamW learning rate .001, weight decay .0001, gradient clipping 1, and four CPU threads. Each seed uses the same 20 dialogue permutations for every arm, generated independently of model initialization. Train through predicted state without gold teacher forcing. Weight state cross entropy to give equal total training weight to unmentioned retention, assigned retention, and changes, using selected training labels only. Each fit has 1,280 optimizer updates and 1,034,820 supervised question presentations.

Keep final weights only for evaluation. No best epoch, seed, architecture, or threshold selection is allowed. Preserve all 15 fits and every failure; a failed scored fit is not silently replaced. A full run has 19,200 updates and 15,522,300 supervised presentations. Validation/tests before freeze are synthetic engineering checks, not empirical success.

Recompute micro accuracy, NLL, Brier score, exact transition bins, three-stratum macro accuracy, and revision accuracy on all/seen/unseen panels. Equally average fit metrics across three seeds and display every point. Retain literal carry and always-NOT_MENTIONED references. No probability floor is added in reporting: a zero target probability yields undefined/infinite NLL and fails the NLL check. Empty strata are missing evidence. Three seeds are not three independent datasets.

Record training wall time, encoder/lexical preparation separately, evaluation wall time, parameters, optimizer updates, supervised questions, initialization/checkpoint/prediction hashes, and padded versus real work. `real_turns` counts public dialogue steps once; `real_question_steps` counts those steps times each dialogue's independent query count. Padded positions count B*T, B*T*Q, and B*T*Q*C. Evaluation includes batch assembly and saving outputs, but excludes the shared encoder. These timings do not establish live API latency or an end-to-end speedup. Keep the frozen shared-encoder cost visible.

The reporter reuses explicitly hash-pinned metric helpers from the previous study. It authenticates inputs, source, all 15 saved fit records, prediction membership, and paired initialization without deserializing model checkpoints or making neural calls. Source-level tests cover label isolation, causal prefixes, candidate/query permutation, padded steps, transition arithmetic, finite gradients, and failure handling.

## Fixed continuation rule

Require all 15 completed fits and all six conditions on **both** seen-service and unseen-service development panels:

- Primary mean three-stratum macro accuracy exceeds literal carry by at least 3 percentage points.
- Primary mean macro accuracy exceeds scalar by at least 0.5 percentage points.
- Primary mean micro NLL is no worse than scalar.
- Primary macro accuracy strictly beats scalar in at least two of three paired seeds.
- Primary mean macro accuracy is no worse than both readout and candidate GRU.
- Primary mean revision accuracy is within 1 percentage point of literal carry or better.

Use exact integer-ratio accuracy comparisons and inclusive margins; do not round to obtain a pass. Missing required support or undefined NLL fails. All 13 checks must pass to justify a separately frozen confirmation study. Report the no-lexical ablation even though it is not a gate. A failed primary remains failed if another arm wins. These are practical development criteria, not significance tests, untouched evaluation, novelty certification, or publication acceptance criteria.
