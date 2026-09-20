# Corrected candidate memory: fixed replication protocol

Status: prospective, before any V2 fit. This is a new training study after the
[internal normalization defect](dialogue-evidence-qualification-results.md).
It tests the same scientific question as the original [15-fit protocol](dialogue-copy-protocol.md)
with explicitly normalized state arithmetic. It introduces no new architecture
claim, gate input, lexical feature, training objective, or hyperparameter search.
Keep the original results and failed qualification attempt intact.

## Question and scope

Does selective replacement improve over the matched scalar replacement rule
after both use valid categorical recurrent states? The original experiment's
final output metrics remain reproducible, but its scalar update amplified
internal mass errors. That confounded the intended comparison. Normalizing
old checkpoints at inference would not correct their training history, so fit
every arm again from the original seed-based initialization.

This remains a supplied-service, supplied-slot, finite-candidate subtask of
Schema-Guided Dialogue. The already inspected official development split is
development evidence only. No official test contents, external API calls,
new encoder calls, teacher annotations, architecture selection, or new data
are permitted. A passing development rule would justify a separately specified
confirmation study, not novelty, full DST performance, or ICLR readiness.

## Fixed inputs and actors

Reuse the exact original feature packet, lexical cache, train/dev membership,
public text, question and candidate descriptions, and labels. The source
dataset revision and MiniLM revision, licenses, actor information boundary,
literal register semantics and ten lexical observations remain those in the
original protocol. No gold state, annotation span, operation label or target
availability enters actor features or an update mask. Every supplied question
updates at every real public turn. Only padding pauses an update.

- Original plan: `609a0758f2a4ae4722052bb27740c74b3d1bdac396d8134eb39f5118f1429a2b`.
- Original completion: `8317abad5ef8235f31f5381591944ad152c42fddb98743fa64fd1f5cd8c3f300`.
- Feature packet completion: `e4503c3dd63d28b74e91877dfc9c69b81f4b880e4f07d240cf8c08331fa0b308`.
- Lexical completion: `2e3e528ae05e55a32aa812d25a9dd609d7878e4daff77130994b0a63c32dad54`.

The retained data have 2,017 training dialogues / 51,741 scored questions and
2,363 development dialogues / 62,329 scored questions. Development contains
29,236 seen-service and 33,093 unseen-service questions, including 241 and
203 revisions, respectively. There are no development clear examples.
Unseen service does not guarantee an unseen domain. Raw dialogue text and
individual predictions remain local; publish aggregate results and receipts.

## Five models and one numerical change

Train `readout`, `scalar`, `selective` (fixed primary),
`selective_no_lexical`, and `candidate_gru` through
`DialogueCopyMemoryV2`. The original projection dimensions, candidate tower,
parameters, observations, losses, and update equations are unchanged.
Readout still has the supplied literal register but no learned recurrent
belief feedback. GRU retains width 16. Non-GRU arms have 99,458 registered
parameters; GRU has 103,411, including its unused shared output head.

V2 subtracts `logsumexp(log_b)` before the belief enters features or the
transition, then normalizes each real proposed update. Masked candidates
retain zero support, padding retains exactly the old state, and gradients
remain attached. No probability floor, detached recurrence, mass clipping,
new learned parameter or gold correction is allowed. The explicit version
must appear in model configuration and every fit receipt.

Scalar, selective and no-lexical arms must start from identical parameter
tensors within each seed. All arms receive identical epoch orders. Readout
and GRU remain conventional controls, not parameter/compute-matched models.
Record their full parameter and common-parameter initialization hashes so
the paired starting condition is independently checkable. Before any optimizer
step, each new fit's full initialization hash must equal the corresponding
original fit's saved initialization hash. V2 changes no constructor parameters
or random draws; a mismatch is an implementation failure, not a new seed.

## Fixed training and execution

Use seeds 4101, 4102 and 4103, all five methods, 20 epochs, batch size 32,
AdamW learning rate .001, weight decay .0001, gradient clipping 1 and four
CPU threads. Use the same independently seeded dialogue permutations and
three-stratum training weights as the original study. Train on predicted
states without gold teacher forcing. Evaluate final weights only.

This gives 1,280 optimizer updates and 1,034,820 supervised question
presentations per fit; 19,200 updates and 15,522,300 presentations overall.
Preserve all 15 fits, without selecting an epoch, seed or variant. The whole
training/evaluation command has a fixed 3,600-second wall cap, starting before
input authentication and checked between batches and before terminal success.
Authentication and output costs count. Stop on failure or timeout, preserve
completed and partial work with receipts, and do not resume, replace or
silently retry the attempt. Do not run other corpus/model experiments alongside
the study. Source review and saved-output reporting may proceed independently.

Freeze the runtime, input receipts, original 13 scientific sources, V2 source
and its tests, new runner and reporter with their tests, and this protocol
(20 source files). Publish the plan before fitting. Output directories must
be exclusive. Source/input drift rejects execution; do not edit frozen files
while it runs.

## Observe internal state, not just final softmax

A monitoring subclass may capture detached statistics around the unchanged
V2 update. It must not change actor computation, random state, gradients,
weights, update masks or output normalization. Synthetic tests must verify
identical outputs and parameter gradients with and without monitoring.

During both training and development evaluation, check every real public
update's incoming and outgoing categorical state, including unscored steps.
Check finite nonnegative probabilities, masked zero support and absolute
mass error at most `2e-6`. Where present, released mass must be finite and
within `[0, 1+2e-6]`. Record maxima, checked counts and any allowed roundoff
above one separately for training and evaluation. No clipping or repair by
the monitor is allowed. The V2 normalization is part of the trained model.

Record audited batched question positions separately from real supplied
question steps if padding introduces dummy question streams. Derive expected
coverage from actor shapes and public turn masks, never gold label masks.
Retain work counts for real turns, real question steps and padded B*T,
B*T*Q and B*T*Q*C positions. Invariant failure stops the study even if the
eventual softmax outputs would be valid.

## Analysis and continuation

Authenticate all saved fits, initialization, checkpoint/prediction hashes,
configuration, exact artifact membership and internal-check coverage before
reporting a complete experiment. The reporter must not deserialize weights,
replay models or fit anything. Preserve every seed point and report micro
accuracy, NLL, Brier, three-stratum macro accuracy and revision accuracy on
all/seen/unseen panels, as well as the individual transition bins and absent
support. Keep literal carry and always-NOT_MENTIONED references. A zero target
probability gives infinite NLL; do not hide it with a floor.

Retain the original 13 scientific continuation checks unchanged: all 15
fits complete, plus each of these six on both seen and unseen panels:

1. Selective mean macro accuracy exceeds literal by at least 3 percentage points.
2. Selective mean macro exceeds scalar by at least 0.5 percentage points.
3. Selective mean micro NLL is no worse than scalar.
4. Selective strictly wins macro in at least two of three paired seeds.
5. Selective mean macro is no worse than both readout and candidate GRU.
6. Selective mean revision accuracy is within 1 percentage point of literal or better.

Use exact integer-ratio accuracy comparisons and inclusive margins, without
rounding or epsilon to obtain a pass. Missing support or undefined NLL fails.
Technical validity, including internal-state checks, is a separate prerequisite;
do not turn extra engineering checks into additional scientific successes.
A failed primary remains failed if another arm wins. Historical V1 differences
are descriptive implementation comparisons, not additional success criteria.

Report training/evaluation time, parameters, optimizer work, supervised
presentations and internal audit work. Show the shared prior encoder and
lexical preparation costs separately. Timing with instrumentation is not a
serving-latency benchmark. Fitting on this repeatedly exposed development
program cannot supply untouched confirmation, even with a frozen protocol.
