# Paired action-effect supervision

Prospective OpenJev development study, `otto-action-effect-v1`. Freeze this
protocol, source closure, input hashes, qualification and seed allocation before
fresh collection. Prior TEST and confirmation panels remain closed. No Astra
calls, model search, replacement cases, retries, early stopping or checkpoint
selection. A technical, budget or scientific failure is terminal for this
registration.

## Question and scope

Can explicit signed action-effect supervision improve a compact recurrent
predictor beyond giving it the same paired action branches? The preceding
probability-distillation study improved forecasting relative to sampled labels,
but failed its gate. Its shifted action-effect error exceeded the action-blind
zero-effect reference and the direct model. This study targets that measured
weakness while retaining ordinary forecast and decision quality checks.

This is a privileged-label training-objective experiment on the unchanged
28-state model. It is not a new architecture, connectome result, calibration
guarantee, Bayesian weight inference, RL or autonomous-control result. Outcome
forecasts and teacher-cost decisions on fixed trajectories are the endpoints.

## Reused TRAIN and fresh DEV

Reuse only the closed parent's 1,046 lambda3 TRAIN cases, horizon4:
`output/otto-belief-distillation-v1/collection-01/train.npz`, SHA256
`add8cf6ff79733c6fcc1293bf2eb2e538ccb6b52ccd89afbf1ce5306f0faecb6`.
The parent closure is pinned to
`6a72021165661ab60a33809ba857e7f304ba64849160374c713597e1c73e6eaf`.
Authenticate that closure, original collection terminal/receipt, summary and
TRAIN bytes before use. The original TRAIN contains factual, normal and opposite
oracle labels. Earlier fits used only factual/normal labels. All arms here get
the same already-collected opposite labels. No old DEV array is decoded or
reused for fitting or this study's evaluation. Registration records the source
of the development hypothesis in the prior published report.

Charge the whole parent acquisition as shared upstream cost:306.72326325seconds,
18,026native steps and5,460teacher annotations. This deliberately includes the
parent's unused DEV acquisition. Do not claim these labels are free or compare
new training cost against a baseline that excludes data acquisition.

Collect only fresh DEV,128 starts at lambda3, seeds328000001..328000128, and128
at lambda4, seeds329000001..329000128. Fit seeds330000001..330000003. Initial
odor cycles1,2,3. Seed review01 scans the declared historical scope for all259
seeds and their million blocks; it does not prove global seed uniqueness.

Use the authenticated original OTTO simulator/teacher and unchanged trajectory
and predictive-label construction from the parent protocol. Each case has eight
observed analytic-policy transitions, nine prefix feature rows, then a
precommitted eight-action block from PCG64(SeedSequence([seed,911])). Exclude and
log prefix-found cases without replacement. Require at least64 retained cases
per regime and48 decision-supporting cases per gate panel. One originating case
contributes one block. Found4 is absorbing; subsequent native and teacher calls
are skipped. Preserve all suffix rows in outcome and effect scores.

The source prior, shadow Bayesian filter, 53-bit categorical conversion,
Euclidean Poisson sensor tables, native draw validation, normal assimilation,
blind unique-visited-position accumulation and opposite mapping a XOR1 are
unchanged. Save the same16-array schema and sensor tables so the independent
audit can reconstruct labels. Hidden source coordinates are audit-only, never
model inputs or predictive targets. No alternate outcomes or observations are
invented. Fresh collection emits dev.npz and opaque TRAIN-lineage metadata only.

## Models, exposures and objective

Four arms and three fits each,12fits total:

* effect_recurrent: action-conditioned GRU transition/assimilation,28state,
  8,299parameters, with explicit signed-effect loss.
* paired_recurrent: identical model/initialization and paired examples, effect
  loss weight zero. This isolates the objective from additional labels.
* paired_blind: same8,299parameters; proposed action inputs are zero, with336
  inactive weights. It retains past observed actions.
* paired_direct: unchanged causal positional action pooling and direct horizon
  decoder,8,107parameters, with the same paired supervision.

All share31-feature prefixes and frozen model source. Each batch computes three
rollouts: factual blind, opposite blind, and factual normal with predict-before-
assimilation semantics. Blind inputs are only prefix, lengths and proposed
four-action block. Full beliefs, probability targets and costs are targets only.
Opposite rollouts get no invented future features, observations or cost targets.

Use80epochs, Adam0.003, batch32, clipping5, no schedule. Share initial seeds and
per-epoch permutations, rotate execution order by seed index, and fit all12models
before the first fresh DEV array decode. Use final checkpoints. Every fit has
identical originating-case exposures and three rollouts per batch. Controls
compute the same effect graph multiplied by zero; actual wall time is measured,
not presumed identical.

Write CE(A), CE(O), CE(N) for soft-target categorical cross entropy on factual,
opposite and normal branches. Outcome objective is
0.25 CE(A) +0.25 CE(O) +0.5 CE(N). Each is averaged over all cases and horizons.
Only normal rows AFTER already observed found have zero loss, keeping the full
denominator. Factual/opposite suffixes are always scored, even after factual
found. The first found prediction in normal mode is scored.

Cost/auxiliary objective is0.5 QAux(A) +0.5 QAux(N). QAux equals normalized
teacher-cost contrast MSE plus0.1 auxiliary-feature MSE on columns19:21. Center
all four costs and divide by64. Average surviving rows within each case, then
all cases, retaining unsupported cases with zero contribution. Cost_scale is
float32 sqrt(max(TRAIN centered cost variance,1e-6)); the cost head emits in
those units and MSE is divided by scale squared, exactly as in the parent.

For each case and horizon define signed model effect
Dq = softmax64(logits_A) - softmax64(logits_O), and signed teacher effect
Dp = oracle_A - oracle_O. Let E be mean(sum_classes((Dq-Dp)^2)), averaging every
TRAIN case and horizon. Let V be max(mean(sum_classes(Dp^2)),1e-6), computed once
from TRAIN. Add0.1 E/V only for effect_recurrent; other arms add0 E/V while
retaining the same computation graph. V is a variance, not its square root.
No per-case ratios, signal filtering or DEV-derived scales. Persist V, its
pre-floor value, cost scale and source labels.

Inference remains float64 softmax without smoothing. Normal forecasts only AFTER
previously observed found are replaced with exact one-hot found for every arm.
Gap/opposite prediction cannot accept realized future outcomes. Impossible zero
prediction on positive teacher support fails visibly.

## Prospective mechanism gate and unchanged legacy gate

All18 comparisons (three controls x three fit seeds x two regimes) must pass:

* At horizons5..8, at least10% lower equal-case signed-effect error E.
* At horizons5..8, at least5% lower teacher-action cost gap.
* Long sampled log loss cannot increase more than1%.
* Normal all-horizon sampled log loss and decision gap each cannot increase
  more than1%.
* At least48 supported originating cases in both long-gap and normal decision
  panels; effect error retains every originating case, including zero signal.

A zero comparator E or decision gap cannot establish a strict relative gain.
This is a new prospective mechanism gate because action effects are now the
primary target. It does not revise or relabel the parent's failed criterion.
Also report the unchanged legacy gate separately: at least1% lower long sampled
log loss and5% lower long decision gap, same normal tolerances/support, all18
cells. No selection between gates after results.

Select the lowest-index legal predicted minimum cost. Average decision gaps over
surviving rows within case and then supported cases. Report unsupported cases,
the full-case zero-contribution alternative, NLL, Brier, oracle KL, horizons1..8,
short1..4, long5..8, and compute. Report oracle action signal S, JS/TV and signed
model error E from all prefixes and both branches. No E/S ratio controls
admission. Fit seeds share DEV cases and are not independent evaluation samples.

## Bounds, qualification, independent audit

Each collection, fit and audit process has a1,800second suspend-inclusive cap,
4GiB RSS and512MiB output. Single numerical CPU threads. Fresh maximums:
256resets,4,096native steps,2,048teacher calls,4,352feature/analytic calls,
256backend setups; extra fixed caps are6,144likelihood lookups,4,096shadow
updates,2,048normal-oracle predictions,256gap and256opposite predictions,
512planned-position blocks,256source initializations/normalizations,
4,352draw validations and2sensor table builds. Upstream reuse costs stay visible.

Qualification uses fabricated examples only, including signed gradients on both
branches, CE/terminal denominators, unchanged cost weights, zero-signal cases,
shared inputs, independent scalar metric comparisons and full-size capacity
projection. Native preflight authenticates source/runtime metadata without
framework import, environment stepping or neural calls. Pin NumPy conversion
source evidence, qualification artifacts and seed/source reviews.

Audit closed collection/fit processes before loading only freshDEV, predictions
and sensor tables. Independently reconstruct sensor laws, prefix beliefs,
factual/normal/opposite oracle labels, scores, effect errors and both18-cell
gates. Verify12checkpoint hashes,960matched epoch permutations, exact parent
TRAIN lineage, three-rollout counts, target-only projections and fixed weights.
Recorded TRAIN scales are authenticated and checked against their saved
pre-floor values; the audit does not independently recompute TRAIN scales or
rerun fitting. This limitation must remain explicit. No neural, native, teacher,
optimizer or solver calls during audit. Separate closure verifies the audit's
own original terminal receipt, reaping and absent process group. Publish raw
receipts, source and checkpoints, including failed engineering attempts.
