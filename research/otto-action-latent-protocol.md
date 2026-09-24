# Predicting through genuine observation gaps

Prospective development pilot, OpenJev. Freeze implementation, seeds, qualification
and this protocol before native collection. Earlier TEST and confirmation panels
remain closed. No Astra requests, retries, replacement cases, hyperparameter
search, intermediate checkpoint selection or claims of architectural novelty.

## Question and dataset

Does a small action-conditioned transition model predict useful future outcomes
when refreshed observations are unavailable? The old GRU already saw actions and
observations. Its existing P4 readout cannot consume this new task unchanged
because it expects fresh public features. This is a different capability test.

Use the authenticated original OTTO TensorFlow teacher and sampled-source
simulator, with 53x53 grid, four hit categories, R_dt=2, Euclidean Poisson sensing,
and the existing public belief filter. Collect 288 independent originating
cases: TRAIN192 at lambda3, DEV48 at lambda3 and DEV48 at lambda4. Exact seeds
are320000001..320000192,321000001..321000048,322000001..322000048;
fit seeds323000001..323000003. Initial hit cycles1,2,3. The scoped metadata seed
review excludes this proposal, reports zero exact historical matches and does
not establish globally unused numeric ranges.

Every case uses eight analytic-policy transitions with observations, giving nine
prefix feature rows including reset. Cases found during that prefix are recorded
and excluded without replacement. Require at least120 TRAIN and24 DEV per regime
before fitting, or close the attempt as failed.

At the prefix, precommit the entire uniform action sequence using a separate
PCG64 SeedSequence([case_seed,911]) stream. TRAIN blocks have four actions; DEV
blocks have eight. All possible blocks remain geometrically legal because the
prefix starts at the grid center. Persist the commitment before the first block
transition. Actual future odors cannot affect these actions. Each originating
case contributes one block to one split.

Targets are the outcome after each action: odor0..3 or absorbing-found class4.
Native found is hit=-2, never odor0; no post-terminal stepping or teacher calls.
After found, retain class4 for remaining horizons and zero masked feature/cost
storage. For surviving states, collect current public features and original
four-action teacher costs. Label-producing future state updates never enter a
blind predictor. Teacher labels are annotation costs, not free observations.

## Models and matched information

All neural methods see the same31-feature observed prefix. Blind rollout receives
only that prefix and its precommitted action IDs. Normal-observation rollout
predicts each next outcome and cost **before** assimilating that step's features;
those features can affect later predictions. These are forecasts of the teacher's
future observed-state decisions, not a policy evaluated by autonomous control.

* Candidate: 28-state GRU assimilation G and action-conditioned GRU transition F,
  with separate five-outcome, four centered-cost, and two auxiliary heads.
* Proposed-action-blind control: same network and parameter count, but zero the
  proposed action input to F. Past observed action information remains available.
* Direct horizon control: same G and heads, shared positional action-token MLP
  and causal prefix pooling followed by a horizon decoder. It has no learned
  action recurrence. All learned weights are active at training horizons; there
  are no new heads or input-slot weights exclusive to horizons5..8. Positional
  pooling need not preserve every action-sequence distinction.
* Solved ridge reference: all nine prefix rows, current observed context, fixed
  causal action-position moments, last proposed action and horizon,336 features.
  Fit categorical one-hot and centered-cost targets with fixed ridge1e-4,
  penalizing every coefficient. Project categorical estimates onto the simplex
  and mix in1e-6 uniform mass. It uses3,024 coefficients. This is a transparent
  history-feature reference, not the earlier fitted P4 readout.

Neural parameter counts are8,299,8,299 and8,107. The action-blind control's336
proposed-action input weights receive zero input. All learned methods have the
same targets and cases. Ridge is fitted once; its predictions are reused in
three paired-seed comparisons without being called independent fits.

## Frozen optimization

Train each neural arm for80 epochs, Adam0.003, batch32, gradient clipping5.
For each fit seed reset PCG64(seed) to produce identical epoch permutations across
arms. Rotate arm execution order across seeds. No learning-rate schedule or
selected checkpoint. All nine fits and the deterministic ridge solve finish
before first DEV decoding.

Average blind and normal objectives equally. Each uses mean categorical cross
entropy plus case-weighted cost MSE plus0.1 times auxiliary MSE. Cost targets are
all-four centered teacher costs divided by64. For each case average squared
costs over actions and its surviving target rows, using zero for no survivors;
then average cases. Set cost_scale to the float32 square root of that TRAIN-only
mean with variance floor1e-6. The network emits cost_scale times its centered
head, and cost MSE is divided by cost_scale squared. Zero-initialize cost heads.
This common fixed standardization balances units without a DEV-dependent search.
Auxiliary targets are future entropy and distance feature columns19:21 and use
the same survivor/case weighting. Terminal rows contribute outcome loss only.

Ridge sees both normal and blind training views equally, using the same per-case
cost weights and a uniform categorical Brier-target row objective. This differs
from neural cross entropy and is reported explicitly. Normal prediction freezes
the model or ridge descriptor after observed found; future actions do not change
an already-terminal prediction. Blind prediction never receives found labels.

## Continuation rule

Primary measurements are case-weighted categorical log score and teacher-action
cost gap at horizons5..8. For every fit seed, each DEV regime and all three fixed
controls, require at least1% lower log score and5% lower decision gap. Require
normal-observation all-horizon log score and gap to worsen by no more than1%
against each control. All18 cells must pass, with at least16 cases with surviving
decision targets in each compared group. An undefined or zero comparator gap
cannot establish a strict relative improvement. Report all failures.

Decisions choose the lowest-index legal minimum predicted cost. Cost gaps are
computed from the original teacher costs. First average surviving rows within an
originating case, then supported cases; disclose excluded cases and full-case
zero-contribution averages separately. Outcomes include all absorbing suffixes.
Report horizons1..8, short1..4, long5..8, Brier, all case/support denominators and
actual compute. Fit seeds share evaluation cases and are not independent samples.
No bootstrap significance, calibration guarantee, control gain or paper claim.

## Bounds and verification

Native collection cap1,800 seconds; fitting plus prediction1,800 seconds;
independent saved-output audit180 seconds. Each has4GiB RSS and512MiB output.
Single CPU numerical threads. The unchanged suspend-aware process supervisor
owns the deadline and process group; receipts must join the original launch,
terminal status, reaping and absent group. Failed attempts remain closed with
partial evidence, never repaired by rerunning under the same registration.

Source qualification uses fabricated cases, causality interventions, terminal
semantics, loss weighting, numerical regression and independent scalar metrics.
Native preflight authenticates prior runtime, source and input identity without
importing numerical frameworks. Collection retains calls, commitments, public
transitions and belief parity witnesses. Publication requires all original
process closures plus a separate audit of saved predictions and training orders,
with no model, teacher, native or optimizer calls in the audit.

Charge collection, annotation, feature construction, fitting, serialization and
inference separately. Timings include the declared implementations and checks;
they are not general hardware speed claims. A pass only motivates a subsequent
planning experiment. A fail is evidence about this registered small pilot.
