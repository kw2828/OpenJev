# Conditional cost labels for a fixed recurrent decision model

Prospective development study `otto-conditional-label-v1`. Freeze this protocol,
source closure, fresh seed roster, fabricated qualification and budgets before
empirical collection. Earlier TEST and confirmation panels remain closed. No
search, replacement cases, scientific retries, early stopping or checkpoint
selection. A technical or budget failure closes this registration as failed.

## Question and scope

The preceding audited diagnostic resolved reference headroom on fixed horizon8
teacher-cost decisions. It did not improve a model. Can averaging conditional
cost targets improve the same small GRU under matched optimizer work, including
under a sensing-regime shift?

This isolates training-target variance reduction. It is not a novel architecture,
TypeSafe RLCD implementation, outcome calibration, reinforcement learning,
connectome advantage or an autonomous-control result. Both arms start from
scratch; prior checkpoints and prior TRAIN/DEV arrays are not decoded. Its
fixed horizon8, unconditional cost-only objective differs from the preceding
surviving-horizon multi-objective losses, so results cannot revise old gates.

## Fresh data and conditional law

Attempt512 TRAIN prefixes at lambda3, seeds336000001..336000512. Attempt128 DEV
prefixes at each of lambda3 (337000001..337000128) and lambda4
(338000001..338000128). Initial hit cycles1,2,3. TRAIN requires256 retained
prefixes and DEV64 per regime; prefix-found exclusions are logged without
replacement. Lambda4 never enters training and is the declared scenario shift.

Retain all nine31-feature rows from reset through eight observed analytic-policy
transitions. Commit eight subsequent actions with
PCG64(SeedSequence([environment_seed,911])). Do not execute native continuations.
The prefix contains public history; latent source coordinates remain audit-only.
Inputs are only prefix, lengths and committed actions. Full beliefs, draws,
teacher costs and future odors are targets/audit records, never model inputs.

Maintain the strict public posterior and original legacy teacher filter
separately. Apply the authenticated53-bit root conversion once to the strict
prefix posterior (total variation <=1e-10), then the same effective53-bit sensor
law as the closed diagnostic. Counterfactual histories follow this declared
numerical reference law; they are not actual environment rollouts. Save all
source-plus-eight-odor integer draws before any terminal shortcut. Found is
absorbing. Evaluate the original teacher on each surviving branch's own legacy
belief at horizon8 only; never evaluate it on the average belief. Found costs
are exactly four zeros, and all four actions are legal on retained interior paths.

TRAIN uses32 hypothetical histories per prefix, MC seeds339000001..339000512.
DEV uses128 histories per prefix, MC seeds340000001..340000256 in complete DEV
roster order. All banks are collected once. There is no reused diagnostic bank,
exact horizon4 tree, selection-stream reference or DEV-based target choice.
Historical source/runtime assets are authenticated, with no old empirical arrays.

## Two arms, one finite-bank objective

For raw costs Q[n,j,a], first set found costs to zero, then define
z[n,j,a] = (Q[n,j,a] - mean_a Q[n,j,a])/64. Use float64 target arithmetic from
recorded float32 teacher outputs. A common TRAIN scale is float32
sqrt(max(mean(z squared),1e-6)), over all TRAIN prefixes,32 draws and4 actions.
Do not remove found draws, discard all-found banks or normalize by survivors.

Both arms are the unchanged action_recurrent model:28-state GRU,8,299 nominal
parameters, same initial state for each paired seed. The unused outcome and
auxiliary heads remain unchanged (203 parameters), while their forward work is
still charged. All eight blind transitions execute; the loss reads horizon8 only.

* `sampled`: one precommitted label per prefix per epoch. The32-member permutation
  from PCG64(SeedSequence([fit_seed, retained_train_index,717])) repeats three
  times across96 epochs. Every saved label is used exactly three times.
* `mean32`: mean of those same32 labels on every exposure. Precompute this
  reduction once, account for it, and repeat it without new teacher calls.

Both minimize mean((prediction-z_target)/scale squared) across each batch's
prefixes and all four actions. At a fixed prediction, averaging32 squared losses
has the same gradient as loss to their average target. Comparing those two
equivalent gradients would be vacuous. The present comparison changes per-update
gradient noise while holding the finite-bank population objective, acquisition,
initialization, batch size and number of updates fixed. It does not match the
number of target contributions per update; that difference is the intervention.
Training bank construction is a shared cost, including all32 labels in both arms.

Fit seeds343000001..343000003, six fits total.96 epochs, Adam0.003 with default
betas/epsilon, batch32, gradient clipping5, no schedule or weight decay. Each
epoch's case permutation is PCG64(SeedSequence([fit_seed,epoch,919])). Both arms
use identical orders; rotate arm execution order by seed index. Record all
permutations, sampled-label schedules, initialization hashes, updates and timing.
Finish and save all six final checkpoints before the first DEV decode. No model
selection, ensembles, hyperparameter tuning or alternative training objectives.

## Evaluation and continuation rule

Every model chooses the lowest-index minimum predicted cost at horizon8. Score
unconditional regret Q[chosen]-min_a Q[a] for every saved DEV draw, with found
draws contributing zero. Average draws within each prefix, then prefixes equally.
All-found banks stay in the denominator. Report every fit's regret, normalized
cost MSE against the independent DEV bank, actions, support and computation.

The primary comparison averages three policies as separate decisions for each
arm, not their logits. Use2,000 paired hierarchical bootstrap replicates per
regime, seed PCG64(SeedSequence([344000001,regime_index])). Allocate all prefix
resampling indices[2000,N] first, then draw indices[2000,N,128], as in the
previous diagnostic; repeated prefixes have independent inner resamples. Share
indices across arms/fits. Linear2.5%/97.5% percentiles describe uncertainty
conditional on these trained models and the fixed training bank, not training
variability or rigorous coverage. Fit seeds and draws are not independent cases.

For BOTH regimes require at least64 retained prefixes, positive sampled-arm mean
regret, all three paired fit point gains positive, and a positive lower interval
endpoint for gain minus0.05 times sampled-arm regret. Otherwise `DEV_FAIL`.
Publish every condition and both outcomes. No new checkpoint is promoted for
architecture, autonomous control or ordinary-observation claims. A pass supports
testing this target construction in a separately registered recurrent mechanism
comparison; a failure means this intervention has not earned that conclusion.

## Bounds and audit

Single CPU numerical threads,4GiB RSS and512MiB output per phase. Collection
has a3,600second suspend-inclusive cap; fit/analysis1,800seconds; independent
audit3,600seconds. Maximum768 resets,6,144 native prefix steps and49,152 teacher
endpoint calls. Found paths skip teacher calls without reallocating unused draws
or budgets. All nested costs and six training/inference times are recorded.
Report measured wall times; no equal-speed claim follows from matched updates.

Before registration: fabricated sampler, loss-gradient identity, found-zero,
normalization, schedule, future-input isolation and independent metric tests;
full-size synthetic capacity projection; guarded native metadata preflight;
source review and a scoped historical seed scan. No empirical arrays, teacher
calls, native steps or existing checkpoints enter qualification. Engineering
failures remain receipts; repaired source requires fresh qualification.

After original collection and fit processes close, independently reconstruct
strict/legacy filtering, saved counterfactual draws, forward-record cost
reductions, TRAIN scale and schedules, DEV regret/MSE and all bootstrap/gate
calculations. Audit only saved records; no new teacher, model, simulator or
optimizer calls. Decode the new TRAIN, DEV, sensor and prediction archives plus
all six new checkpoints to check schemas and canonical state hashes without
executing them. No prior checkpoint is decoded. State any saved-record
limitation, including lack of independent training
replay. Close the audit only from its original terminal receipt and absent
process group. Publish sources, receipts, checkpoints and failures.
