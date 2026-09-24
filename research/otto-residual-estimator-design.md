# Frozen-feature residual estimators: proposed mechanism screen

**Proposal only. Not admitted, collected, fitted or evaluated.** This is a new
study design. It does not revise the [query-memory protocol](otto-query-memory-protocol.md),
change its **DEV FAIL 6/13**, promote one of its controls, or open that study's
unused TEST split. Its exact source, input and fresh seed identities still need
qualification and a published registration before execution.

## One question

Can accumulated directional precision improve residual corrections across both
OTTO settings when the recurrent representation, cue history and teacher access
are held fixed?

The [closed DEV report](otto-query-memory-dev-results/query-memory-dev.md) showed
that trace-delta memory substantially worsened lambda3, while lambda4's aggregate
improvement did not survive every paired fit seed. This motivates testing the
online estimator. It does not identify stale memory, establish useful uncertainty,
or demonstrate that a different existing control is a successful candidate.

The sole candidate in this proposal is **full-covariance RLS with an unattenuated
read**. No learned gate, new backbone, gradient fitting, adaptive query schedule,
forgetting or process noise is included. Those would introduce additional
explanations before establishing whether this simpler estimator is useful.

## Shared representation and fresh paths

Reuse all three existing TRAIN-derived `pretrained` GRU checkpoints and their
corresponding TRAIN-derived `trace_delta` projections, with fit seeds 309000001,
309000002 and 309000003. Retain the same three TRAIN-derived `joint_aux`
checkpoints as a stronger ordinary continued-training baseline. The source
study's frozen 18-fit lineage establishes
that these weights were completed before DEV access. Bind the actual original
checkpoint bytes, training receipt, independent audit and original process
closures in the new registration. Do not select a fit seed or projection based
on its DEV score. The projection came from the failed candidate; its reuse is
representation control, not evidence that it generalizes or is estimator-neutral.

Collect a new, independently reserved cohort:

| Split | Cases per setting | Settings | Collectors per case | Paths |
| --- | ---: | ---: | ---: | ---: |
| Development | 3 | lambda3 and lambda4 | 3 | 18 |
| Confirmation | 6 | lambda3 and lambda4 | 3 | 36 |
| Total | 9 | 2 | 3 | 54 |

Keep analytic, neural and period-four held-teacher collectors, the public
environment and teacher, complete score census and original durability checks.
Register the exact seeds and collector ordering before any native call. Neither
split may reuse old TRAIN, DEV or TEST paths or seeds. The failed study's TEST
remains unread by this study even though its files exist.

For each pretrained checkpoint, build one authenticated feature cache on each new
split. The encoder sees public observations and only actual P4-scheduled answers.
All other teacher scores remain evaluator targets. Cache the complete slow
shadow forecast, ordinary action forecast and the same normalized history cue
that the frozen trace projection supplies. Preserve the current first-query
exclusion, read-before-write order, zero initialization and episode resets.
The correction never feeds back into the GRU or changes a collector action.

Each `joint_aux` checkpoint gets its own frozen P4 forward on the identical
public observations and scheduled answers. It uses no residual estimator or
trace-projection cache. Keep all three seeds and charge these three additional
forward passes over the split's full path roster on each split. Its weights
remain fixed; no additional gradients or optimization occur. It was the
strongest lambda3 control in the failed screen, so omitting it would weaken the
usefulness comparison. Estimator isolation is assessed within the shared
pretrained representation; `joint_aux` is a separate stronger utility reference.

The cache is an offline economy, not a deployment optimization result. Its
generation must be tested against the uncached source implementation on
fabricated examples before registration. Confirmation cache generation and
numerical decoding require the completed development decision and its original
independent audit closure.

The cache builder uses one complete episode at a time, physical batch size one,
with chronological 32-step chunks and poisoned final padding. Both recurrent
forwards use that same geometry. Projection remains a `[1,28]` to `[1,8]` linear
call on every eligible key step. Carry the unnormalized float32 trace between
chunks and normalize only its read/write cue, matching the original kernel.
The comparison does not claim bitwise parity with a prior run that used a
different physical batch geometry.

## Exact methods and numerical convention

All residual estimators consume the identical per-seed cache and P4 answer
schedule. `joint_aux` uses the separately charged frozen forward described above.
Use the
centered residual against the full shadow forecast, in score/64 units. Use
float64 for every online residual estimator, with the cached float32 cues and
forecasts promoted exactly to float64. Form corrected action scores in float64,
then round once to float32 before applying the unchanged legal near-minimum
decision rule. Actual query outputs are always copied from the observed answer.
The no-correction baseline preserves its original float32 action scores.

This deliberately makes estimator arithmetic common. The delta control uses the
current update equation in float64; it is not claimed to reproduce the previous
float32 experiment bit for bit. Qualify this convention for every method before
admission. Invalid arithmetic fails the registered run; there is no jitter,
clipping, variance floor, reset or replacement key that rescues a result.

| Method | Role and fixed behavior |
| --- | --- |
| No correction | Ordinary frozen GRU baseline |
| Joint AUX | Frozen TRAIN-derived continued-training GRU; its own P4 forward, no residual estimator |
| Last error | Overwrite with the latest eligible centered residual; decay 0.75 at each key step |
| Normalized delta | Current trace cue, step size 0.25, denominator epsilon 1e-6, matrix decay 1 |
| **Full RLS** | Sole candidate; full posterior covariance, read multiplier 1 |
| Diagonal RLS | Mechanistic control; same gain denominator, then discard posterior off-diagonal entries |
| Full RLS, multiplier 0.25 | Prespecified weaker read of the same full posterior |
| Full RLS, multiplier 0.5 | Prespecified weaker read of the same full posterior |
| Full RLS, multiplier 0.75 | Prespecified weaker read of the same full posterior |

All RLS variants start with zero mean and isotropic covariance. Their posterior
updates use the **unattenuated** prewrite posterior mean, even for the three
weaker-read controls. The multiplier affects application of the correction only;
it cannot change targets, queries, cue computation or posterior evolution. These
three controls are distinct fixed comparisons, not alternative candidates.

Use the [qualified RLS reference](otto-query-memory-bayesian-baseline.md) and its
explicit diagonal moment-projection definition. The diagonal method is not
diagonal precision accumulation. All three methods with weaker reads share full
RLS's already-defined posterior mathematically; report any execution reuse and
its accounting explicitly.

## Small, fixed development choice

No gradient optimization is performed. The only candidate hyperparameter choice
is the isotropic prior-to-noise ratio `tau` from this ascending fixed grid:

`{0.01, 0.1, 1, 10}`.

Set observation variance to 1 in normalized residual units and prior covariance
to `tau * I`. This fixes the regularization scale; it does not estimate or
calibrate real observation noise. The ratio determines the RLS mean updates.
An uncertainty value from this specification is not a probability of choosing
the correct action.

Evaluate every grid value on the new development split. Use that same value for
its full, diagonal and weaker-read variants. Select one common `tau` across both
settings and all three frozen fit seeds by minimizing

`max(mean_later_gap_lambda3, mean_later_gap_lambda4)`

for **unattenuated full RLS only**. Compute each mean by equal fit-seed, then
originating-case, then collector-path weighting, retaining zero-support paths.
Break an exact tie by the first value in the ascending grid, without a tolerance.
This is a prospectively declared candidate-selection step, not held-out evidence.

Preserve all grid outputs, including diagonal and weaker-read results. They
cannot choose `tau`, the candidate, a setting-specific configuration or a
different continuation threshold. There are **72 development views**: four
usefulness baselines times three seeds, plus five RLS read/covariance variants
times four ratios times three seeds, `4 * 3 + 5 * 4 * 3`. The confirmation
allocation is **27 views**, `(4 + 5) * 3`. Require the registered implementation
to verify this exact roster and these counts.

## Usefulness gate and exact continuation

Use the same primary outcome definitions as the previous screen: legal teacher
score gap on fixed paths, full nonqueries excluding actual queries, and later
nonqueries from step 5 onward. Report initial steps 1-3, every age, collector,
case and fit seed as secondary breakdowns. Agreement and residual MSE are
secondary; neither can rescue a worse decision-gap result.

The **usefulness controls** are no correction, Joint AUX, last error and normalized
delta. They answer whether the new candidate improves on ordinary recurrent and
residual baselines. Diagonal RLS and weaker reads answer separate mechanism questions
below. This is an explicit design choice for a new study; it does not replace
or reinterpret the failed study's controls or acceptance rule.

After selecting `tau`, require all 13 conditions on development:

1. One genuine technical-completion condition, including independent saved-output
   audit and successful original producer and audit supervisor closures.
2. For each setting separately, at least two originating cases support the later
   scope.
3. For each setting, the candidate's mean later gap is at most 90% of, and strictly
   below, the lowest mean among the four usefulness controls. A zero best-control
   gap fails this condition.
4. For each setting, its mean full-nonquery gap is no higher than the lowest
   corresponding usefulness-control mean.
5. For each setting and each of the three paired fit seeds, its later gap is no
   higher than the lowest same-seed usefulness-control gap.

The count is `1 + 2 * (1 + 1 + 1 + 3) = 13`. Conditions are a conjunction, with
no tolerance, pooled rescue or post-result exceptions. If development fails,
close this study and leave confirmation unused. Do not switch to diagonal RLS,
a weaker read, another grid value or a different seed.

If development passes, freeze the selected `tau`, all source/checkpoint/cache
identities and the selection receipt before a separately supervised confirmation
phase. Evaluate all nine methods and all three frozen seeds exactly once.
Apply the same 13 conditions, except require **four** supported originating cases
per setting. Require successful independent confirmation audit and original
process closures. Any failed condition closes the screen at FAIL.

Only a confirmation usefulness pass permits a new, separately registered
autonomous quality-versus-total-compute comparison of the same full-RLS candidate.
It does not automatically admit an uncertainty gate, a different model, a paper
claim or an altered query budget.

## Mechanistic contrasts, kept separate from usefulness

For every ratio on development and the fixed selected ratio on confirmation,
report full RLS versus diagonal RLS and each strict read multiplier. Report mean
and paired-seed later/full gaps separately for lambda3 and lambda4, including
ties and regressions. These comparisons never change the usefulness verdict or
promote a control into the candidate slot.

Evidence that full covariance earns its complexity requires, on confirmation,
all of: at least a 10% and strict later-gap improvement over diagonal RLS in each
setting; no worse full gap in either setting; and no later-gap regression in any
paired fit seed. A tie, zero diagonal gap, setting failure or seed regression
does not support this explanation. This is a matched-regularization contrast,
not a claim that no separately tuned diagonal model could perform as well.

Strict attenuation controls are `{0.25, 0.5, 0.75}`. Multiplier 1 is the candidate
itself and is not included in a strongest-control requirement that would make
improvement over an identical predictor impossible. There is no requirement to
beat a weakened read by 10% to establish usefulness. If a weaker read is better,
or ties full RLS, report that unattenuated
correction strength is not established as necessary. Do not rename that weaker
control the successful candidate.

Interpret negative outcomes directly:

- If the ordinary or last-error baseline explains the performance, accumulated
  cue-conditioned regression has not earned its complexity.
- If diagonal RLS matches or improves on full RLS, cross-direction covariance has
  not earned its cost.
- If weaker reads help, excessive correction strength remains an explanation;
  this is not evidence that uncertainty-dependent gating would work.
- If RLS improves residual MSE but worsens decision gap, reject the utility claim.
- If either setting or a paired seed fails the usefulness rule, preserve that
  failure even when the pooled mean looks favorable.

## Costs, limits and claims

Charge full census collection, one feature extraction per pretrained checkpoint
and split, all three additional Joint AUX forwards per split, cache
serialization/verification, every estimator replay, hyperparameter
selection, every read multiplier, metrics and independent audits. Shared work
must be visible rather than counted as free or duplicated invisibly. Report
state size, actual wall time and peak memory. The float64 full-RLS reference
also performs Cholesky validation after updates; its extra validation cost must
remain in the measured implementation budget. Equal teacher access is not equal
total computation, and cache reuse does not prove deployment acceleration.

Exact source manifests, fresh seed reservation, per-phase resource caps, qualified
cache/estimator integration, admission commands and original-process evidence
are outstanding implementation work. None of the prose above admits execution.

The [related-work comparison](otto-query-memory-related-work.md) and
[Bayesian baseline proposal](otto-query-memory-bayesian-baseline.md) already
identify delta-rule memory, RLS/ALPaCA-style regression and uncertainty-dependent
plasticity as established methods. A positive screen would establish useful
Bayesian residual adaptation under this restricted protocol. It would not
establish a new learning rule, connectome advantage, world model, RLCD, conformal
coverage, calibrated action probability or ICLR-level architectural novelty.

Only after useful adaptation is established should a separate experiment test
whether uncertainty can choose correction strength. That comparison would need
constant and query-count-based attenuation controls: decreasing posterior
variance alone can track observation count while missing model bias or drift.
