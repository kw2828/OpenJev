# Fixed spectral evidence memory: implementation and saved-path tradeoff

21 September 2026. Prospective protocol for a new **untrained representation
screen on existing development histories**. The preceding
[evidence diagnostic](otto-memory-evidence-results.md) found that both detections
and absences affect decisions. This screen implements a compact additive memory
and measures distortion, retained state and complete computation. It neither
reverses the original 9/10 failure nor admits a learned or autonomous control
study. No new simulator episodes, training, external model calls or tuning.

## Public information and frozen numerical sensor

Use the authenticated original 53x53 cohort, its completed worker/supervisor
receipts and independent audit. Select all 96 original `space_full` histories,
without filtering by their outcomes. There are 2,164 pre-action prefixes; 538
have more than 32 completed observations. Those counts are known from the
previous diagnostic and must be reconciled in the new output.

Actors receive only public initial and subsequent packets: position, hit, done,
step and valid actions. They receive the same supplied observation model and
initial-hit-conditioned prior. No source coordinates, random streams, seed IDs,
simulator posteriors or future observations enter the actor. All proposed actions
are scored on the recorded full-policy path; none is executed.

The original probability kernel has shape `(4,107,107)`. A read-only inspection
of that saved kernel found zero counts of **1, 1, 1 and 6,993** in its four categories. The last
category is calculated upstream by subtraction from one, which rounds many tiny
positive analytic tail probabilities to zero. These numerical zeros are part of
the frozen runtime. Do not replace its kernel, add floors, or describe this
screen as exact filtering under an ideal analytic Poisson model.

Every full-memory additive or spectral actor retains a **persistent hard-support
mask**. It excludes the initial prior's zero cells, all cells visited without
finding the source, and every zero-likelihood cell from every completed reading.
This is more information than a visitation ledger alone. It occupies one boolean
array, not an uncounted hidden posterior. Reject empty support without repair.

## Fixed recurrent representations

The exact additive control stores a full 53x53 accumulated finite log-evidence
field. The compressed variants store only the top-left `q x q` coefficients of
an orthonormal two-dimensional DCT-II, for fixed `q` values **4, 8 and 16**.
Thus they allocate **16, 64 and 256 evidence coefficients**, respectively. The
constant coefficient is redundant after posterior normalization but remains
allocated and counted. Recurrence is additive: each observation adds its
projected log-likelihood to the previous coefficients. No forgetting or weights
are learned.

For positive kernel entries, use their exact floating-point natural logarithm.
At zero entries, test both predeclared finite extensions:

- **Neutral:** substitute log value zero.
- **Nearest:** substitute the same category's log likelihood at Euclidean
  distance one, including at off-origin numerical zeros.

Both variants have the same hard-support mask. At full rank the substituted
values are excluded after reconstruction. At smaller ranks their projected
influence can spread to supported cells; this is an explicit sensitivity
comparison, not an opportunity to select the better extension after the run.

Decode by zero-padding the coefficient array to 53x53, applying the corresponding
orthonormal inverse transform, adding the exact initial-hit log prior, enforcing
the support mask, and normalizing with a stable log-sum-exp calculation. Return
temporary probabilities and stable log probabilities without retaining a dense
belief cache. A zero observation updates memory normally. Include the initial
hit once via the prior. The terminal `-2` sentinel is never encoded as odor, and
no action or further observation update is allowed after termination.

Shared immutable arrays may hold the original kernel, finite extensions and
the three initial priors. Count all of them. No history-dependent shared cache
or per-actor list of old observations is allowed in the spectral or exact-additive
models. A fresh actor resets its coefficients and support mask.

## Twelve arms and strong controls

All arms use the identical pinned space-aware one-step score function and its
first-action-within-`1e-10` tie rule:

1. Original full-history probability-space Bayes filter.
2. Full-grid additive log-evidence control.
3. Recent-32 evidence reconstruction with its permanent visitation ledger.
4. Recent-32 soft evidence with persistent full-history hard support.
5. Six compact combinations: three ranks times the two finite extensions.
6. Two full-rank `q=53` extensions, used as numerical qualification controls.

The added recent-32 hard-support control distinguishes retaining numerical
zero constraints from retaining older soft evidence. The original recent-32
baseline continues to forget old likelihood constraints when they leave its
window. Both recent-window controls use stable log arithmetic on the same
retained evidence, rather than claiming bitwise reproduction of probability
multiplication. Both retain the exact initial prior and account for their observation
buffers and masks. Rotate execution order across all twelve arms by case index
plus number of completed moves; do not group favorable or cheaper variants first.

Require the original full filter, exact additive control and both full-rank
extensions to reproduce every saved selected action, with action-score error
at most `1e-8`. Exact-additive and full-rank posterior total variation from the
original full filter must be at most `1e-10`. Stable logs may retain positive
mass where repeated probability multiplication underflows; record such support
differences, rather than promising bitwise equality. Any violated qualification
stops the run with a failure receipt.

## Quality, state and computation

Persist every case and every pre-action prefix, including each arm's action,
four scores, total variation from full belief, `KL(full || arm)`, full-belief
heuristic objective excess and selected-action agreement. Calculate KL using
stable log probabilities on the positive reference support; do not floor
probabilities. Objective excess is dimensionless, not realized search-time regret.
The full filter is a reference, not an optimal policy oracle.

Summarize all prefixes and the subset after more than 32 observations. Report
initial-hit-stratum and per-case results as well as conditional case-weighted
and prefix-weighted means. A case's original mixture weight is its initial-hit
weight divided by 32. Renormalize over the explicitly eligible population.
Publish both extensions at every rank and their disagreement; do not promote
one as an efficacy winner.

Measure initialization, update, decoding and shared action-planning time for
each arm and case. Include kernel extraction, log-field encoding and transform
cost in updates. Include full-grid decoding and planning; isolate shared model
initialization and whole-analysis overhead. Use CPU with one numerical thread.
This single rotated pass yields descriptive timings, not a deployment benchmark.

Count allocated mutable arrays and observation buffers, shared immutable arrays,
and disclose dense transient working arrays. Report whole-process peak RSS
separately: it contains all arms, saved traces and analysis outputs, and is not
the peak memory of one deployed actor. Fewer stored coefficients establish
neither lower total memory nor lower latency.

## Qualification, execution and consequence

Before the real saved-path invocation, run synthetic tests of probability-product
equivalence, full-rank extension invariance, low-rank extension sensitivity,
constant-offset invariance, saturated counts, zero evidence, exact support,
chronology, terminal/reset behavior and state accounting. Also test summary
weighting, arm rotation and stable KL with underflowed decoded probabilities.
Bind model, runner, tests, this protocol and inherited math/clock sources in a
new hash-pinned plan committed before execution.

One saved-path invocation in an exclusive directory: **300 suspend-inclusive
seconds, 2 GiB peak RSS and 64 MiB output**. Preserve any failed attempt, all
partial output and its cause. No retries, extra ranks, alternate fills, numerical
floors or budget extensions inside this protocol. A separately recorded repair
can address an implementation defect, but cannot amend a scientific outcome.

There is no new learned-performance admission condition. This screen produces
a complete fixed compression tradeoff and identifies whether representation
error or numerical extension sensitivity warrants further work. A useful small
state would strengthen the additive baseline that any learned recurrence must
beat. A failure would motivate a distinct representation hypothesis, not a
connectome or JEPA efficacy claim. Autonomous search utility, training efficiency,
robustness to scenario shift and architectural novelty remain untested.
