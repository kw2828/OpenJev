# Learned deletion from a bounded Gaussian memory

Prospective protocol, 24 September 2026. Version `retention-v1`.
The [residual correction experiment](residual-memory-results.md) remains failed,
9/11 conditions. Its full-covariance comparison motivates this separate test;
it does not establish that a learned selector will improve anything.

## Question and contribution boundary

Can a small learned deletion rule improve path-risk decisions over strong
analytic retention at the same logical storage cap? This is a trained selector
on an established sparse Gaussian recurrence, not a claim of new GP inference,
biological wiring, calibration, robotics performance or general world modeling.

The state defines `q(f) = p(f | f(Z)) N(f(Z); m,S)`. Each real observation expands
the dictionary, conditions exactly once, then discards one coordinate while
retaining the remaining posterior marginal. Outside the dictionary, the prior
conditional is restored. No discarded raw observation or hidden archive remains
accessible to the model. Cross-query residual covariance is preserved.

Analytic deletion minimizes forward KL from the augmented Gaussian to the
retained-marginal/prior-conditional projection. This is related to
[sparse online GPs](https://eprints.soton.ac.uk/259182/), but is not a reproduction
of the reverse-KL optimizer in
[streaming variational GPs](https://arxiv.org/abs/1705.07131).
[Decision-directed belief compression](https://papers.nips.cc/paper_files/paper/2002/hash/14ea0d5b0cf49525d1866cb1e95ada5d-Abstract.html)
and [recurrent interdomain GP memory](https://arxiv.org/abs/2502.08736) are relevant
prior art. A useful contribution requires comparative results for the specific
learning rule and resource tradeoff, followed by a second environment.
The field itself is static; recurrent posterior updates are not evidence of
learning a transition model for a changing world.

## Fixed world and requests

The latent field is a zero-mean GP on a 17 by 17 grid in `[-2,2]^2`, spacing
0.25. Its kernel is RBF with length and amplitude 1, plus a spatial white
component of variance `1e-5` shared at exactly equal coordinates. This component
is part of the supplied generating law, not adaptive numerical jitter.
Public observation noise has independent variance `0.09` at each event.
All models receive the same known kernel and noise parameters.

Each context streams a random permutation of distinct grid locations with their
noisy observations. Only after the stream ends are four requests revealed.
Each request offers four distinct paths, each containing four latent grid
locations. Exposure is their mean; it contains no additional observation noise.
Path cost is `0.02 + P(exposure > 0.5)`. Deferral costs `0.20`.
Choose the first minimum-cost action in path order 0-3, then defer at index 4.
Thus the full within-path covariance matters; between-path covariance is not
needed to minimize this particular expected cost.

TRAIN and BASE sample uniformly from 374 axial paths. Adjacent path points are
two grid steps apart. SHIFT samples uniformly from 242 diagonal paths at the
same stride and point count, changing both orientation and geometric length.
LONG changes only observation count. No context is filtered by outcome,
difficulty, reference confidence or control performance.

Namespace `443260924`. Per-context `SeedSequence([seed, context])` produces
independent order, field, observation-noise and request streams. TRAIN uses the
namespace seed, 256 contexts, 64 observations and four requests. Evaluation uses
three independent cohorts of 128 contexts per population:

| Population | Seed offset + cohort 0,1,2 | Observations | Path geometry |
| --- | ---: | ---: | --- |
| BASE | 100 | 64 | axial |
| SHIFT | 200 | 64 | diagonal |
| LONG | 300 | 192 | axial |

There are 1,152 evaluation contexts, 4,608 requests and 18,432 path outcomes.
Requests and path outcomes within a field are correlated. All three fits share
TRAIN fields; evaluation cohorts are independent. The 49,152 training trajectories
reuse fields and are not independent test examples.

## Learned rule and training

The learner keeps eight dictionary coordinates. Each deletion candidate supplies
six public-state features: x/2, y/2, prior-conditional mean residual divided by
its prior standard deviation, log posterior/prior conditional variance ratio,
log(1+KL), and posterior/prior marginal variance ratio.
A shared 6-to-4 tanh-to-1 MLP has exactly 33 float64 parameters. Its retention
score is `log(KL + 1e-12) + 2*tanh(MLP(features))`. Lower scores are deleted.
The last layer starts at zero, so deterministic initial inference equals KL8.
The `1e-12` is a score offset, not a covariance correction.

Seeds 11, 23 and 37 each train for 16 epochs over all 256 TRAIN contexts,
shuffling with that seed. Batch size is eight contexts, each with four independent
categorical trajectories under the same current policy and observations.
Sampling logits are negative retention scores, temperature 1. Terminal reward
is negative mean conditional regret on that context's four TRAIN requests,
using the full public-history GP as a training teacher. Future paths and teacher
probabilities never enter write features.

The detached baseline is the mean reward of the other three trajectories in
the group. Divide the resulting advantage by a fixed 0.05. Multiply by the
trajectory's mean deletion log probability. Averaging the 56 deletion log
probabilities scales the episodic REINFORCE gradient by the constant 1/56.
Subtract `0.002 * mean_entropy`; use Adam at 0.01 and clip gradient norm at 1.
This is group-relative on-policy REINFORCE with leave-one-out baselines. There
is no PPO clipping, critic, straight-through deletion or reuse of stale rollouts.

Each fit has 512 optimizer updates and 16,384 group trajectories. Total:
1,536 updates and 49,152 trajectories. Keep the final model from every seed.
There is no validation-selected epoch, early stopping, replacement seed,
architecture search or tuning against evaluation. Clear parameter gradients,
discard Adam state from inference, and freeze the policy before evaluation.

## Controls and actual logical storage

The common cap is 1,024 bytes per independent inference context. Count all
retained numeric arrays, four float64 kernel/noise scalars, one int64 event
counter, and every learned parameter. No cached factors, ages, normalizers or
RNG state remain during deterministic inference. Python object overhead is
excluded for every method; this is logical state, not process RSS.

| Method | Retention | Bytes |
| --- | --- | ---: |
| learned-11/23/37 | eight-point posterior plus 33 learned parameters | 1,008 |
| KL8 | same recurrence, minimum-KL deletion | 744 |
| KL9 | extra useful slot purchased within the cap | 904 |
| FIFO9 | same nine-point recurrence, discard first retained coordinate | 904 |
| FIC9 | fixed off-grid 3 by 3 inducing lattice at -1.8,0.1,1.8 | 904 |
| coverage41 | 41 coordinate/label triples, exact retained-row GP | 1,024 |
| recent41 | most recent 41 triples, exact retained-row GP | 1,024 |

Coverage retention processes every observation. On overflow, discard the first
point whose nearest-neighbor squared distance is smallest, preserving retained
order. It uses coordinates only. FIC incorporates residual variance into each
observation likelihood and uses event-independent residuals in path prediction;
it is explicitly an approximation, including for repeated query locations.
The projected dictionary models preserve shared-coordinate residual covariance.

The full-history GP and its covariance-diagonalized diagnostic are larger-memory
references, not budgeted controls. Always defer is a decision baseline. Exact
retained-row GP means exact conditioning on those rows, not on the full stream.
The benchmark driver stores datasets and traces for audit; models cannot access
those external buffers. This distinction is explicit in the write-only API.

## Evaluation and fixed continuation rule

Report decision regret against the full public-history conditional law,
latent-exposure Gaussian NLL, Brier score for threshold exceedance, MSE, central
90% interval coverage, defer rate, and absolute risk-probability error. Full-GP
zero regret is definitional; it never receives private realized exposures.
Means average scores across contexts and fits, not model predictions.

All 42 conditions must pass:

- BASE and LONG mean learned regret must be at least 10% below each of six
  budgeted controls. SHIFT may be at most 5% higher. Add only absolute `1e-6`
  to each regret limit: 18 conditions.
- Each population's mean learned NLL must be at most 0.02 nat above the best
  budgeted control's NLL: three conditions.
- Each population must beat always defer: three conditions.
- Each of nine evaluation cohorts must have mean learned regret no higher than
  its strongest budgeted control (SHIFT allows 5%); add `1e-6`: nine conditions.
- Each fit on each population must satisfy the same strongest-control
  noninferiority limit (SHIFT allows 5%); add `1e-6`: nine conditions.

Pass yields `ADVANCE_RETENTION` for further validation. Otherwise
`DO_NOT_ADVANCE_RETENTION`. A mean win cannot erase a failed cohort, fit,
predictive-quality check or resource constraint. Neither outcome establishes
general architecture novelty or a calibrated real-world decision service.

## Runtime, numerical checks and evidence

One-thread CPU float64. Total registered run cap is 1,800 seconds, including
generation, all fits, evaluation and timing. No retry after empirical failure
or timeout. Preserve completed fits, partial fit weights/Adam/traces, original
process closure and failure receipts. Partial weights may contain an interrupted
update and are never promoted as final.

Time the first context of each population: complete stream plus four requests,
three warmups and ten measured repetitions per method. Include selection and
matrix reconstruction. Report Python/NumPy traced peak allocation separately;
native Torch/BLAS workspace and peak RSS are unmeasured. No claim of equal peak
memory, equal compute, significance or a general speedup follows from this probe.

Qualification uses fabricated cases for dense-GP parity before compression,
marginal projection, KL identities, covariance-sensitive decisions, independent
noise/shared-coordinate semantics, causal writes, no hidden arrays, storage,
initial KL equivalence, nonzero learning gradients, failure persistence and all
continuation conditions. The only KL repair is flooring values in
`[-1e-10,0)` to zero with a count; more negative values fail. No adaptive jitter,
posterior-variance clipping or replacement samples.

Freeze these sources, this protocol and registration before any scientific
generation or fitting. Save all data, final predictions, retained states,
deletion traces, initial/final policies, optimizer states, training actions,
group rewards and field IDs. Evaluation has 99 prediction groups, 81 states
and 33 resource records. The independent auditor uses its own NumPy/SciPy
conditioning, validates saved predictions/states and reconstructs metrics and
the 42-condition rule. Independent numeric comparisons use rtol and atol `1e-8`;
priority comparisons allow that tolerance and count near ties. Exact historical
tie behavior, training dynamics and timings remain source/receipt evidence.
