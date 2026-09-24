# Query-centered nonlinear features: a bounded learning pilot

Registered before empirical generation or training, 24 September 2026.
This tests a prerequisite for query-time memory refinement. It is not a
recurrent world model, biological architecture, text model or novelty claim.

## Question and prior art

Does recomputing a learned nonlinear representation around the current query
improve uncertain decisions beyond static learned features and rank-matched
kernel features? A positive result would justify a separate experiment that
learns when and what to recompute. This pilot always recomputes every block.

[ALPaCA](https://arxiv.org/abs/1807.08912) already combines learned features and
Bayesian linear regression. [Deep kernel learning](https://proceedings.mlr.press/v51/wilson16.html),
[MOCA](https://arxiv.org/abs/1912.08866), and
[Attentive Neural Processes](https://arxiv.org/abs/1901.05761) are close prior art.
Query centering and a Bayesian head alone are not new contributions.

## Data and public information

Each context contains K independent scalar functions of two-dimensional x,
drawn from a zero-mean RBF Gaussian process with length 1 and amplitude 1.
All observations have independent Gaussian noise with variance 0.0225.
Each public, separately identified archive block has 16 observations.
Each request has four noisy examples from a uniformly selected unknown block
and one query coordinate. Its target is a new noisy observation, not the
noiseless function value. The model receives archive coordinates and values,
the request examples and query coordinate. It never receives the true block
identity, query target or observations from other requests.

For every block, basis observations and all request coordinates in a context
are sampled jointly from RBF covariance plus noise times identity. Selection
then exposes only the chosen block's request examples and target. Requests
share functions and are statistically dependent within a context. They are
scored separately from their own public information; no request target is
assimilated or used to route later requests. There is no observation replay
that adds evidence twice.

Namespace 441260924: TRAIN uses that seed, 1,024 contexts, K4, coordinate
extent [-2,2], two requests each. BASE cohorts use namespace+100,+101,+102,
128 fresh contexts each, four requests, K4 and extent [-2,2]. SHIFT cohorts
use namespace+200,+201,+202, 128 contexts each, four requests, K8 and extent
[-3,3]. SHIFT is a compound transfer test, not an isolated memory-capacity
intervention. Generate evaluation cohorts only after all fits are final.

## Five fitted arms and a supplied-law reference

All Bayesian heads use independent N(0,I) coefficient priors, known noise,
joint four-example likelihoods to infer a uniform-prior block identity, and
full mixture predictions. They condition once on request examples.

| Arm | Representation | Fitted parameters |
| --- | --- | --- |
| static16 | 2 -> 32 tanh -> 16, divided by sqrt(16) | 624 |
| centered16, candidate | Same network, all x replaced by x-query_x | 624 |
| static32 | 2 -> 32 tanh -> 32, divided by sqrt(32) | 1,152 |
| nystrom16 | RBF features from a fixed 4x4 grid on [-2,2]^2 | 2 |
| centered_nystrom16 | Same grid after query centering | 2 |

Nyström fits log length and log amplitude, both initialized at zero; its
inducing covariance receives fixed 1e-6 jitter. No diagonal variance repair
or missing-rank correction is added. The full GP reference receives the true
generating kernel, uses all public observations, marginalizes block identity
and includes observation noise. It is a privileged conditional-law reference,
not a learned competitor with equal prior information.

Centered predictors use a consistent feature map within each request. This
does not establish one coherent joint GP across different queries. Model-based
uncertainty is not an empirical calibration guarantee. The separately tested
ReplayEvidence component enforces idempotent evidence IDs for a fixed exact
linear model; it is not a learned calibration claim or an extra trained arm.

## Training and stopping

Paired fit seeds 11,23,37. Static16 and centered16 have identical initialization
within a seed. All arms see the same TRAIN data and per-seed context shuffles,
using numpy RNG namespace+10000+fit_seed. Each context contributes both requests.
Sixteen epochs, batches of 32 contexts (64 requests), 512 Adam updates per fit,
32,768 request exposures per fit. Adam learning rate 0.003, default betas and
epsilon, zero weight decay; global gradient norm clipped at 10. Train mixture
negative log likelihood, float64 on CPU, one thread, deterministic algorithms.
Use the last checkpoint. No early stopping, best-epoch selection, tuning,
replacement fit, retry, extra training or evaluation-driven arm addition.

Fifteen fits total. Nyström initialization is deterministic, so its three fits
vary only in training order. Fit seeds reuse TRAIN and are not independent
datasets. Any numerical error or 3,600-second total run cap stops the run and
retains partial results. A separate later proposal is required to change a
failed run. Fabricated algebra tests and runtime qualification precede freeze.

## Decisions, scoring and continuation rule

Actions have stable IDs negative, defer, positive. A wrong sign costs 1, a
correct sign costs 0, and defer costs 0.2. Each model chooses the lowest
predicted expected cost; ties choose the first action in that order. Score
its expected cost under the exact public-information GP mixture, minus that
mixture's minimum expected cost. The private selected block is not used.
Also report target NLL, Brier score, MSE, all three action frequencies and
always-defer regret. The correct-law reference has exactly zero conditional
regret by construction; its observed NLL need not win every finite panel.

Advance centered16 to a selective-refinement pilot only if, separately on BASE
and SHIFT, it meets all three requirements against each of static16, static32,
nystrom16 and centered_nystrom16:

1. Mean conditional regret is at least 10% lower and strictly lower.
2. Mean NLL is no more than 0.02 nat/request worse.
3. Mean conditional regret is lower on each of the three cohorts, averaging
   the three paired fit predictions' scores, without ensembling predictions.

This is a screening gate, not a significance test, novelty threshold or
generalization guarantee. Average requests within contexts and give every
context equal weight. Report all 90 fitted-arm evaluation rows, six GP rows
and all eight comparison groups, including failures. Do not promote a
different winner after seeing results. A failed candidate ends this pilot;
it does not disprove all query-dependent memory architectures.

## Resources and evidence

Record actual training time, parameter/buffer bytes, raw archive bytes,
static cache tensor bytes and batched evaluation time. Separately time the
first archive of cohort0 in each population with four sequential requests:
three warmups and 20 timed repeats, batch size one. Static models encode the
archive once per repeat and reuse its statistics; centered models recompute
per request. Include the GP reference's actual unoptimized NumPy latency,
which recomputes each request. Timing includes validation and cache checks.
No inference speed or equal-compute claim follows from equal training exposure.
Array/tensor bytes are not peak process memory; temporary workspace and Python
overhead are excluded. Store unrounded per-repeat timings.

Save data, final checkpoints, full prediction distributions, epoch losses,
metrics, resource observations and a source/environment registration. Freeze
the exact sources before collection and verify them again on completion.
Independent audit must recompute scores and GP conditionals from saved arrays
without training, generating new data or selecting new settings. Publish the
outcome, qualification evidence and reproducible artifacts, including failure.
