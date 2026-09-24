# Residual memory: test correlated uncertainty before learned retention

Prospective design note, 24 September 2026. This proposal is conditional on a
separate residual-preservation pilot. It reports no new experiment, architecture
advantage or calibration guarantee. The [query-feature failure](query-feature-results.md)
remains closed; this is not a rescue of query centering or its evaluated seeds.

## What the prerequisite must distinguish

The existing Nyström Bayesian last layer already keeps dense posterior
covariance inside its feature span. The missing kernel component is
`R(x,x') = K(x,x') - phi(x)^T phi(x')`, outside that span. Another gate on the
same feature covariance cannot, by itself, recover that component.

Compare the unchanged low-rank model, a query-variance-only diagnostic, proper
fully independent conditional (FIC) inference, and a full GP using the same
fitted kernel parameters on fresh data. Keep the supplied true-kernel GP
separate. FIC must affect archive conditioning, joint few-shot likelihoods,
identity weights and subsequent conditioning, not just the final error bar.
[Sparse pseudo-input GPs](https://papers.nips.cc/paper_files/paper/2005/hash/4491777b1aa8b5b32c2e8666dbe1a495-Abstract.html)
already provide this family of residual-variance corrections.

A gain from that correction would establish a useful uncertainty baseline,
not learned memory. If FIC leaves little decision gap to the full fitted GP,
do not add a retention network merely to continue this branch. A remaining
gap motivates a separately registered test of residual correlations; it does
not prove that correlations caused the previous failure.

## One bounded hypothesis

At fixed total inference storage, can a learned retention rule preserve
decision-relevant residual correlations better than analytic retention?
The proposed recurrent state augments a fixed global feature posterior with
a small residual basis and its joint information statistics. Each new real
observation is conditioned on once. A learned rule chooses which residual
directions survive compression. It sees only available observations and
public requests; a future request is unavailable until it arrives.

The representation must define a coherent positive-semidefinite covariance
across queried points, with discarded uncertainty explicitly represented.
Positive marginal variances alone are insufficient. No exact-posterior or
calibration claim follows from that constraint. Changing a basis must transport
or explicitly approximate old information, never silently re-count evidence.

Use a small noisy spatial GP field as the initial CPU task. Stream observations,
then reveal a choice among four short paths of four or eight locations and a
safe fallback. Define failure by a threshold on cumulative path exposure,
with a disclosed path cost. The Gaussian law of that linear functional gives
a tractable decision reference: its variance includes cross-location
covariances. This avoids a pointwise task where correcting only marginal
variance could suffice. The nonlinear field, observation stream and request
geometry are public task assumptions, not inferred physical discoveries.

Freeze one residual-state budget and training recipe before collection. Train
on independent fields; evaluate fresh fields, longer histories and a separately
reported change in request geometry. Do not use the parent evaluation cohort
for fitting, policy selection or threshold choice.

## Controls and stop rule

Include proper FIC, an analytic sparse online GP, and the identical residual
representation with fixed or variance-based retention. Match their available
evidence, kernel knowledge and total state bytes. A full GP is the larger-memory
reference; always taking the safe action is a decision control. Charge all raw
observations retained, inducing coordinates, covariance/information matrices,
indexes, caches and policy weights; disclose peak temporary workspace and time.
A fixed active state with an unbounded archive is not bounded total memory.

Precommit decision regret, predictive log loss and interval coverage alongside
storage and runtime. Advance only if learned retention improves decisions over
the strongest matched analytic retention control without sacrificing the
declared predictive-quality tolerance. Numerical thresholds need a new protocol,
not retrospective selection. If analytic retention matches it, stop the learned
mechanism. A second environment is required before a general architecture claim.

## Closest prior art and novelty boundary

- [Kalman Delta Networks](https://arxiv.org/html/2609.07816v1) already couples
  residual writes to uncertainty and studies information lost by diagonalization.
- [Kalman Linear Attention](https://arxiv.org/html/2602.10743v1) provides parallel
  Bayesian recurrences; its uncertainty applications were not empirically established.
- [Online interdomain GP memory](https://arxiv.org/html/2502.08736v1),
  [sparse online GPs](https://eprints.soton.ac.uk/259182/), and
  [low-rank Kalman filtering](https://proceedings.mlr.press/v232/chang23a.html)
  already cover recurrent probabilistic compression and structured covariance.
- [Value-directed belief compression](https://papers.nips.cc/paper_files/paper/2002/hash/14ea0d5b0cf49525d1866cb1e95ada5d-Abstract.html)
  predates this proposal: preserving decision-relevant information is not new.
- [Dynamic Compression](https://arxiv.org/html/2608.17896v1) retains raw history
  for selective rescanning; [RISE](https://arxiv.org/html/2608.20430v1) allocates
  imagination by expected planning benefit. Neither makes a new selector sufficient.

The possible contribution is a specific, tested retention update that improves
the decision-quality versus total-resource tradeoff. Gaussian conditioning,
FIC, a Bayesian last layer, replay, or biological terminology alone is not it.
