# Stop this deletion recipe; test better information retention

Prospective direction after the closed [retention study](retention-results.md),
24 September 2026. This note registers no experiment or continuation.

The 33-parameter residual-KL deletion recipe stops here. Its frozen outcome was
`DO_NOT_ADVANCE_RETENTION`: 12 of 42 conditions passed. Do not tune its policy,
reward, temperature or memory size against these evaluation fields, or relax
the failed rule. Preserve this study and the earlier failed studies unchanged.

Learning did improve deletion within the same eight-point representation:
mean regret fell 17-28% relative to KL8. That is a useful local result, but the
representation and policy together were dominated under the common byte cap.

| Population | Learned mean regret | KL8 | Coverage41 | Full-GP diagonal diagnostic |
| --- | ---: | ---: | ---: | ---: |
| BASE | 0.008394 | 0.011598 | 0.001147 | 0.000080 |
| SHIFT | 0.011735 | 0.014150 | 0.002916 | 0.000061 |
| LONG | 0.011082 | 0.013771 | 0.003688 | 0.000105 |

Coverage41 retains raw coordinate/label triples and conditions on those rows.
It used 1,024 logical bytes versus the learner's 1,008, achieved 3.0-7.3 times
lower regret, and was faster in every reported timing probe. Mean learned
stream-plus-four-request times were 14.63/13.21/40.02 ms for BASE/SHIFT/LONG,
versus 1.71/1.47/7.71 ms for Coverage41. These are small-context timing probes,
not a general speed claim; logical state excludes native workspace and RSS.

The larger-memory diagonal diagnostic preserves full-history posterior means
while dropping within-path off-diagonal covariance. Its tiny regret makes
covariance loss alone an inadequate explanation for the learner's much larger
decision gap in this task. It does not prove that mean error is the sole cause:
compression also changes marginal uncertainty and their interaction with action
selection. The learner's exposure MSE was worse than KL8 in all three populations
despite lower regret. Neither metric alone identifies the causal bottleneck.

The closest prior art also limits the claim. [McIntire et al. (2016)](https://ai.stanford.edu/~ermon/papers/sparse-gp-uai.pdf)
already use task-weighted KL in budgeted online GP reduction for optimization.
[Uhrenholt et al. (2021)](https://proceedings.mlr.press/v161/uhrenholt21a.html)
learn discrete inducing-point inclusion with score-function gradients and a
variational objective. [Moss et al. (2023)](https://proceedings.mlr.press/v206/moss23a.html)
allocate inducing points using task utility and diversity. Learned selection,
REINFORCE and decision-aware sparsification are not new architectural ideas.

A separate follow-up is worthwhile only if it targets the stronger raw-memory
baseline: how can a compact state retain more useful measurement information
than 41 well-spread observations? One candidate is noise-aware measurement
consolidation or a Gaussian memory over linear functionals, retaining evidence
about regional averages instead of paying for a dense posterior over a few
point values. This is a hypothesis, not evidence that consolidation will win;
inter-domain inducing features are [established machinery](https://papers.nips.cc/paper/3876-inter-domain-gaussian-processes-for-sparse-inference-using-inducing-features).

Before more RL, test an analytic task-weighted compression rule against
Coverage41, Recent41 and KL9. Any request weighting must use a declared public
request distribution, never the unrevealed evaluation paths or forgotten data.
Count every coefficient, covariance entry, functional definition, policy weight
and persistent cache; report update and query costs as well as decision regret,
mean accuracy and uncertainty quality. No hidden full-history state is allowed.

Use fresh fields and a separately frozen rule. Keep the original task as a
comparison; do not manufacture a covariance-heavy variant to rescue this result.
A meaningful architecture direction needs a substantial advantage over strong
raw-memory controls at an honest resource budget, followed by independent task
validation. This failed control gap does not provide that evidence.
