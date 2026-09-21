# Compact evidence memory: mechanisms and the controls they require

21 September 2026. Design note, not a trained-model result or an admitted
learning campaign. The [53x53 comparison](otto-large-memory-results.md) failed
its fixed consistency condition. The [saved-history diagnostic](otto-memory-evidence-protocol.md)
asks which older observations change decisions; it cannot establish an
autonomous improvement for a new controller.

## Retention should follow the observation model

For the supplied static-source model, Bayes filtering accumulates spatial
log-likelihoods. On cells not excluded by the permanent visitation ledger,

\[
\log b_t(s)=\log b_0(s)+\sum_{\tau=1}^{t}\log P(h_\tau\mid s,x_\tau)-\log Z_t.
\]

Here the initial public hit is already in `b_0`, `x` is a public sensor position,
and the source location `s` indexes hypotheses rather than an observed label.
A measured zero count contributes `-mu(s,x)` under the Poisson sensor. It is
evidence against nearby source hypotheses, not a missing observation. Counts
one and two contribute `h log(mu) - mu - log(h!)`; the saturated category must
use `log P(K >= 3)`, not the expression for exactly three. All formulas apply
only on surviving cells; exact visited-cell zeros remain outside a positive
latent decoder.

This identity suggests a compact spatial accumulator before a generic learned
forgetting gate. In a stationary known model, arbitrary decay discards valid
evidence. Learned gates would need to compensate for a demonstrated compression
error or address a separately defined nonstationary task, rather than being
justified by biological terminology alone.

## Relevant prior art

| Primary work | Mechanism to borrow | Limit and required control |
| --- | --- | --- |
| [Álvarez-Salvado et al., eLife 2018](https://elifesciences.org/articles/37815) | Separate temporally filtered ON and OFF responses describe walking-fly responses during odor and after its loss. | Their behavioral setting includes wind and continuous odor signals. It does not identify a connectome-based spatial posterior. Compare separate channels with tied channels and simple timers before attributing gains to biology. |
| [Roy, Gordon and Thrun, JAIR 2005](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume23/roy05a-html/node4.html) | E-PCA compresses beliefs using an exponential decoder and an unnormalized KL objective. | Their [belief update](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume23/roy05a-html/node6.html) reconstructs a full belief, applies Bayes and compresses again. Compact state alone does not establish cheaper online updates. E-PCA and fixed additive compression are serious baselines. |
| [Shaj et al., CoRL 2020 / PMLR 2021](https://proceedings.mlr.press/v155/shaj21a.html) | Action-conditional recurrent Kalman networks combine action-dependent latent prediction with uncertainty-weighted observation updates. | The evidence concerns robot dynamics, not this multimodal source posterior. A zero odor reading must not be treated as missing data, and a Gaussian latent is not guaranteed to preserve multiple spatial hypotheses. |

These papers motivate implementation choices and rule out broad novelty claims
for recurrence, separate sensory channels or compressed beliefs. The additive
filter identity above is a consequence of the supplied probabilistic model,
not a new research contribution.

## A falsifiable model question, if separately justified

Can a fixed-size recurrent representation retain useful spatial evidence with
lower total cost than an exact belief grid, while avoiding the rare long searches
caused by truncating history?

Use a shared decoder, permanent visitation ledger, observation interface and
space-aware action selector. Separate three questions rather than changing all
of them together:

1. **Representation:** full filtering versus a fixed basis and E-PCA-style
   compressed beliefs. Include decoding and any full-space updates in cost.
2. **Update:** fixed additive evidence versus a learned tied gate and separate
   detection/absence gates, with a parameter/state-matched GRU control. The
   public position or exact egocentric shift should supply known geometry.
3. **Wiring:** only after an update advantage exists, compare biological wiring
   with degree-preserving rewires and dense recurrence using the same update
   equation, input mapping, decoder and state budget.

Use training trajectories that are disjoint from evaluation, identical fitting
and tuning budgets, autonomous rollouts, all fitting seeds, and a prospective
scenario shift. Measure belief log loss, action decisions, search-time tails,
success, total latency and total retained state. A hidden full-resolution cache
or an uncounted visited mask cannot support a compact-memory claim. Better
teacher-forced agreement does not establish better search.

If fixed additive compression matches the learned updater, report the simpler
mechanism. If compressed belief quality improves without improving autonomous
utility at matched cost, stop the architecture claim. No training or new
simulator episodes are authorized by this design note itself; the next study
must stand on a separately stated scientific rationale and frozen protocol.
