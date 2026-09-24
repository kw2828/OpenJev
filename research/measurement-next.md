# Beyond the supplied-law grid

Prospective design only, 24 September 2026. This note identifies a possible next test; a separate protocol must be frozen before running it.

**Budget correction, 24 September 2026:** the sketch below is an unimplemented proposal. Its 1,008-byte table omits inputs waiting for delayed labels. The [subsequent sensor-screen protocol](sensor-screen-protocol.md) uses all seven public inputs and a 24-hour delay; its raw pending-input queue alone costs 1,344 bytes. The original table is not a feasible deployment budget for that interface. A learned comparison must be rebudgeted and separately registered.

The [completed consolidation study](measurement-results.md) passed 26/26
conditions. Spectral118 had zero measured regret on LONG and LONG_SHIFT, while
DCT118 had only 2.65e-13 and 1.05e-14. The static, supplied-law grid is effectively
solved at this decision resolution. Spectral inference took roughly 40/44 ms per
long context, versus 5.7-29.8 ms for the raw controls. This is a useful analytic
baseline, not a learned architecture or a speed result. Do not tune more memory
variants against these evaluated fields. The original novel-architecture goal remains unmet.

The next research question must remove the known kernel or finite grid, ideally
both, and face chronological real data or a concrete robotics-memory task with
strong conventional baselines. The bitmap in this experiment encodes every past
input location cheaply. On a continuous domain, measurement sums alone cannot
reconstruct arbitrary RBF cross-covariances with forgotten inputs.

One prospective falsification option is chronological sensor calibration on
[UCI Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality), which has
real sensor drift and reference-analyzer measurements. Use three public inputs,
such as a sensor response, temperature and humidity, to predict benzene before
its reference label is revealed after a fixed delay. Train only on an earlier
period; freeze normalization, features, alarm/defer costs and thresholds before
later blocks. Exclude other reference-analyzer columns from inputs. This would
be an artificial bounded-risk decision task on real measurements, not a
validated health-warning service. Missing-value handling must be frozen too.

The candidate would learn which regression information to retain: freeze 16
training-learned RBF features, append each available row `[phi(x), y]` once to a
three-row sketch, and let a small rule choose a singular direction to discard.
Future requests never enter writes. A possible 17-weight score correction acts
on the squared components of each current right singular vector. This is a
specific mechanism to falsify, not evidence that the rule is useful or new.

| Persistent float64 state and parameters | Bytes |
| --- | ---: |
| Three-by-17 augmented measurement sketch | 408 |
| Sixteen three-dimensional centers, length, amplitude and noise | 408 |
| Seventeen learned compression weights | 136 |
| Three input means, three scales, event counter | 56 |
| **Total** | **1,008** |

Every parameter is charged per context; no optimizer, input archive or cached factor remains. Use target values in their original units unless extra target
normalization storage is explicitly charged. Workspace and runtime are separate.

Required controls are a four-row Frequent Directions sketch at the same bytes,
the same three-row rule with its learned correction disabled, exact Bayesian
sufficient statistics with 11 features and a packed symmetric matrix, and coverage/recent retention of 17
continuous input-label tuples using the same 16-feature model. Larger-memory
exact 16-feature inference diagnoses approximation error. Stop if the learned
rule fails to beat the strongest budgeted control on untouched chronological
blocks, or worsens proper predictive scores while improving decision cost.
With `z=A y`, a new historical projection `A' y` is recoverable only if `A'=R A`.
Appending a new event adds information; rotating the remaining rows cannot
restore a discarded direction. The rule must compress only the current sketch
and new row. Its regularized Gram matrix can define a positive Gaussian working
model, but label-dependent compression is not automatically an exact posterior
conditional on the selection history or a calibration guarantee.
[CaGP](https://arxiv.org/abs/2411.01036) already learns projection actions;
[ALPaCA](https://arxiv.org/abs/1807.08912) learns features and Bayesian priors;
[Frequent Directions](https://arxiv.org/abs/1501.01711) is established streaming
matrix compression. Learned row-space selection alone is weak novelty. A real
contribution would require a distinct mechanism and reproducible decision and
resource gains beyond these controls, followed by independent task validation.
