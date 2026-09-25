# Next comparison: learn only what the linear model misses

**Prospective, not registered or executed.** The
[multivariate correction pilot](fsm-correction-results.md) fails its rule.
Selective routing loses to dense correction in all three seeds; the best
FIT-fitted VARX control is much more accurate and faster. The current routing
candidate stops here. These results favor a stronger starting predictor, but
are not proof that a nonlinear extension will help.

A practical next comparison is a frozen linear backbone with a small,
zero-output-initialized residual network. At initialization both learned arms
must reproduce the same VARX forecast, within a qualified numerical tolerance.
Compare **output-only correction**, where the linear trajectory evolves
independently, with **feedback correction**, where the corrected output enters
the next recurrent lag state. Keep the linear coefficients frozen, and match
trainable parameters, seeds, observations, input prefixes, loss and update budget.
Retain the untouched VARX backbone and the existing GRU results as controls.

The narrow question is whether propagating learned nonlinear corrections
improves later forecasts enough to pay for recurrent computation. It is not a
new architecture claim. [NL-LFR with linear initialization](https://arxiv.org/abs/2004.05040)
already uses a linear dynamics starting point and learned nonlinear feedback;
[dynoNet](https://arxiv.org/abs/2006.02250) composes differentiable dynamical
filters with nonlinearities. [SUBNET](https://arxiv.org/abs/2210.14816) provides a
relevant encoder and multistep-training reference. Borrowing these ideas creates
stronger baselines; novelty would need an additional precise contribution.

Freeze the exact backbone selection, residual architecture, training budget,
causal prefix initialization and success margins before fitting. The current
DEV realizations are exposed and must remain labeled development data. A
positive result must beat both the linear model and output-only correction,
with full costs and all seeds retained, before designing an untouched test.
No official-test or 300 mV measurements are admitted by this note. A separate
FIT-only implementation of the authors' BLA28/LFR references remains necessary
before making a broad competitiveness claim.
