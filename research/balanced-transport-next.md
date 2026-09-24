# Test persistent transition balance against matched initialization

Prospective mechanism note, September 24, 2026. The completed
[saved-state diagnostic](finite-transport-diagnostic-results.md) associates
accurate fits with more balanced transitions and stronger weak directions.
It does not establish why the fits differ. This note specifies the next
engineering question; no new learning run is registered or admitted here.

The first numerical step is now [qualified on 42 fabricated-input checks](finite-balanced-transition-qualification-results.md):
a fixed 64-sweep log-domain primitive with verified finite-computation gradients.
Model integration and the learning comparison below remain prospective.

## Hypothesis and comparison

Would constraining each action's learned transition T to be doubly stochastic
reduce poor fits under the same training-time allowance? The current model
enforces column sums of one. The candidate would also enforce row sums of one,
so T preserves a uniform vector. Keep emissions, hazards, the learned cost
head, recurrence, likelihood objective and H1/H2 training targets unchanged.

Only T is balanced. The survival-weighted operator
`S = diag(1-h) @ T` must retain state-dependent mass loss; balancing S would
change the event model and is not the proposed intervention.

Three prospective arms separate the initialization and ongoing constraints:

1. **Original free transition:** existing column-softmax initialization.
2. **Matched free transition:** initialize column-softmax logits from the
   balanced candidate's initial T, then allow unrestricted column-stochastic
   training.
3. **Persistently balanced transition:** apply differentiable balancing at
   every probability construction throughout learning and inference.

All three share initial emission, hazard and cost fields. The second and third
must also pass forward agreement at initialization to a declared numerical
tolerance, including reset conditioning, observed and blind forecasts, found
probability and cost outputs. No parent checkpoint is a warm start.

Use one learning schedule across these arms so transition parameterization is
the intervention. Prefix-then-joint is the prospective schedule because the
completed time-allocation study already tested it. Charge construction,
balancing, residual checks, backward passes and discarded late updates to the
same eligibility window. Record complete elapsed time and executed normalization
work; equal wall-time allowance does not mean equal updates or FLOPs.

All arms store **352 raw parameters**; that does not establish equal effective
capacity. Four unconstrained column-stochastic 8x8 matrices have 224 degrees
of freedom; under exact doubly stochastic constraints they have 196. The
finite numerical implementation enforces the additional balance constraints
to a residual tolerance, not as an exact symbolic projection. A comparison
must state this structural difference.

## Numerical qualification before learning

Start with ordinary log-domain Sinkhorn row and column normalization. Declare
the normalization order, finite iteration rule, residual tolerance and cap
before any learning experiment. Differentiate the executed finite sequence;
do not claim an exact infinite-iteration derivative. Reject nonconvergence,
underflow and nonfinite values without clipping or switching to a free model.
If an adaptive stopping rule is used, its discrete iteration decision is not
itself a differentiable operation and must be disclosed.

Fabricated checks should cover uniform transport, a positive
permutation-plus-uniform matrix, asymmetric positive inputs, large representable
logit offsets, row/column residuals, strict positivity, finite-difference
gradients, state relabeling, unchanged input storage, and explicit cap failure.
The probability readout and hazard mass accounting must remain unchanged.
First qualify this primitive; then qualify model integration independently.

Integration must also replace the prefix bridge's direct column-softmax
calculation: its transition log prior must use the same balanced log
probabilities as the forward filter. Preserve the existing 0.001 prefix-only
prior and the absence of that prior from joint minibatch training. Reusing the
old helper unchanged would optimize a different transition from the one used
for inference.

A subsequent learning protocol must freeze fresh data and seeds, all three
arms, the schedule, resource bounds and a decision-based continuation rule.
Save every initial, boundary and final model before untouched evaluation.
Report every seed and require reliability as well as mean benefit against
both controls. Geometry alone cannot pass the decision-performance rule.
No extra H4 supervision or closed-evaluation tuning is part of this proposal.

## Prior art and scope

[Mena et al., Gumbel-Sinkhorn Networks](https://arxiv.org/abs/1802.08665)
already use the continuous Sinkhorn operator as a differentiable relaxation
for latent permutations. Our first implementation would use deterministic
balancing, not claim to invent that operator or reproduce their experiments.

[Arjovsky et al., Unitary Evolution RNNs](https://proceedings.mlr.press/v48/arjovsky16.html)
motivate controlling recurrent geometry for long dependencies.
[Saxe et al.](https://arxiv.org/abs/1312.6120) analyze learning dynamics and
special initial conditions in deep linear models. These results have different
assumptions from our positive probabilistic filter. A generic orthogonal or
unitary matrix is not a stochastic transition; a square nonnegative orthogonal
matrix is a permutation matrix. Balancing is a weaker constraint and still
allows the completely mixing uniform matrix.

The benchmark's true transition is permutation-plus-uniform, so this test
injects a correct structural prior. Even a successful result would need a
later world with nonuniform stationary mass, fresh replication and a second
environment before a broader recurrent-model claim. It supplies no evidence
for a biological connectome. Biological versus rewired sparse topology remains
a separate comparison once a useful learning mechanism is established.
