# Spatial processing before belief compression

Status: conditional hypothesis, not an admitted training experiment. The paired
Bellman/Monte Carlo study is still running. None of its evaluation outcomes were
used to select this proposal, and its frozen protocol remains unchanged.

The completed [symmetry study](otto-symmetry-head-results.md) failed despite
retaining exact posterior inputs. In the [scalar-value study](otto-return-value-results.md),
lower scalar validation error did not imply better autonomous decisions.
These observations motivate investigating the readout; they do not establish
that representation, memory, or state coverage caused the failures.

The current [value model](../src/openjev/research/otto_return_value.py) compresses
11,025 spatial entries into eight linear measurements before applying a
nonlinearity. One candidate would instead apply a small shared nonlinear
function to each cell's probability, nearby probability mass at fixed scales,
agent-relative geometry, boundaries, and supplied sensing length, then pool
with probability weights. Keep the exact posterior and observation branches.
This is a spatial value readout, not a recurrent world model or a biological
network. Set-based architectures already have substantial prior art, including
[Deep Sets](https://arxiv.org/abs/1703.06114). Combining familiar components is
not itself a novelty result.

## What would distinguish a useful mechanism

The necessary controls are a wider ordinary MLP, a standard small CNN, the same
shared cell model without neighboring-mass inputs, and a small head on familiar
mass, entropy, and expected-distance summaries. Retain analytic control as the
competence anchor. Report complete computation and parameters; equal updates
or hidden widths do not imply equal work or capacity.

An optional synthetic probe can construct beliefs with matching current-model
measurements but different reference decisions. Matching only the current
state's eight projections is insufficient: deployment evaluates sixteen
successor beliefs. A policy-level collision must preserve each branch's raw
mass and projected unnormalized numerator, with the position, kernel, mass
floor, and action eligibility fixed. Numerical equality and reference decision
separation require independent verification. Such beliefs may be unreachable;
they cannot be reported as gameplay failures or evidence of their prevalence.

Any architecture comparison should use a common frozen target sequence, the
same data, and fresh evaluation cases. Independently generated targets would
also change the learning procedure. Wait for the current study's complete audit
before choosing between a readout experiment and a state-coverage experiment.

Drop the spatial-interaction explanation if wider dense processing, ordinary
convolution, the neighbor-free model, or the familiar-statistics head accounts
for its gains. Also drop it if lower fitting error does not improve autonomous
utility after counting all computation. Information loss alone is not evidence
that the discarded information is needed for this task. A recurrent or
connectome-based extension would require separate matched evidence.
