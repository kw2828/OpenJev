# Query-time memory: stronger controls before a new architecture

Reading and design note, 24 September 2026. The observation-reliability branch
remains closed. This is a distinct proposal motivated by OpenJev's interface:
store observations, then answer decisions specified later by the application.
No new neural training or scientific efficacy claim is made in this note.

## What changed the research direction

[Dynamic Compression in Recurrent Networks](https://arxiv.org/html/2608.17896v1)
lets a recurrent model revisit selected past examples after a task becomes
known. Its raw prefix remains stored, so small active state is not small total
memory. Selection uses either task-index supervision or a codebook distilled
from a repeat model. The experiments concern synthetic linear function reuse.

The [official repository at the reviewed revision](https://github.com/jyopari/dynamic-compression/tree/5c61e1d68a1ac99b1613e1ea52986c64bed25019)
has MIT-licensed code. No trained checkpoint was found in that Git tree or its
release listing during this review. That observation does not prove no model
exists elsewhere. Reproducing its training would be separate substantial work.

The immediate [strong-control diagnostic](function-reuse-reference-protocol.md)
asks whether cached block least squares already solves the disclosed noiseless
linear subproblem. Public block IDs and overdetermined examples make this a
natural control. Its function-class knowledge and O(K*d*d) map bank differ
from a learned fixed-size recurrent state. A favorable result would not refute
the paper's capacity study or be a matched neural-model comparison. It would
tell us not to use this task's raw accuracy alone as evidence for our novelty.

## A mechanism worth defining more precisely

The prospective question is whether **query-conditioned computational refinement
can improve decisions without treating replay as new evidence**. A candidate
would keep three explicit components:

1. An immutable archive of uniquely identified real observations and actions.
2. A small learned recurrent summary used to select relevant archived blocks.
3. A query-local predictive state rebuilt or corrected using those unique blocks,
   with separate counters for real environment time and internal computation.

This is an architectural question, not an implemented model. Merely adding
retrieval, a second pass or a confidence head is already covered by related
work. Repeated optimization on a fixed objective is legitimate. Multiplying
the same observation's likelihood twice because it was replayed is a different
operation and generally yields the wrong posterior. A world-model replay must
also not advance the physical-time transition as if a new action occurred.
Neither issue has been demonstrated in the cited neural paper; they are design
requirements for our proposed probabilistic extension.

Before neural training, qualify a probabilistic reference on an identifiable
noisy function family with known likelihoods. Replaying or reordering the same
unique evidence should preserve its exact posterior. New independent evidence
should change it. A controlled duplicate-likelihood implementation can test
the detector but is a failure fixture, not a serious competing algorithm.

Only then consider a learned uncertain nonlinear function/world-model task.
Define the public request, dynamics, outcome law and decision costs before
collection. A task must need predictive generalization beyond a lookup table,
and offer headroom beyond the appropriate regression or Bayesian reference.
Noise and nonlinearity by themselves do not establish that headroom.

## Baselines and the possible contribution

[Gated DeltaNet](https://arxiv.org/abs/2412.06464),
[TTT-Linear](https://arxiv.org/abs/2407.04620), and the cited selective-replay
model are relevant learned controls. A shared encoder and the same examples
must feed ordinary retrieval plus local regression, so better observations
cannot masquerade as better memory. Include always-replay, never-replay and
simple residual-triggered replay, and charge selection and archive search.

Measure decision regret, predictive log loss, stationary retention and shifted
task performance against total storage and compute. Count the archive, block
statistics, indexes, active state and transient workspace separately. Query
answers and true function IDs must never enter deployed retrieval. A fixed
training schedule and an untouched evaluation set are required; do not choose
the best inference budget retrospectively and call it a policy.

A worthwhile contribution would be a specific learned refinement rule that
improves this quality-resource frontier over these controls, with causal replay
and evidence-identity interventions explaining why. Biological connectivity
would be a later matched structural factor, alongside sparse and rewired
controls. Neither the current baseline diagnostic nor this design establishes
a new architecture, connectome advantage, calibration guarantee or ICLR result.
