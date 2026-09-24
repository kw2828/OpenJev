# Preserve useful action differences before adding biological wiring

Primary papers checked September 24, 2026. This is a prospective reading note,
not a registered experiment or a claim of novelty. It does not change the
[time-allocation comparison](finite-training-allocation-protocol.md) or reopen
earlier failed studies.

The present recurrent models can improve public-history likelihood while
remaining unreliable at ranking future actions. A later architecture test
needs an explicit hypothesis about which decision-relevant distinctions are
lost, and how a proposed mechanism preserves them.

| Source | Relevant method | Consequence for OpenJev |
| --- | --- | --- |
| [PhyLatent, sections 3 and 4](https://arxiv.org/html/2608.05720v1) | Studies failures of physical invariance, state distinction and alternative-action separation despite global latent non-collapse. Its separation margin depends on action differences; other losses use physical training targets. | Measure whether action branches with different outcomes become indistinguishable. Hidden physical labels are extra supervision and must be shared across controls or disclosed. An action difference alone need not imply an outcome difference in our discrete world. |
| [TD-JEPA, sections 3 and 4](https://arxiv.org/html/2510.00739) | Uses temporal-difference latent prediction and policy-conditioned predictors for longer-term dynamics and zero-shot RL. Its theoretical results have specific representation and approximation assumptions. | Compare a direct predictive representation against repeatedly rolling forward a learned latent state. Recurrence, TD targets and latent prediction are established ingredients; the paper does not automatically validate a history encoder for our partially observed task. |
| [Physically Grounded JEPA](https://arxiv.org/html/2609.03565v1) | Adds inverse-dynamics and physical-state alignment objectives for goal-conditioned robotic planning. | Adding state or action supervision is already prior art. Preserve the same supervision and planner across methods; a better probe is insufficient without better control. |
| [Predictive Representations of State](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf) | Describes state through predictions of action-conditioned observation tests. | A finite predictive test bank needs a sufficiency argument; choosing tests does not by itself identify a useful state. |
| [DeepMDP](https://proceedings.mlr.press/v97/gelada19a.html) | Learns representations preserving reward and next-state prediction, with connections to behavioral equivalence. | Separating action labels is not bisimulation. Partial observability requires sufficient histories or state estimates. |
| [Successor Features](https://proceedings.neurips.cc/paper/2017/hash/350db081a661525235354dd3e19b8c05-Abstract.html) | Factors policy-conditioned discounted feature occupancy from linear reward weights. | Our committed action-block endpoint forecasts are a different target; existing transfer guarantees cannot be imported unchanged. |

These papers' reported results have not been reproduced here. The source review
does not establish that their methods solve OpenJev's current failure.

## A specific later mechanism question

Our proposed question is whether a recurrent representation preserves
**differences between action consequences that matter to the decision**, while
treating equivalent consequences alike. This is a hypothesis to test, not an
established new algorithm. The predictive-state, successor-feature and
bisimulation connections above are substantial prior art. A more precise
mechanism and broader comparison would be needed before any novelty claim.

One tempting objective is redundant. For K action prediction errors e,
`sum(i<j, (e_i-e_j)^2) = K*sum_i(e_i^2) - (sum_i e_i)^2`.
For centered errors this is K times ordinary squared error. With four actions,
a uniformly weighted pairwise squared-difference loss therefore adds no new
constraint beyond rescaled centered cost MSE. Removing a common cost offset
does not by itself establish a new predictive representation.

First qualify a diagnostic on fabricated worlds with analytically distinct
and equivalent action branches. A model should not receive credit merely for
separating every action label. Measure prediction error and action-ranking
error separately, including states where different actions have equal
consequences. Do not infer identifiability from latent variance.

If that diagnostic reveals a material failure in a fresh, registered cohort,
compare the same recurrent backbone under ordinary predictive learning and
an outcome-conditioned relational objective. Retain a direct observable
prediction control. Match training information and measured computation,
and use decision utility as the advance criterion. Training-only targets must
not enter the deployed state update. Any new teacher queries or simulator
branches belong in a separate prospective protocol. Closed evaluation sets
remain excluded from training and selection. The earlier failed readout
replication's longer-horizon-supervision follow-up remains closed. This note
does not authorize an H4-supervised continuation or select a new algorithm.

## Where connectomes would enter

A biological graph becomes a separate mechanism test after the learning and
information controls work. Compare it with degree-preserving rewires, a sparse
nonbiological graph and a conventional recurrent control, using identical
interfaces, parameter allowances, supervision and execution budgets. Keep
readout capacity and temporal gating explicit. Existing OpenJev connectome
comparisons have not established biological-topology superiority; this note
does not replace those outcomes.

The eventual robotics test must hold the planner fixed and evaluate a sensing
or dynamics shift as well as the original task. A useful result would connect
a controlled representation intervention to improved action selection and
transfer. The contribution needs to be that demonstrated mechanism and its
measured benefit.

[Earlier connectome and robotics review](connectome-world-model-robotics-reading.md) ·
[Existing learning program and completed failures](connectome-learning-program.md).
