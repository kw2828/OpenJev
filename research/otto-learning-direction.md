# Learning the control objective before compressing memory

The completed scalar-return study fails all 54 conditions. Its minimum-linear family has lower scalar validation MSE than both neural controls but worse autonomous success. These are [audited observations](otto-return-value-results.md), not evidence identifying a cause. The [saved consistency diagnostic](otto-return-consistency-design.md) covers only two early teacher trajectories and cannot make that causal attribution either.

## What the closest prior work changes

Loisy and Heinonen train a value network using Bellman targets, replay, a delayed target network and exploratory collection. Their algorithm evaluates possible successor observations rather than using only realized teacher returns. Their larger isotropic network has three hidden layers of 1,024 units, far larger than our width-eight pilot. The paper also compares neural learning with piecewise-linear POMDP solvers. This makes target construction, visited-state coverage and capacity distinct questions, not evidence that adding recurrence will fix our result. [Primary paper, Algorithm 1 and Tables 5–6](https://auroreloisy.github.io/papers/Loisy2023a_EurPhysJE_drl-benchmark.pdf).

Grimm and colleagues define value equivalence through matching Bellman updates for specified functions and policies. This motivates testing whether a representation preserves useful planning calculations, rather than relying on state-prediction or scalar-regression accuracy alone. It does not show that our representation is value equivalent. [Primary paper](https://arxiv.org/abs/2011.03506).

Bellemare and colleagues study operators that preserve optimality while increasing action gaps, making greedy choices less sensitive to approximation error. That is relevant prior art for any proposal centered on more reliable action differences; merely adding an action-gap loss would not be a new contribution. Its assumptions and operator cannot be transferred to our signed, floored cost calculation without a separate derivation. [Primary paper](https://arxiv.org/abs/1512.04860).

There is also a target/structure distinction. Finite-horizon conditional plans have values linear in belief; selecting the best plan produces the familiar convex reward envelope, or concave cost envelope. A belief-dependent heuristic need not select that envelope. Consequently, concavity of optimal cost does not imply concavity of our teacher's cost. This is a mathematical observation, not an empirically established cause of this failure. [Hansen, Sections 2–3](https://arxiv.org/pdf/1301.7380).

Our eight unrestricted regression planes have no associated action and observation-conditioned successor plan. They therefore are not policy-realizable alpha vectors and provide no certified planning bound. Point-based backups connect the value representation to such conditional plans. A minimum operation alone does not reproduce that machinery. [Pineau, Gordon and Thrun, PBVI](https://www.cs.cmu.edu/~ggordon/jpineau-ggordon-thrun.ijcai03.pdf).

## Concrete next learning comparison

First finish the saved diagnostic. Then specify a separate experiment that changes the target recipe while keeping the model families, initial checkpoints, data access and evaluation protocol matched. A natural baseline is ordinary fitted value learning with a delayed target network and explicit observation backups. Retain ordinary and homogeneous MLP controls alongside minimum-linear values. An equal-work continued Monte Carlo fit controls for additional optimizer updates; separately count all target-network forwards and branch construction.

Start with the ordinary MLP as the positive-control learning test. If each representation later supplies its own delayed target network, its targets differ: that comparison measures representation plus learning dynamics. An architecture-only comparison instead needs a common, frozen target sequence. Do not describe matching initial data or optimizer counts as matching the targets or total computation.

Separate two questions instead of changing both at once:

1. Can branch-based targets improve choices on a fixed training cohort, without introducing new rollout data?
2. If not, does prospective exploratory collection provide necessary state coverage when every family receives the same collection and training budget?

Use fresh, fixed evaluation cases. Previous EVAL trajectories remain exposed diagnostic data and must not become targets or an untouched test set. Do not choose new hyperparameters from those results. Preserve failed searches and charge all collection, training and controller costs. Any competence improvement must still be distinguished from a utility-versus-compute improvement.

## Architecture question, conditional on a competent learner

A useful later hypothesis is a small recurrent controller whose sparse state updates preserve successor values and action ordering. Dense recurrence, ordinary value heads and exact-belief control are necessary controls; shuffled or degree-matched wiring is necessary for a biological-graph claim. A successful result must transfer beyond these two diagnostic episodes and beyond one known sensor setting.

This is a proposed research direction, not an executed training experiment, new architecture result or revision of any frozen condition. It does not admit compact-memory escalation on the basis of the failed scalar-return pilot.
