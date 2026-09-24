# Next diagnostic: solve the decision readout with dynamics frozen

**Proposal only, not registered or executed.** The [factorized-dynamics comparison](finite-factorized-dynamics-results.md) failed its short-horizon and blind-extrapolation criteria. All original verdicts stay closed; the earlier replication's H4 training follow-up remains unadmitted. Do not select the successful seed, extend its training or replace a failed fit.

The smaller model improves several averages, but seed 426261002 has eight-step regret 0.388265 while seed 426261001 reaches 0.010825. Both pass the observed-event KL criterion. Accurate event predictions alone have not established reliable decision costs. The next diagnostic should separate a readout-fitting problem from errors that persist in the learned recurrent states.

## Intervention

Retain **all nine completed checkpoints**, with explicit source, checkpoint and original-study hashes. Freeze transition, emission, hazard and prefix-filter parameters. Materialize each model's latent prior states from the original TRAIN public histories and H1/H2 action blocks. Observed states may consume only the observations permitted by the original chronological interface. No hidden world state or oracle posterior is a learner input.

Refit only the cost matrix with a constrained convex problem:

`C[:,s] = 0.25 - P[:,s]`, with `P[:,s] >= 0` and `sum_a P[a,s] = 1`.

Minimize the original TRAIN blind-cost mean squared error **plus** observed-cost mean squared error, retaining the original action/horizon normalization and endpoint population. With dynamics frozen, survival, event cross-entropy and prefix likelihood are constants. Do not introduce H4/H8 training labels. Keep an unchanged-head arm for every checkpoint.

This squared-loss objective is convex in P. Require a predeclared numerical optimality check, such as projected-gradient/KKT residual plus objective tolerance, within a fixed solver budget. Report every solver failure and original-versus-solved TRAIN objective. A failed solve supplies no best-readout claim and must not be silently restarted.

The closed simplex permits exact boundary vertices that finite softmax logits approach only in a limit. Disclose this closure difference; a gain cannot be attributed solely to changing the optimizer. The intervention is an established linear readout diagnostic, not a new learning algorithm. It also differs from the earlier [OTTO direct readout study](otto-direct-readout-results.md) in model, data and task; that study's verdict remains unchanged.

## Fresh evaluation and scope

Freeze the solver, original nine-checkpoint roster, compute caps, tolerances and fresh development namespace before loading models or creating arrays. Every TRAIN-only solve must finish before the new development pool is generated. This is further development after inspecting previous results, not untouched final confirmation.

Evaluate original and solved readouts on the same fresh public histories and H1/H2/H4/H8 targets for every model. Report all existing absolute criteria and same-checkpoint decision regret/MSE differences. Recompute event and survival predictions to verify that they stay unchanged; record prediction equality tolerances and separately charge checkpoint loads, state extraction, solver work, validation and evaluation.

If fitting costs improve on TRAIN but fresh decision performance does not, readout refitting has not resolved the failure. If decision performance improves reliably, the evidence supports a readout/optimization bottleneck under this constrained family. If it does not, the remaining error persists after this linear-head diagnostic; that does not prove information is absent from the latent state or that non-linear heads cannot help.

Do not advance a model on a favorable mean when an applicable all-seed absolute criterion fails. Scenario shifts, a second environment and a distinct mechanism would still be needed for a broader paper claim. Conformal sets, connectome masks and RL updates should not be added to this diagnostic without a separate question they can answer.
