# Action-focused training before another memory architecture

Status: implemented and qualified; the [fresh twelve-fit study](otto-action-focused-protocol.md) has a completed worker receipt, but original supervisor closure is unverified after a session interruption. Its technical-completion condition remains false. A separate [saved-output diagnosis](otto-action-focused-interruption-protocol.md) is pending and cannot qualify the original run. Protocol and sources were frozen before collection in commit `1b61df2`. The [ranking diagnosis](otto-ranking-diagnosis-results.md) leaves the earlier failed studies closed.

## Question

Can a teacher-cost-sensitive training objective improve legal choices while retaining useful score forecasts for subsequent corrections? Aggregate boundary-pair MSE can improve while decisions regress, so merely giving those pairs more squared-error weight is not yet a demonstrated remedy.

The first comparison should keep the existing shared-output explicit model and ordinary GRU fixed. Shared-output GRU with prior supervision is the required strong control. Do not resurrect the failed separate-output recipe as the only comparator.

## Concrete loss candidate

For legal actions L, teacher costs q and predicted costs p, define c=q/64 and z=p/64. Let u be the uniform distribution over **exact** legal teacher minimizers. The finite-action specialization of [SPO+](https://arxiv.org/html/1710.08005v5#S3.SS2) is:

`SPO+(z,c) = max_{a in L}(c_a - 2 z_a) + 2 sum_a u_a z_a - min_{a in L} c_a`.

The proposed treatment adds this term with coefficient one to the unchanged nonquery legal-centered MSE and prior-query all-four MSE. The control retains both MSE terms alone. Use the existing sampled-TRAIN importance weights and fixed episode denominator, without renormalizing the realized sample. Preserve the same query schedule, targets, model initialization, sequence order and final-checkpoint rule within each paired comparison.

Coefficient one and scale 64 are prospective conventions, not empirically optimal values. The two terms have different homogeneity in score scale. Record their magnitudes and costs; do not select a coefficient from exposed VALID results. Teacher scores and weights are detached. A single legal action has zero decision loss; padding has zero loss and gradient.

This is an established objective, not a new RL algorithm. It targets teacher-cost regret rather than agreement, does not calibrate probabilities, and does not uniquely identify score differences. The retained MSE terms help constrain the scores used by recurrent correction, but their adequacy remains empirical. Deployment uses the existing float32 near-minimum rule, which differs from the exact optimization oracle in the SPO theory. Distributional consistency results are not automatically applicable to this correlated teacher-imitation task. [Theory and assumptions](https://arxiv.org/abs/2108.08887).

## Next controlled screen

Use two shared-output architectures, two objectives and three fresh paired fit seeds, totaling twelve fits. Reuse no exposed VALID trajectories for selection or promotion. Collect fresh TRAIN and VALID cases under a separately frozen protocol, close all fits before decoding VALID, and retain every fit and failure. Keep training exposure and optimizer updates matched, and measure actual training and deployed computation rather than calling equal updates equal compute.

Before execution, specify exact cohort identities, objective arithmetic, optimizer, budgets, loss diagnostics and the joint continuation rule. The rule must require improved later teacher-score gap in both settings with full/initial choice quality protected, using the ordinary-GRU comparison as a required control. A TRAIN-only gradient check can reveal implementation failure; it cannot select a winning scientific recipe. Nothing in this design retroactively changes the closed gates.

If both architectures benefit, the finding is an objective effect. If the ordinary GRU accounts for the gain, report it. A comparison with the established temperature-based [policy-distillation loss](https://arxiv.org/abs/1511.06295) would then test whether the mechanism is specific to SPO+; any temperature must be frozen without exposed-VALID tuning.

## Separate memory hypothesis

Only after resolving the objective question, test whether a query-written associative memory supplies a useful additional state. A candidate delta write is `M <- M + beta (v - M k) k^T`, with public-feature keys and teacher values supplied only at genuine scheduled queries. Episode resets, chronological carries, detach rules and memory costs must be explicit. Between-query labels must never write into the state.

Compare an ordinary shared GRU, a matched additive-write memory, and a delta-write memory with identical key/value access. Keep the objective fixed across memory arms. The [fast-weight delta rule](https://proceedings.mlr.press/v139/schlag21a.html) and [DeltaNet](https://arxiv.org/abs/2406.06484) are prior art; adding them here does not establish novelty or biological learning. Preserve the existing GRU innovation path when adding a residual memory readout so the first comparison does not also alter its correction signal.

The present analysis does not identify memory capacity as the cause of the failures. Any retained development gain still needs untouched-seed confirmation, an unseen scenario shift, autonomous quality-versus-total-compute evidence and a second environment before a paper-level architecture claim.

## Implemented comparison

The standalone [SPO+ loss](../src/openjev/research/otto_spo_plus_loss.py) is implemented and independently source-reviewed. All [27 fabricated tests](../tests/test_otto_spo_plus_loss.py) and Ruff passed, including hand-calculated values, gradients, exact ties, padding, illegal actions, fixed weights and an exhaustive 98,415-case formula/regret grid. [Qualification receipt](../output/otto-spo-plus-loss-v1/engineering-01/qualification-01/receipt.json).

It exposes per-row losses and a weighted scalar. The maximum uses the first maximizing action for its subgradient; the teacher reference is uniform over exact raw-float32 minima. The new [trainer](../scripts/train_otto_action_focused.py) integrates the loss with unchanged shared-output models. All **190 fabricated checks** and Ruff passed across collection, loss, training, metrics and the independent saved-output audit. [Qualification receipt](../output/otto-action-focused-v1/engineering-01/qualification-01/receipt.json).

The frozen comparison uses 54 fresh TRAIN paths, 36 VALID paths and three paired seeds for each of four cells. Its worker reports twelve completed fits and 8,640 updates, with every checkpoint closed before VALID. That receipt cannot replace the missing original supervisor exit record. The models and predictions are preserved without a training retry; numerical interpretation awaits the separate diagnostic.
