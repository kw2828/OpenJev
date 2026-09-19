# Reacher memory: a conditional next experiment

Primary-source review, 2026-09-18. This memo is prospective, unfrozen and unrun. It does not use running study outputs or propose changing the current study. The current protocol SHA supplied by the coordinating task is `05f09ee5425190f5d652aedb10af1641037035d85e906d04d86baf33552a5786`, with execution/audit caps of 3600/300 seconds. No scientific data draws, model calls, fitting or source edits were performed for this review.

The next question should be whether useful memory is simply a retained position measurement or actually an estimate of hidden motion. If persistent memory fails its current qualification, stop this branch and report that result. If it qualifies, first challenge it with explicit public-history baselines. A structured recurrent observer becomes worthwhile only after that comparison.

## Six directly relevant papers

| Primary source | Established mechanism and relevance | Boundary for OpenJev |
| --- | --- | --- |
| [Shaj et al., Action-Conditional Recurrent Kalman Networks, CoRL 2020 proceedings (2021), arXiv:2010.10201](https://proceedings.mlr.press/v155/shaj21a/shaj21a.pdf) | The closest precedent. Adds an action term to a latent Kalman-style transition; robot forward models consume joint-angle histories and actions. The Panda experiment removes three quarters of training observations. Figure 4 also includes a previous-state predictor. | Section 4.1 explicitly treats the action as known, leaving covariance propagation unchanged by action conditioning. Our issued command is known, but the applied actuator command is noisy. This is a motivation for an ablation, not evidence of novelty. Its forward-model results are not our closed-loop control result. |
| [Becker et al., Recurrent Kalman Networks, ICML 2019, arXiv:1905.07357](https://proceedings.mlr.press/v97/becker19a/becker19a.pdf) | Learned locally linear dynamics and a factorized latent belief make recurrent mean/covariance updates efficient. Relevant to uncertain hidden-state estimation and missing observations. | A latent covariance is not automatically calibrated physical velocity uncertainty. Preserve mean-only and conventional-filter baselines. [Author implementation](https://github.com/LCAS/RKN). |
| [Kloss et al., How to Train Your Differentiable Filter, arXiv:2012.14313](https://arxiv.org/pdf/2012.14313) | Compares differentiable EKF, UKF, Monte Carlo UKF and particle filtering, including position observations with hidden velocity and process noise. Examines learned constant, heteroscedastic and correlated uncertainty. | Better tracking can coexist with an inaccurate learned noise model. Evaluate predictive calibration and state error separately; do not infer uncertainty identification from reward or MSE alone. This supports a strong conventional filter comparator. |
| [Revach et al., KalmanNet, IEEE TSP 2022, arXiv:2107.10043](https://arxiv.org/html/2107.10043v3) | A recurrent network learns the correction gain inside a partially known state-space filtering procedure. Relevant when dynamics/noise assumptions are imperfect. | The original method trains on labeled true states, including state-estimation loss. Copying that supervision would cross our public-only training boundary. An observation-prediction adaptation must be named as such and tested independently. [Author implementation](https://github.com/KalmanNet/KalmanNet_TSP). |
| [Hafner et al., Learning Latent Dynamics for Planning from Pixels, ICML 2019, arXiv:1811.04551](https://arxiv.org/abs/1811.04551) | PlaNet combines deterministic and stochastic recurrent state, action-conditioned latent prediction, reward modeling, CEM planning and latent overshooting. | A learned recurrent state plus CEM is established. Prediction quality and uncertainty must earn their value through actual control. The original agent collects data through interaction; our fixed-corpus pilot does not reproduce that learning loop. |
| [Ni et al., Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs, ICML 2022, arXiv:2110.05038](https://proceedings.mlr.press/v162/ni22a/ni22a.pdf) | Carefully implemented recurrent actor-critic methods are strong POMDP baselines. Architecture, sequence length and optimization choices can explain apparent mechanism gains. | Relevant if we later make an RL comparison. A recurrent policy trained through new interaction is a separate budget and supervision setting; it is not a free replacement for an offline world-model arm. |

## Strongest simple challenge: remember measurements explicitly

This is an engineering proposal inferred from the state-estimation problem, not a claimed method from one paper.

The first learned comparator should be a small **two-valid-measurement history model**, with a position-only ablation. Store the last two genuinely observed angle packets and their timestamps, current target, current validity/age, and issued-action context. Two valid packets can remain available across a blackout; a window of three consecutive packets cannot. Fit this comparator on exactly the same public corpus, with its own declared initialization and the same training-selection budget.

Use these nested feature choices:

1. **Last valid angles only.** Retain sine/cosine of the last measured angles and their age. Do not label cached angles as a current valid observation. A missing public packet stays missing.
2. **Last valid angles plus measured interval motion.** Add `wrap(theta_last - theta_previous) / elapsed_time`. Before the second valid measurement use an explicit unavailable flag, not fabricated velocity. This is interval-average angular velocity, not instantaneous velocity. Use actual elapsed time, especially across missing packets.

Both should retain the same action-conditioned rollout interface and comparable decoder/reward capacity. Train each under its actual history semantics. Do not feed cached features into an already trained current-only network and call the resulting distribution shift a fair baseline. Keep optimizer-update matching distinct from measured training/deployment compute.

The strongest simple analytic competence comparator is an **angle/velocity EKF or UKF with supplied nominal dynamics, public-angle updates, and actuator-noise propagation**. It should sit beside the existing public kinematic observer. A position-only observer using the same supplied physics and planner is the clean companion ablation. Label both supplied-physics references; their information and modeling assumptions differ from the learned models.

The current 25-check gate does not include either learned explicit-cache model. Passing it can therefore show useful persistent information without identifying velocity inference. Nor is the current public kinematic reference a hold-last-angle ablation: it propagates a nominal simulator and re-estimates interval velocity.

## One recurrent mechanism worth a later test

My proposed test is a **small recurrent position/velocity belief model with observation correction and action-noise propagation**. It is a structured filtering adaptation, not a new architecture claim.

Maintain an explicit two-joint position/velocity mean and a small positive-semidefinite covariance. An action-conditioned transition predicts the next mean. On a valid measurement, use the wrapped angle innovation to correct the estimate; on a missing packet, perform prediction only. The known observation map to sine/cosine anchors angular meaning. A kinematic relation between position and velocity gives velocity a physical interpretation, while the acceleration/transition component can be fitted from public observation sequences.

The narrow ablation is whether propagating uncertainty in the *applied* command adds value beyond a constant learned process-noise term. With `a = clip(clip(u) + epsilon)`, use the mean and covariance of the clipped applied action, including saturation effects. Do not assume its mean is always the issued command. A first-order covariance proposal is:

`P_next = A P A^T + B Cov[a | u] B^T + Q_model`,

where A and B are state/action sensitivities of the declared transition. This is standard uncertainty propagation. A conventional EKF must receive the same control-noise treatment; otherwise the comparison would handicap the obvious analytic baseline. Account for Jacobians, covariance operations, observation corrections and any extra prediction samples.

The experiment should test a specific explanation: under unobserved actuator noise, does the structured belief preserve useful motion information through gaps and improve decisions? It should not be presented as evidence that a recurrent latent state is a world model merely because it stores past observations. Nor does a noise covariance become epistemic uncertainty or a conformal guarantee.

For public-only training, supervise future observed angles and total executed rewards; retain native velocity and realized actuator noise as audit-only labels. If a likelihood term is added, predeclare it and compare against a version without that term. This would be an adaptation of filtering ideas, not a faithful KalmanNet reproduction. Do not add online learning, an RL policy, a JEPA objective and a new planner at the same time.

## A prospective comparison that could answer the question

Freeze a new protocol only after the current gate and independent audit are complete. Retain all paired fits and compare persistent GRU, last-valid-angle model, two-valid-measurement motion model, conventional filter, and the proposed structured model. Include constant-noise and action-noise-aware versions of the same structured model. This keeps the mechanism comparison within one architecture while the simpler models test whether it is needed at all.

Keep public training data, candidate search, native cost, fit-selection allowance and evaluation cases paired. Report actual training cost and total deployment work, not only optimizer steps or CEM candidate count. Covariance-aware planning through extra samples is a separate compute factor; hold the planner fixed for the first estimator comparison or give all comparators the same sample budget.

The critical identification diagnostic should use independently generated, untouched cases with similar latest visible positions but distinct preceding motion/action histories. Measure causal blackout angle error and explicit velocity-estimate error against native audit labels, stratified by missing duration. No true velocity enters fitting or decisions. Shortest-angle differences can alias rotations; report those cases rather than treating finite differences as truth. Even with known dynamics, unobserved future actuator noise leaves irreducible uncertainty, so perfect per-trajectory state recovery is not an appropriate target.

Before evaluating, specify the utility margin, all-fit consistency rule, full-sensing degradation limit, runtime budget, noise/gap shift and calibration metric. Continue only if the proposed model beats each predeclared simple learned comparator on native utility and useful motion estimation, remains competitive with the conventional filter under its stated information advantage, and improves the measured utility-versus-compute tradeoff on fresh shifted cases. A better likelihood with worse control is not enough. A GRU gain explained by the two-valid-measurement baseline is an informative negative mechanism result.

Only after that result should a second task introduce genuinely different hidden dynamics, such as actuator lag or friction, with its own fixed evaluation. The cited robotics work already demonstrates recurrent filtering on complex actuators; repeating the general idea on Reacher is not a novelty claim.
