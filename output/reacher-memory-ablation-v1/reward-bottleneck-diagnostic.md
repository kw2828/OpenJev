# Prior memory study: is reward prediction a control bottleneck?

**Answer: it is a plausible next diagnostic target, not an established causal bottleneck.** Persistent recurrence substantially improves angle prediction, but its reward-prediction advantage is smaller and its control advantage over the packet MLP is small and inconsistent across the three fits. Most remaining selected-action reward error lies in the learned residual after the known actuator-cost correction. Saved one-step predictions cannot establish whether that error misranks the candidate action sequences that matter for control.

This is a post hoc analysis of the **completed and published memory study only**. It changes no original outcome, gate, model, training data or protocol. No current cache-study outputs were read. The calculation makes zero model calls, native-environment calls, rollouts or random draws.

## Authenticated scope and reproduction

- Prior plan SHA-256: `05f09ee5425190f5d652aedb10af1641037035d85e906d04d86baf33552a5786`.
- Completed independent audit receipt SHA-256: `2589383326a2312d7f738821da7fd11f82b9bee1b7168aba2e7f44aaa4946d5d`.
- Completed execution receipt SHA-256: `d63866e590799b46e88461197aa074b641f2a35c795bbf30e7cb04cba4db9ee1`.
- All 12 fitted models, all three panels, all 36 learned-control rows, 64 cases per row and 50 steps: 115,200 selected-action transitions. Separately, all 12 saved predictions on the same 96 held-out exploration episodes: 57,600 prediction evaluations of 4,800 shared transitions.
- All 98 numerical/provenance/source inputs used by the script are checked against the authenticated audit, its execution-member manifest, or its pinned runtime identity before use. No fitting checkpoints are opened. Detailed results retain every fit, every panel and observation-mask group.

From the repository root, reproduce with `.venv-robotics/bin/python output/reacher-memory-ablation-v1/reward-bottleneck-diagnostic.py`. The script imports NumPy and the standard library only. It checks reward decomposition, native-distance identity, saved-audit control costs, common-cohort reward MSE, and a closed-form moment sanity case. The original full independent audit is not rerun.

## First establish the reward timing

The authenticated `Reacher-v5.step` advances physics before obtaining observation and reward. The native reward is the negative fingertip-target distance plus the negative applied-action squared norm, with both weights one. The public wrapper instead supplies sine/cosine angles from final `qpos`, a target, validity and elapsed observation age. It adds clipped Gaussian actuator noise before native stepping. Rewards, realized noise and native geometry are audit labels, not controller inputs.

The saved native `raw_obs` final two coordinates reproduce `reward_dist` to numerical precision at the same post-step index. However, analytical forward kinematics from **true final qpos** does not reproduce that cached native geometry exactly. Across the 36 control rows, the maximum distance discrepancy is 2.743 mm and the largest per-row RMS is 0.106 mm. On the common exploration cohort, maximum/RMS distance discrepancy is 11.575/0.190 mm. The pinned XML uses RK4, and native reward reads cached body positions; this is not evidence of a one-decision label shift.

Consequently, converting decoded angle predictions to analytical distance is **invalid as an exact native-reward replacement**. That comparison was not computed. It could be a declared approximation in a later experiment, after accounting for this geometry/timing difference. These submillimetre RMS discrepancies are much smaller than the learned residual's approximately 5 cm control RMSE, so the cached-field discrepancy does not itself explain the larger model error.

## All-family results

Angle MSE is the mean squared error of four sine/cosine features against saved native angles, including missing endpoints as audit-only labels. It is not radians squared. Reward MSE uses the selected issued action and its subsequently observed reward. The control rows use each model's own trajectories, so they are not identical-input prediction comparisons.

| Panel | Family | Episode cost | Angle-feature MSE | Reward MSE | Residual-distance MSE |
|---|---|---:|---:|---:|---:|
| full | Persistent GRU | 8.085204 | 0.0005834 | 0.0026677 | 0.0025994 |
| full | Current-packet GRU | 8.726558 | 0.0019380 | 0.0040406 | 0.0039509 |
| full | Bounded-history GRU | 8.572863 | 0.0003875 | 0.0030216 | 0.0029656 |
| full | Packet MLP | 8.058288 | 0.0012028 | 0.0028333 | 0.0027778 |
| ordinary | Persistent GRU | 8.094290 | 0.0027336 | 0.0028284 | 0.0027648 |
| ordinary | Current-packet GRU | 8.920803 | 0.1004690 | 0.0042757 | 0.0042075 |
| ordinary | Bounded-history GRU | 8.692720 | 0.0550008 | 0.0035564 | 0.0035169 |
| ordinary | Packet MLP | 8.180700 | 0.0983606 | 0.0035899 | 0.0035364 |
| shift | Persistent GRU | 8.228193 | 0.0083674 | 0.0030793 | 0.0030212 |
| shift | Current-packet GRU | 9.041908 | 0.1693213 | 0.0047415 | 0.0046832 |
| shift | Bounded-history GRU | 8.800974 | 0.1066135 | 0.0039330 | 0.0038999 |
| shift | Packet MLP | 8.372592 | 0.1569461 | 0.0040281 | 0.0039766 |

Against the packet MLP, persistent recurrence reduces angle MSE by **97.22% ordinary / 94.67% shifted**, reward MSE by **21.21% / 23.55%**, and total episode cost by only **1.06% / 1.72%**. Under full sensing it reduces angle MSE by 51.50% and reward MSE by 5.84%, but has 0.33% higher episode cost. The bounded-history model's best full-sensing angle MSE also does not yield the best control.

The same distinction appears without model-dependent trajectories. On the common 96-episode public-history cohort:

| Family | Angle-feature MSE | Reward MSE | Residual-distance MSE |
|---|---:|---:|---:|
| Persistent GRU | 0.0115992 | 0.0038678 | 0.0022753 |
| Current-packet GRU | 0.1126954 | 0.0054086 | 0.0037974 |
| Bounded-history GRU | 0.0729470 | 0.0047029 | 0.0031209 |
| Packet MLP | 0.1184374 | 0.0056798 | 0.0040883 |

Persistent recurrence improves common-input angle MSE by 90.21%, reward MSE by 31.90% and residual-distance MSE by 44.35% versus the MLP. All three paired fits improve reward MSE on these identical histories. Their differences are -0.001597, -0.001749 and -0.002090. This is evidence of predictive benefit, not evidence of the resulting policy's action-ranking quality. Exploration commands also have much higher actuator costs than controller-selected commands, so this cohort's prediction MSE is not a deployment-error estimate.

## What contributes to reward error?

All four arms already subtract the analytic expected cost of clipped Gaussian actuation from their learned reward output. Write `R = -D - C`, where `D` is native distance and `C` is realized squared actuator cost. Recover the negated learned residual as `D_hat = -R_hat - E[C | issued_command]`. Then:

`reward_error = -(D_hat - D) + (C - E[C])`.

The script evaluates the exact empirical MSE identity, including the cross term. It computes the expected action-cost mean and variance independently from truncated-normal moments, without using realized disturbance as a prediction input.

For persistent recurrence on its ordinary/shift trajectories, residual-distance discrepancy contributes **97.75% / 98.12%** of total reward MSE; realized actuator-cost innovation contributes **2.27% / 2.06%**, with a small cross term closing the identity. These are a descriptive decomposition, not independent variance components. Distance also changes with actuator noise, so action-cost variance alone is not the total reward's irreducible noise floor. The learned residual was trained through total reward, not directly supervised as a pure distance predictor.

Persistent reward predictions are optimistic on these selected actions: mean predicted-minus-realized reward is +0.02843 ordinary and +0.02973 shifted, versus +0.01084 and +0.01131 for the MLP. These biases may matter, but scalar calibration bias alone does not determine which actions CEM ranks highest. An action-independent offset could leave rankings unchanged.

Missing observations amplify the MLP's reward error. At missing **decision roots**, ordinary/shift reward MSE is 0.006935/0.006490 for the MLP and 0.002663/0.003094 for persistent recurrence. Memory helps prediction precisely where information is absent, but this alone does not establish useful velocity inference or the amount of attainable utility gain.

The realized utility difference also includes an action-effort tradeoff. Persistent recurrence reduces distance cost relative to the MLP by 0.16717 ordinary and 0.23250 shifted, while increasing actuator cost by 0.08076 and 0.08810. Thus improved proximity is partly offset by greater actuation. Under full sensing, its 0.05269 lower distance cost is more than offset by 0.07960 extra actuator cost. These are outcomes, not a claim that changing the reward penalty would improve the existing task.

The GRU search scores the learned reward head; its angle decoder is not converted directly into native distance. Better angular reconstruction is therefore not guaranteed to produce proportional score improvement. The MLP does feed predicted packets into its imagined continuation, so decoder quality can affect its multistep scores differently. Neither observation establishes a new architectural mechanism.

## Paired uncertainty and all fits

These intervals are **post hoc normal approximations over case clusters after averaging the three fixed fits**. They retain common-case pairing and avoid treating the three model evaluations of a case as independent. They do not estimate uncertainty over new training fits, are not multiplicity-corrected, and do not replace the study's original gate or intervals. Negative differences favour persistence.

| Panel | Reward-MSE difference vs MLP | Approximate conditional 95% interval | Episode-cost difference | Approximate conditional 95% interval |
|---|---:|---:|---:|---:|
| full | -0.0001656 | [-0.0004932, 0.0001621] | 0.026916 | [-0.275107, 0.328938] |
| ordinary | -0.0007615 | [-0.0013285, -0.0001945] | -0.086410 | [-0.419897, 0.247077] |
| shift | -0.0009488 | [-0.0016764, -0.0002212] | -0.144399 | [-0.510351, 0.221552] |

Ordinary reward error improves in two of three paired fits; shifted reward error improves in all three. Control cost improves in only two of three on both gaps, and its conditional intervals include zero. Thus a stronger predictive difference coexists with unresolved small utility differences.

Every fitted model is retained below. Reward columns are selected-action one-step MSE. The complete JSON additionally contains all mask-specific errors, biases, episode distance/action splits, matched differences and alignment canaries for every row.

| Fit | Full cost | Ordinary cost | Shift cost | Full reward MSE | Ordinary reward MSE | Shift reward MSE |
|---|---:|---:|---:|---:|---:|---:|
| residual_gru-pair0 | 7.532381 | 7.577299 | 7.654219 | 0.0035632 | 0.0038525 | 0.0038874 |
| residual_gru-pair1 | 8.755736 | 8.786396 | 9.038461 | 0.0019627 | 0.0020196 | 0.0025096 |
| residual_gru-pair2 | 7.967494 | 7.919175 | 7.991898 | 0.0024773 | 0.0026131 | 0.0028409 |
| current_gru-pair0 | 9.193466 | 9.546609 | 9.760126 | 0.0047135 | 0.0051432 | 0.0059485 |
| current_gru-pair1 | 8.962583 | 9.131272 | 9.199878 | 0.0032462 | 0.0035884 | 0.0041239 |
| current_gru-pair2 | 8.023626 | 8.084528 | 8.165719 | 0.0041621 | 0.0040957 | 0.0041522 |
| bounded_gru-pair0 | 8.284260 | 8.530864 | 8.703753 | 0.0037870 | 0.0040670 | 0.0041638 |
| bounded_gru-pair1 | 9.440288 | 9.485365 | 9.516139 | 0.0021596 | 0.0031100 | 0.0038265 |
| bounded_gru-pair2 | 7.994042 | 8.061930 | 8.183031 | 0.0031183 | 0.0034921 | 0.0038088 |
| packet_mlp-pair0 | 8.165755 | 8.112797 | 8.239293 | 0.0032650 | 0.0037668 | 0.0040539 |
| packet_mlp-pair1 | 8.075799 | 8.271360 | 8.514799 | 0.0029607 | 0.0035612 | 0.0038858 |
| packet_mlp-pair2 | 7.933310 | 8.157944 | 8.363685 | 0.0022741 | 0.0034416 | 0.0041445 |

## What this supports next

The strongest supported statement is that useful memory improves reward prediction as well as angles, yet angular MSE substantially overstates the observed gain in native control. The residual reward model deserves a targeted test. This does **not** prove that a more accurate reward model would improve CEM decisions: only selected actions were executed, their trajectories differ, and saved one-step errors do not measure unchosen branches or long-horizon reward ranking. Prediction error could also reflect planning-induced distribution shift, nonessential angular detail, common model bias, or attainable-control limits.

After the cache result is terminal and audited, a separate prospective experiment could hold the transition model, dataset, CEM proposal budget and reward penalty fixed while comparing its original reward head with a carefully timed distance-based or residual-correction head. Native matched-root branch scoring and selected-action utility would need separate evaluation, and the approximate FK timing contract must be explicit. Any training or planner intervention requires a new frozen comparison; none was performed here. This evidence does not justify a connectome, plasticity or uncertainty-gating claim.

JSON SHA-256: `27e258fa5e3fef5fb34c624f0f8591b9a720850f52b4c9d7e97cdbee731b92ef`. Diagnostic script SHA-256: `92447b45e51fbd64e759af87666d26c8ef39840c1c21c4a2a172a0845f93d4ec`.
