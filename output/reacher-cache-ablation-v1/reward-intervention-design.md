# Frozen-transition reward intervention

Prospective alternative, unfrozen and unrun. Based only on the completed [prior reward diagnostic](../reacher-memory-ablation-v1/reward-bottleneck-diagnostic.md), authenticated to memory-study audit `2589383326a2312d7f738821da7fd11f82b9bee1b7168aba2e7f44aaa4946d5d`. No current cache results were read, and no implementation, model, native-environment or training calls were made. The current cache outcome determines whether to pursue this or the explicit-motion baseline.

**Yes: replace the scoring function while freezing the learned transition and decoder weights.** This is a reward-interface intervention requiring zero new fits. It tests whether available decoded state is more useful for action selection than the existing reward head; it does not establish a new world-model architecture.

## Intervention and controls

Predeclare all three persistent-GRU fits and all three packet-MLP fits from the completed memory study. The GRU is primary; the MLP checks whether the intervention depends on accurate decoded angles. Keep every selected family's fit, without choosing the best seed.

Compare two scores:

- **Original:** the saved model's learned residual reward minus its existing analytic expected clipped-actuator cost.
- **Geometry:** convert predicted next sine/cosine pairs to angles using `atan2`, compute planar fingertip position with the pinned 0.10 m and 0.11 m link offsets, and use negative distance to the public target minus the **same** expected actuator cost.

Retain original reward clipping, horizon, action blocks, CEM256 settings, root assimilation and selected-action advancement. Initially call the unchanged model `advance` in both arms and override only its returned score. This intentionally retains the unused reward-head work and makes transition identity directly checkable. Charge the additional geometry operations; claim no speedup.

Do not silently use the four decoded values as unit-circle coordinates: their norms need not equal one. Declare the `atan2` projection, monitor both pair norms, and treat a norm below a fixed numerical threshold such as `1e-6` as a recorded invalid-prediction failure, not an invisible learned-head fallback. Test this contract before scoring. Record predicted joint-limit violations without adding an undeclared penalty.

A third, fixed **50:50 hybrid of the two residual-distance estimates**, with the actuator term subtracted once, is optional. Declare it before evaluation if both sources of error deserve investigation. Do not tune its weight on evaluation roots, add it after seeing a failure, or call fallback geometry the pure intervention. No reward-head refitting is necessary for this first test.

## Native timing and uncertainty

Native reward is post-step, but it reads cached body positions, while the decoder targets final-qpos angle features. Under the pinned RK4 environment, even true final-qpos forward kinematics differs from native distance: control maximum 2.743 mm and largest row RMS 0.106 mm; common exploration maximum 11.575 mm and RMS 0.190 mm. Geometry is therefore an explicitly approximate reward, not ground truth. Do not shift labels by one decision to conceal this discrepancy.

Those RMS differences are much smaller than the prior learned residual's roughly 5 cm control RMSE, but they do not guarantee useful rankings. Distance of a projected mean angle is generally not expected distance under actuator uncertainty. The known expected actuator penalty corrects only that penalty, not nonlinear distance uncertainty.

## Common-root ranking first

Use a small declared diagnostic set, for example the first eight prior cases at roots 12 and 32 on each panel: 48 shared roots, labeled exposed development data. Reconstruct each checkpoint's causal public-history belief from the same root history. No privileged state enters model scoring.

Score one identical saved 64-sequence proposal bank under both scores. Separately run each scorer's CEM256 with paired initial banks and innovations, then append its selected full sequence to a deduplicated native-test union. Later adaptive CEM proposals need not match. With six checkpoints and two scores, the union contains at most 76 sequences per root.

An offline evaluator restores the saved native root and tests that union with four explicitly named, common actuator-noise realizations, full horizon 12 and correct terminal truncation. These native labels never reach CEM. This example costs at most 175,104 native transitions, excluding restoration checks; charge every model, search and native call separately.

Report rank correlation, top-choice native return, and regret relative to the **tested union**, not an unattainable global optimum. Keep per-root noise variability and all fits. Better one-step MSE alone cannot qualify the intervention. These exposed-root diagnostics cannot confirm generalization.

## Paired closed-loop confirmation

After fixing the intervention contract, run both scoring arms on fresh paired resets, disturbances and schedules: all three panels, 64 cases, 50 steps, six frozen checkpoints, giving 36 control rows and no optimizer updates. Preserve model hashes before/after and verify identical predicted states for identical root/action prefixes. Closed-loop states will diverge when actions differ.

Freeze a utility margin, all-fit consistency, full-sensing allowance, invalid-output rule and wall-time cap before drawing these cases. Report native distance and actuator-cost components as well as total cost, conditional paired intervals and full deployment cost. Include failures. Do not select geometry or the hybrid retrospectively.

CEM may exploit inaccurate decoded poses, projection near zero norms, unreachable configurations or long-horizon drift. Better geometry rankings with worse closed-loop utility would reveal an unresolved planning/distribution problem. Continue only for repeatable native utility gains on fresh cases; otherwise preserve the failed intervention and reconsider the diagnosed bottleneck.
