# Review: matched 80-versus-320-epoch training

**Prospective design review, not an empirical result.** This comparison tests whether the fixed optimizer budget is a material limitation before changing architecture. It uses the published [target-precision result](otto-target-precision-results.md) and existing source contracts only. No new checkpoint predictions, labels, fitting, simulation or empirical array reads were performed for this review.

The precision trial passed 7/33 conditions and no absolute competence condition. Its online training curves continued improving, but those changing-model minibatch losses cannot establish the final model's fit or action ordering. Reusing the completed R64 labels makes a fixed optimization comparison cheaper and more informative than immediately collecting more labels. Longer fitting may also memorize noisy or conflicting labels, so reduced TRAIN loss alone is insufficient.

## What must stay paired

Use all 558 anchors, the exact cached float32 features, eligibility masks, R64 centered targets and original episode weights. Preserve the inherited R16 global target scale for both arms. Do not recompute an R64 scale, clip targets, filter uncertain anchors, alter the loss, add data or reuse evaluation for selection.

Both arms start fresh from identical ordinary 2,836→32→16→4 Tanh parameters for each seed 30101-30103. Keep Adam at 0.0003, batch size 128, clipping 5 and eight D4 views. Reset the permutation generator to `seed + 20000` for each arm. Every epoch has four batches of 128 and one of 46 rows; weight the partial batch by its actual row count when reporting epoch loss, without renormalizing episode weights within batches.

The short arm performs 80 epochs/400 updates. The long arm performs 320 epochs/1,600 updates. Across three pairs this is **six fits, 1,200 fit-epochs and 6,000 updates**. This intentionally changes paid optimization work, not model capacity or labels.

Verify all first-80 epoch permutations and corresponding batch loss/gradient-norm witnesses exactly. Save the long arm's epoch-80 checkpoint and compare every tensor's dtype, shape and bytes to the short final. The long arm then continues the same Adam state; it must not restart from that checkpoint. A mismatch is a technical failure, not permission to restart a seed. The intermediate checkpoint is diagnostic only, never an additional evaluated policy or a selection candidate.

There are exactly **15 checkpoint artifacts**: six initial, six final and three long-arm epoch-80 checkpoints. Both members' initial checkpoints must be durably published before their pair updates. Save/export/parity work and any failed operation remain in the ledger. No warmup with TRAIN data is hidden in setup or omitted from costs.

## Fixed-final TRAIN diagnostics

Evaluate only the six final exported heads, after durable checkpoint verification. Use all 558 rows and all eight prescribed feature views. Five batches per view produce **240 counted NumPy forward calls and 26,784 evaluated view rows**, plus six explicitly timed diagnostic loads. Save raw float32 predictions with shape `[558,8,4]` per fit. Do not instantiate extra Torch models, score the epoch-80 prefix or ensemble the deployed action.

For each view, permute targets and masks identically, subtract each prediction's and target's eligible-action mean, then average squared errors over eligible actions. Average the eight views and apply the fixed row weights with denominator 558. Perform these reporting reductions in float64 from the saved float32 predictions and rounded training targets. Use stored float32 training weights upcast to float64. This is a fixed-final deployed-NumPy reconstruction of the mathematical training objective, not bitwise Torch training loss or the online epoch statistic.

Reuse identity-view predictions for single-view centered MSE and deployed actions. Select the first eligible numeric action whose Python-float score is strictly within `1e-10` of the eligible minimum, matching deployment. Report physical saved-label regret as the chosen action's R64 mean cost minus the eligible minimum. Also retain exact optimal-action-set membership and agreement with the first exact label argmin. Report these decision metrics with original float64 episode weights divided by 558, including ties and zero gaps. These measure agreement with finite-sample teacher-following labels, not true optimal regret, independent generalization or a new efficacy gate.

The existing first-16-TRAIN NumPy/Torch final-export parity remains six calls per backend. It is separate from full-cohort diagnostics and does not certify every near-tied action. Six later, separately charged deployment loads prevent diagnostic initialization from making evaluation setup appear free.

## Autonomous decision and costs

Evaluate only six fixed finals plus the analytic teacher on 72 fresh paired cases: 24 at each sensing length, seven arms and **504 full-horizon episodes**. Preserve the 2,188-move censoring horizon, initial-hit mixture, eight paired blocks, arm rotation and all three seeds. Proposed case ranges 1080001-1080024, 1090001-1090024 and 1100001-1100024 require the root's reservation before freeze.

Keep all 33 conditions: three analytic positive controls, eighteen long-arm absolute competence conditions and twelve family-relative conditions. A relative gain cannot compensate for failure of competence. Publish every seed and setting, even if lower training error coincides with poorer autonomous behavior.

The prospective worker allocation is 7,200 seconds, one numerical thread, 4 GiB RSS and 16 GiB output. Record input preparation, paired initialization, short and long updates, checkpoint/storage work, diagnostics, deployment loads and full evaluation separately. Historical label acquisition remains a disclosed inherited cost; it is not physically paid again. This comparison is equal in information and initialization, deliberately unequal in updates and compute.

If both full-cohort error and autonomous competence improve, insufficient fitting is supported for this recipe. If error and saved-label regret improve but control fails, more epochs have not repaired the gap between finite teacher-following labels, visited-state coverage and repeated learned decisions. If error barely improves, optimization settings, representation and conflicting labels remain unresolved. None of these outcomes establishes a novel, recurrent, connectome or biological learning mechanism. The actor already receives the exact public Bayesian state; recurrence has no demonstrated inference advantage here.
