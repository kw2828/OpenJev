# Saved-label conflicts at identical encoded inputs

22 September 2026. Prospective saved-data diagnostic after the completed [80-versus-320-epoch comparison](otto-training-budget-results.md). This calculation changes no checkpoint, target, evaluation result or continuation condition. Its purpose is to distinguish an exact-input label-conflict floor from the residual error of the six existing fits before allocating another model experiment.

## Inputs and complete scope

Bind the completed training-budget plan `a204495fb0fd66ea3e7ea7c5052315f85bf3fa75438c243cf6a6f17e92e4099a`, worker receipt `47004eb796ac7ab9a836f88c2942dd6b6f00e008d277b205afbdf85e3a8eb182`, original parent `d99dbbd53bb3ca281b15db82f577f5e8ea6e62ddecb210a40b86e73004b2d276`, agreeing audit `c3ddca8593fb5ffe9f7b01b69728f22f64d80217deb1f38472417fb0c8422b84` and audit parent `1bea061a7463349c0787896300d569453b09eb1fdbd4a1fb96102beab4f68bb9`. Bind the exact required cache, six saved final diagnostic arrays, diagnostic summary and source/runtime records through their authenticated descriptors. The complete earlier source/input closure is inherited from that agreeing audit; do not rerun teacher panels or historical evaluation.

Use all **558 TRAIN rows, 144 source episodes and eight D4 views**, including every duplicate, repeated symmetric view, mask, tie and signed-zero encoding. Use the existing float32 features, rounded scaled float32 R64 targets, float32 training weights and eligibility masks. No rescaling, row selection, deduplication for training, label replacement or statistical noise estimate. This is **4,464 view occurrences** and zero new neural predictions, optimizer updates, teacher samples or native episodes.

## Fixed calculation

Apply the exact qualified feature and action transformations in `otto_symmetry_head.py`. A view consists of the transformed float32 feature vector, transformed eligible-action mask and transformed rounded target. Upcast targets and the saved float32 row weight to float64. Center each target over eligible actions using the same mathematical convention as the fixed-final training diagnostics. Every view inherits its row's weight.

Group occurrences by complete transformed feature bytes **and** complete transformed mask bytes. Verify actual bytes within any hashed bucket. Keep signed zeros distinct and do not merge approximate neighbors or different transformed inputs. This ordinary dense head uses augmentation, not enforced equivariance. Splitting the same feature vector across different masks relaxes the shared-output constraint and therefore gives a conservative floor.

For each group, calculate the weighted eligible-target mean. Sum the direct weighted squared deviations from that mean, divided by the group's eligible-action count, then divide the total by **558 × 8**. Do not divide by the rounded weight sum or give groups equal weight. Singletons contribute zero. Preserve every membership, group size, eligible count, summed weight, mean and contribution. Use direct deviations rather than subtracting large second moments.

Independently reduce each of the six saved final `[558,8,4]` float32 score arrays into the same eight-view float64 MSE with the original float32-upcast training weights. Compare the reconstruction with the saved audited measurement. Report each model's `L`, common floor `B` and **untruncated `L-B`**. Also report the maximum within-group centered-score spread from those saved outputs: batching and floating-point arithmetic can make a mathematical same-input assumption only approximate in recorded computation. This adds no model call.

The unrestricted weighted mean relaxes both model capacity and floating-point output restrictions. Report arithmetic discrepancies and declared tolerance. Do not claim a machine-exact lower bound on Torch's float32 loss, population noise, held-out action quality or autonomous control.

## Execution and verification

Freeze implementation, protocol, qualified transform, runtime, evidence descriptors and fabricated-test receipts before decoding empirical arrays. One original supervised execution has **120 seconds, one numerical thread, 1 GiB RSS and 64 MiB output**. Full transformed features require 50,639,616 raw bytes; masks, targets, weights, group IDs and compact group records must fit within the remaining space. Verify output and memory bounds during execution. Preserve failed attempts and complete original supervisor status; no implicit retry or budget extension.

The meaningful fabricated checks must cover weighted conflicts, eligible-action normalization and masked entries, unequal group sizes, all-view retention, signed-zero distinctions and consistent transformed targets/masks. An independent saved-data checker verifies all memberships, transformed bytes, group arithmetic, saved-score reductions and original process closure without importing a model, optimizer or simulator. It receives a separately frozen source/input plan and bounded allocation.

## Interpretation and next action

There is no pass threshold or checkpoint promotion in this diagnostic. A substantial floor identifies incompatible saved labels at exactly identical encoded inputs under this relaxed objective. A small floor cannot exclude finite-sample label noise, limited support, optimization or representation problems. Residual excess does not identify its cause. The result guides a separate, prospectively controlled model intervention; it cannot establish the user's requested novel-architecture advantage by itself.
