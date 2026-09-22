# After the fixed training-budget comparison

**Proposed follow-up, not frozen or admitted.** The empirical worker completed once and the independent saved-record audit completed with agreement. No proposed diagnostic, new fitting, label collection or simulation has been executed for this note.

## What this comparison says

The [80-versus-320-epoch comparison](otto-training-budget-protocol.md) reports **10/33 conditions passed**: 3/3 analytic positive controls, 0/18 absolute competence conditions and 7/12 relative conditions. Equal-seed family success improves in all three settings, but remains far below the analytic controller:

| Sensing length | 80 epochs | 320 epochs | Analytic control |
|---|---:|---:|---:|
| 3 | 10.19% | 15.34% | 100% |
| 4 | 20.42% | 21.57% | 100% |
| 5 | 4.52% | 9.52% | 100% |

These are initial-hit-mixture-weighted family means over all three fit seeds. The seven arms share 72 paired cases; 504 episodes are not 504 independent cases. The worker also reports an 11.38% reduction in aggregate final TRAIN MSE and a 14.78% reduction in the gap to the best sampled R64 label. Neither is held-out accuracy or true environment regret.

The intervention changes optimization work while holding labels, scale, features, initialization and the first 80 epochs fixed. Longer fitting helps these measured outcomes but does not establish a competent controller. It does not identify whether the remaining error comes from conflicting labels, the function class, optimization, limited visited-state coverage or the teacher-following objective. Another automatic increase in epochs or rollout replicates is not justified by this result.

Evidence: [worker receipt](../output/otto-training-budget-v1/run-01/receipt.json), SHA `47004eb796ac7ab9a836f88c2942dd6b6f00e008d277b205afbdf85e3a8eb182`; [original supervisor terminal](../output/otto-training-budget-v1/supervision-01.terminal.json), SHA `d99dbbd53bb3ca281b15db82f577f5e8ea6e62ddecb210a40b86e73004b2d276`. Original process wall time was 1,123.37858 seconds. The [agreeing audit receipt](../output/otto-training-budget-v1/audit-01/receipt.json), SHA `c3ddca8593fb5ffe9f7b01b69728f22f64d80217deb1f38472417fb0c8422b84`, records 27,768,190 checks. Its [successful original supervisor terminal](../output/otto-training-budget-v1/audit-supervision-01.terminal.json), SHA `1bea061a7463349c0787896300d569453b09eb1fdbd4a1fb96102beab4f68bb9`, covers 36.259107459 seconds. The audit checks saved records and diagnostic arithmetic; it does not regenerate neural scores, gradients or native trajectories.

## Next bounded diagnostic: conflicting targets at identical inputs

The previous scalar-return duplicate floor does not measure this four-action R64 objective. Propose one saved-data calculation on **all 558 TRAIN rows and all eight D4 views**, preserving the original feature bytes, masks, rounded targets and weights. No head evaluation, optimizer, source sampling or simulator is needed.

For view `j=(i,v)`, construct the qualified transformed float32 feature vector. Transform the four target entries and mask using the same action permutation as training. Upcast the saved float32 scaled target and weight to float64. Center the target over that view's eligible actions, exactly as the fixed-final diagnostic specifies. Call that eligible vector `y_j`, its action count `k_j`, and its weight `w_i`.

Group by **complete transformed feature bytes and transformed mask bytes**, verifying actual byte equality within any hashed group. Do not group approximate neighbors, discard duplicates, merge signed-zero encodings or canonicalize distinct D4 views into one input. Keep all 4,464 view occurrences, including repeated views of symmetric inputs.

Within group `g`, the mask and eligible count `k_g` agree. Define:

```
W_g = sum(j in g, w_i)
mu_g = sum(j in g, w_i * y_j) / W_g
B = sum(g, sum(j in g, w_i * sum_a((y_ja - mu_ga)^2) / k_g)) / (558 * 8)
```

The action sum includes eligible actions only. Use direct weighted deviations rather than subtracting two large second moments. **Do not normalize by the rounded sum of weights**, give each group equal weight, or divide by a global action count. Singleton groups contribute zero. Preserve all per-view memberships, group counts, eligible counts, total weights and contributions so the aggregate can be checked independently.

`B` is a relaxed empirical loss floor: a deterministic head must give one raw score vector at an identical input. Group-specific free predictions relax the finite MLP parameterization; splitting different masks also relaxes their shared raw-output constraint. Distinct transformed inputs need not satisfy equivariance because this ordinary head uses augmentation, not enforced equivariance. The unrestricted weighted group mean is conservative even if float64 centering leaves a tiny nonzero mean residual. Report numerical roundoff explicitly; do not claim a machine-exact bound on Torch's float32 loss.

Compare the same common floor against each of the six saved final eight-view MSEs. Report `L`, `B` and untruncated `L-B`, with arithmetic discrepancies visible. A large floor identifies incompatibility of these saved labels at these encoded inputs. A small floor does not rule out sampling noise or prove that this MLP can fit the remaining error. This calculation cannot estimate population noise or demonstrate control efficacy.

Before execution, freeze source/input pins, reductions and complete outputs. A prospective allocation is 120 seconds, one numerical thread, 1 GiB RSS and 64 MiB output, subject to a source-level size check. This note does not authorize that run or introduce a pass threshold.

## How it would inform the next experiment

If conflicting identical-input labels account for much of the residual objective, first investigate their provenance and uncertainty without replacing labels or filtering anchors. If the floor is small, separate fitting capacity from action quality before choosing a new architecture: the existing final TRAIN errors alone cannot distinguish representation from optimization, and a good fit would still need fresh autonomous competence. Any subsequent intervention should change one of these factors with matched controls, rather than automatically increasing both data and compute.

Recurrence or a learned world model would need a distinct, falsifiable role. The actor already receives the exact public Bayesian belief and supplied observation kernel. This result supplies no evidence of a missing hidden-state inference mechanism, a biological learning advantage or a novel architecture.
