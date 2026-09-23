# Fresh comparison of decision-focused recurrent training

This prospective development screen follows the [saved ranking diagnosis](otto-ranking-diagnosis-results.md). That diagnosis found that lower legal-centered and boundary-pair squared error can coexist with worse choices. It did not identify a memory-capacity failure. All previous studies remain closed; no exposed VALID trajectories or selected fitted checkpoint are reused here.

## Fixed comparison

Train four shared-output cells at three paired fit seeds, for **12 final models**:

| Cell | Architecture | Objective |
| --- | --- | --- |
| `innovation_aux` | Explicit error correction, 5,978 parameters | Nonquery MSE + prior-query MSE |
| `innovation_spo` | Same explicit model | Same MSE terms + nonquery SPO+ |
| `gru_aux` | Ordinary error-fed GRU, 5,996 parameters | Nonquery MSE + prior-query MSE |
| `gru_spo` | Same ordinary GRU | Same MSE terms + nonquery SPO+ |

Use the immutable shared `otto_prequery_scores` implementation. Within each architecture and seed, both objectives start from identical tensors. Both architectures receive the same 31 public features, legal actions, genuine query scores at absolute steps 0,4,8,... and chronological exposure. A later-query prior is computed before assimilating the observed query. Between-query labels and future scores never enter model inputs. Scores stay anchored to the most recent actual query; only the training objective changes. There is no new architecture in this intervention.

## Fresh data and seeds

Collect **54 TRAIN and 36 VALID complete paths**, using analytic, always-neural and period-four-held-score collectors for each originating case. Each sensing setting, 3 and 4, has nine TRAIN and six VALID cases. Collector paths from a case share source and indexed observation streams and are not independent cases. Preserve the 2,188-step horizon, initial hits `1 + case % 3`, rotating collector order, public filtering, pinned eight-view TensorFlow teacher and terminal updates.

TRAIN seeds: 291000001-291000009 and 292000001-292000009. VALID seeds: 293000001-293000006 and 294000001-294000006. Fit seeds: 295000001, 295000002, 295000003. Window-selection seeds: 296000001-296000054 in global TRAIN episode order. The [prewrite collision review](../output/otto-action-focused-v1/engineering-01/seed-review-01.json) examined 4,131 current source/protocol/plan/seed files and found no digit-delimited occurrence of the 87 values. This is a scoped repository check, not a universal historical-uniqueness claim.

For an episode of length T, let W=ceil(T/4), k=min(8,W), and M=T-W. Sample k disjoint period-four windows without replacement using the held PCG64 sampler, including query-only tails. TRAIN annotation covers those windows plus every actual query row. Missing labels are reconstructed only after the complete trajectory and public-state reconciliation. Annotation cannot change deployed actions or held scores. VALID is a complete teacher-score census. All four cells consume identical projected data and episode orders.

## Exact objective and optimization

Preserve the existing legal-centered nonquery MSE after division by 64. Each selected nonquery row has importance weight W/(k*54*M); M=0 contributes zero. Prior-query MSE centers all four scores, including currently illegal actions, in the coordinates used by correction. With K=ceil(T/4)-1 later queries, each prior row has weight 1/(54*K); K=0 contributes zero. Do not renormalize realized weights or omit zero-support episodes.

For legal set L, teacher q and prediction p, define c=q/64, z=p/64 and u as the uniform distribution over exact legal minima of the original float32 teacher. The added loss is the established finite-action [SPO+ surrogate](https://arxiv.org/abs/1710.08005):

`max_{a in L}(c_a - 2*z_a) + 2*sum_a u_a*z_a - min_{a in L} c_a`.

Use the qualified anchored float32 implementation, with teacher and weight detachment, the first maximizing action for the max subgradient, zero gradient on illegal actions, and connected zero padding. Exact teacher ties are identified before scaling. The deployment near-minimum tolerance is not used in this training loss. SPO+ coefficient is exactly **1**, and prior coefficient remains **1**. These are prospective fixed conventions; no temperature, coefficient, range normalization or clipping sweep is permitted. The loss is not a probability calibration method or a novel RL algorithm.

All fits use CPU float32, one numerical thread, 80 epochs, Adam at 0.003 with zero weight decay, global gradient norm clipping at 5, batches of six complete episodes and chronological chunks of 32. Each paired seed uses the same local permutation of all 54 episodes each epoch. Model parameters stay fixed for the entire episode batch; detach state between chunks, accumulate gradients, then clip and update once. Weighted batch sums use 54/actual_batch_size. Both objective arms include all prior-active chunks. Zero-target batches still receive the defined zero-gradient optimizer step.

There are 720 updates per fit and **8,640 total**. Save all orders, projections, targets, weights, checkpoints, TRAIN scores, and ordinary/prior VALID scores and masks. Save final nonquery, prior and SPO+ loss diagnostics for both arms, with the objective-specific total. Report forward/backward work, whole-fit time, optimizer work and final rescoring. Equal exposure and update counts do not imply equal compute. Do not add per-component backward passes to the scientific training trace merely to measure gradients.

Finish and close **all 12 final checkpoints before decoding VALID**. There is no best-epoch selection, early stopping, missing-cell substitution, post-VALID fitting or reuse of incomplete collection.

## Metrics and frozen continuation rules

Keep the original strict float32 `1e-10` near-minimum sets. Choose the first legal predicted near-minimum, accept any teacher near-minimum for agreement, and subtract the actual minimum teacher float32 score in float64 for the raw gap. This is a teacher-score gap, not observed environment regret. Report initial nonquery steps 1-3, full paths, and primary later nonquery steps at or after 5. Retain all age-1/2/3, case, collector and fit-seed results. Every declared episode retains its equal denominator share within each scope; unsupported episodes contribute zero. Query-prior MSE is a separate all-four quantity.

Scientific comparisons average the three fit metrics equally, separately in each sensing setting. There are **41 unique conditions**, combined into three named decisions, not an omnibus success fraction.

**Eleven common conditions:** one requires complete finite evidence, successful original collection/training/audit supervisors and independent saved-output agreement. For each setting, require at least four originating VALID cases with initial support, at least four with support at each primary age 1/2/3, and a strictly positive held-score primary gap. These are support/closure conditions, not model gains.

**Twelve objective conditions per architecture:** in each setting, SPO+ must satisfy all six:

1. Mean primary gap is at most 0.9 times the corresponding AUX mean and strictly lower.
2. Mean full agreement is at least the corresponding AUX mean.
3. Mean initial agreement is at least the corresponding AUX mean.
4. Primary gap does not increase versus paired AUX at seed 295000001.
5. The same nonregression at seed 295000002.
6. The same nonregression at seed 295000003.

Each architecture's objective decision requires its twelve conditions and all eleven common conditions: **23 checks**. Improvement in both architectures is an objective effect, not an explicit-correction advantage. The ordinary-GRU objective decision is reported independently.

**Six cross-architecture conditions:** in each setting, explicit SPO+ must have primary gap at most 0.9 times the lower of the GRU-AUX and GRU-SPO+ mean gaps, and strictly below that lower mean. Its full and initial agreement must each be at least the higher corresponding GRU mean. This compares against both ordinary controls, rather than gaining from a potentially weakened GRU-SPO+ recipe. Architecture continuation requires these six plus the explicit objective decision's 23: **29 checks**.

No named pass is statistical confirmation or paper readiness. A pass permits only a separately frozen fresh scenario-shift and autonomous quality-versus-total-compute comparison. Both settings here occur in TRAIN and are not an unseen shift. All seeds and failures remain in the report. If ordinary GRU explains a gain, retain that explanation.

## Admission, budgets and closure

Before collection, freeze this protocol, all scientific sources, exact seed reservation and successful fabricated qualification, including production-batch loss integration and independent scalar SPO+ arithmetic. Historical closed prequery capacity evidence is reused only for the unchanged shared architectures and runtime. It does not measure the new SPO+ arithmetic or certify a new speedup. Its historical configuration and successful original supervisor remain authenticated. Actual new-loss costs are recorded during this study.

Collection: **7,200 seconds**, one CPU thread, 4 GiB RSS, 2 GiB outputs. Bounds include 90 native resets, 196,920 moves, 54 public replay resets, 118,152 replay updates, 144 teacher-policy bindings, and 138,708 teacher/TensorFlow-value calls. These account for full neural TRAIN paths, sampled-window plus query-anchor annotations on the other TRAIN paths, and full VALID censuses. Setup, annotation, I/O and cleanup count.

Training and validation: **14,400 seconds**, one CPU thread, 4 GiB RSS, 2 GiB outputs. Independent saved-output audit: **240 seconds**, one CPU thread, 2 GiB RSS, 256 MiB outputs. The audit independently reconstructs losses, metrics, support, work counts and conditions from saved evidence without any model, optimizer, teacher or simulator call.

Each phase authenticates original plans, source/runtime/input hashes and its original supervisor; uses an exclusive directory; and preserves failure receipts and pending work. No scientific retry, resumption, replacement seed, cap extension or training on incomplete collection. Precollection engineering revisions are permitted only before data exposure and must be preserved and requalified. All original studies and their failed decisions remain unchanged.

This screen establishes no biological wiring, Bayesian inference, learned environment dynamics, calibrated uncertainty or novel algorithm by itself. The immediate question is whether a decision-focused objective repairs the observed forecasting-to-choice mismatch under strong ordinary recurrent controls.
