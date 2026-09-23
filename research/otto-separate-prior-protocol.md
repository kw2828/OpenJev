# Fresh comparison of shared and separate forecast readouts

This prospective development screen follows the [saved-output diagnosis](otto-prequery-decision-diagnosis-results.md). That diagnosis localized the mean full-path agreement regression to initial steps 1-3 in both architectures. It did not establish gradient conflict. The old prequery study remains closed at FAIL51/55. No old VALID prediction, checkpoint or selected fit is reused for training or evaluation here.

## Fixed factorial comparison

Train all eight cells at three paired fit seeds, for **24 final models**:

| Cell | Recurrent architecture | Readout | Training objective |
| --- | --- | --- | --- |
| `innovation_shared_mse` | Explicit correction | Shared | Original MSE |
| `innovation_shared_aux` | Explicit correction | Shared | MSE plus prior loss |
| `innovation_separate_mse` | Explicit correction | Separate | Original MSE |
| `innovation_separate_aux` | Explicit correction | Separate | MSE plus prior loss |
| `gru_shared_mse` | Ordinary error-fed GRU | Shared | Original MSE |
| `gru_shared_aux` | Ordinary error-fed GRU | Shared | MSE plus prior loss |
| `gru_separate_mse` | Ordinary error-fed GRU | Separate | Original MSE |
| `gru_separate_aux` | Ordinary error-fed GRU | Separate | MSE plus prior loss |

The explicit separate-auxiliary cell is the architecture candidate. Shared models retain 5,978 / 5,996 parameters; separate models have 6,098 / 6,112. The new readout changes both direct output-parameter sharing and the prediction used for later correction. Recurrent parameters remain shared between objectives. Separate-MSE controls distinguish changes to the readout/correction path from auxiliary supervision; added parameter counts remain a confound for a pure capacity-matched architecture claim.

Use the qualified, pinned model implementations. Both architectures receive the same 31 public features and genuine teacher scores at absolute steps 0,4,8,... only. They retain caller-owned state across queries. A later-query prior is computed before assimilating that query's observed scores. Nonquery predictions remain offsets from the latest actual score anchor. The first query has no prior target. Neither unselected nonquery labels nor future scores may enter model inputs. The ordinary GRU receives the error through its standard recurrent update; explicit correction adds `tanh(B e)`. Scale remains 64.

## Fresh trajectories and identical exposure

Collect **54 TRAIN and 36 VALID complete paths**, with analytic, always-neural and period-four-held-Q collectors for each case. Length settings3/4 have nine TRAIN and six VALID originating cases each. Three collector paths from a case are paired observations, not three independent cases. Preserve the 2,188-step horizon, initial-hit stratification, rotating collector order, paired source/indexed observation streams, public filtering, original eight-view TensorFlow teacher and final updates through discovery or horizon.

TRAIN case seeds are 281000001-281000009 and 282000001-282000009. VALID seeds are 283000001-283000006 and 284000001-284000006. Fit seeds are 285000001, 285000002, 285000003. TRAIN window-selection seeds are 286000001-286000054 in global TRAIN episode order. The prewrite [scoped collision review](../output/otto-separate-prior-v1/seed-review-01.json) enumerates 3,263 current source/protocol/plan/seed/config/manifest files and finds no digit-delimited occurrence of the 87 reserved values. It does not claim universal historical uniqueness.

For length T, W=ceil(T/4), k=min(8,W), M=T-W. Keep the held PCG64 sampling of k disjoint period-four windows without replacement, including query-only tails. TRAIN annotations combine those windows with all actual query rows. Deployment returns are reused; missing annotations are reconstructed only after a complete path and public-posterior/final-update reconciliation. Annotation never changes deployed actions or cache state. VALID is a complete teacher-score census. All eight cells consume exactly the same projected data and orders.

## Loss and optimization

Keep original eligible nonquery MSE, centered over legal actions and scaled by 64. Each selected nonquery row receives weight W/(k*54*M), or zero if M=0; do not renormalize realized sampling weights. Later-query prior MSE centers all four scores, including currently illegal actions, matching correction arithmetic. For K=ceil(T/4)-1, each prior row has weight 1/(54*K); K=0 contributes zero. Auxiliary coefficient is exactly 1. MSE-only optimization must not numerically inspect auxiliary labels or weights. Final diagnostic rescoring can report both components without changing its objective.

Each fit uses 80 epochs, Adam 0.003, zero weight decay, norm clipping 5, CPU float32, six complete episodes per batch and chronological chunks of 32. At a paired fit seed, all cells use the same local permutation of 54 episodes each epoch. Parameters stay fixed throughout the batch; gradients accumulate across chunks with detached state, then one clip and one optimizer update occurs. Weighted batch sums scale by 54/actual_batch_size. Zero-target chunks can run without autograd; all forwards still count. Zero-target batches receive a defined zero-gradient Adam update for every parameter.

There are 720 updates per fit and 17,280 total. Equal updates are not equal computation: record per-cell forward/backward work, optimizer work, whole training wall time and all final rescoring. Save input projections, targets, weights, orders, final checkpoints, TRAIN predictions and every VALID ordinary/prior prediction and mask. Finish **all 24 checkpoints before decoding VALID**. No best-epoch selection, early stopping, optimizer change, fitting after exposure or missing-cell substitution.

## Metrics and distinct continuation decisions

Preserve strict float32 near-minimum selection at 1e-10: choose the first legal near-minimum predicted action, accept any teacher near-minimum for agreement, and calculate raw chosen-minus-best teacher gap in float64 from original float32 scores. Report initial nonquery steps 1-3, full nonquery trajectories and primary postcorrection steps 5 onward. Every declared episode retains equal denominator share within its scope; unsupported episodes contribute zero and support is reported. Prior MSE is centered all-four raw-score MSE averaged over later queries within each episode, again retaining zero-support episodes.

All scientific comparisons below average the three paired fit metrics equally, separately in each setting. Retain all seeds, cases, collector arms and postcorrection ages 1/2/3. Also report the readout-by-loss interaction: `(separate_aux - separate_mse) - (shared_aux - shared_mse)` for initial/full agreement, prior MSE and primary gap, both by seed and as means. This interaction is descriptive and does not isolate gradient conflict.

There are **39 unique conditions**, combined into named decisions rather than one omnibus pass rate.

**Nine common conditions:** complete collection/fits, finite evidence, successful original supervisors and independent saved-output audit form one technical condition. The remaining eight require at least four distinct originating VALID cases with support in the initial scope and each of the three primary ages, in each setting.

**Ten mechanism conditions per architecture:** for each setting, separate-auxiliary must meet all five:

1. Full agreement is at least the maximum of shared-auxiliary plus 0.01, shared-MSE and separate-MSE.
2. Initial agreement is at least the maximum of shared-auxiliary plus 0.01, shared-MSE and separate-MSE.
3. Full raw gap is no greater than the minimum of shared-auxiliary, shared-MSE and separate-MSE.
4. Primary raw gap is no greater than the minimum of shared-auxiliary and 0.9 times separate-MSE, and strictly smaller than separate-MSE.
5. Prior MSE is no greater than the minimum of shared-auxiliary and 0.8 times separate-MSE, and strictly smaller than separate-MSE.

The 0.01 agreement increment is one percentage point. Strict comparisons prevent zero errors from being treated as an improvement. Each family's mechanism decision requires its ten conditions plus all nine common conditions: **19 checks**. This demands meaningful initial/full agreement recovery while preserving forecast and later-decision gains. A benefit in both families is a readout/objective result, not an explicit-correction advantage.

**Ten architecture comparisons:** in each setting, explicit separate-auxiliary must beat or match ordinary-GRU separate-auxiliary on primary agreement, initial agreement, full agreement and full raw gap. Its primary gap must be at most 0.9 times the ordinary GRU's and strictly smaller. An explicit-architecture continuation requires these ten plus the explicit mechanism's 19: **29 checks**. The ordinary GRU's separate mechanism gate is reported independently.

This small development screen is not a significance test, autonomous result or paper-novelty threshold. A named pass permits only a separately frozen fresh scenario-shift and autonomous utility-versus-total-compute comparison. Both remain required before advancing an efficacy claim. If only the ordinary GRU benefits, continue with that result and retire this explicit-correction recipe. Do not select favorable seeds, settings or collector paths.

## Capacity and execution

Before collection, freeze all scientific sources, this protocol, seed review and successful fabricated qualification. Run one synthetic capacity probe of the actual new training batch for each of the eight cells: six fabricated length 2,188 episodes, chunks 32, eight sampled windows per episode and all later-query prior targets. It includes chronological forwards, backward accumulation, clipping and one optimizer update per cell. No empirical trajectory, saved fit, teacher or environment is used.

Admission is fixed as `1.5 * 3 * 720 * sum(eight full-batch seconds) + 180 <= 16200` seconds, inside a 21,600-second training cap. The 50% multiplier and fixed 180-second overhead reserve room for final rescoring, evaluation and closure. This is a capacity estimate, not a guarantee. Probe cap: 180 seconds, CPU 1, 4 GiB RSS, 128 MiB outputs.

Collection cap: 7,200 seconds, CPU 1, 4 GiB RSS, 2 GiB outputs. Teacher-value cap 138,708 includes 18 neural TRAIN paths at 2,188 calls, 36 other TRAIN paths at 547 query anchors plus 24 sampled nonquery rows, and 36 full VALID censuses. Other bounds retain 90 native resets, 196,920 moves, 54 public replay resets, 118,152 replay updates and 144 teacher-policy bindings. Setup, annotation, I/O and cleanup count.

Training cap: 21,600 seconds, CPU 1, 4 GiB RSS, 2 GiB outputs, including all 24 fits and scoring. Saved-output audit cap: 240 seconds, CPU 1, 2 GiB RSS, 256 MiB outputs. Audit cannot call a model, optimizer, teacher or environment. Every phase authenticates its original plan, source hashes, inputs and supervisor; uses an exclusive output directory; and preserves pending work and failure receipts. No scientific retry, resumption, replacement seed, cap extension or training on incomplete collection. Precollection engineering failures remain recorded and can be revised only under a new qualified source revision.

No result from this protocol alone establishes biological wiring, Bayesian inference, a learned environment dynamics model, calibrated uncertainty or a new RL algorithm. The immediate question is whether a small readout intervention resolves a measured recurrent-training tradeoff, and whether any advantage survives ordinary recurrent controls and paid computation.
