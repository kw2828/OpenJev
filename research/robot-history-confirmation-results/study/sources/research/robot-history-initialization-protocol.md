# Robot history initialization: prospective development protocol

Pre-fit protocol, version1. Current execution status is tracked in the [experiment archive](experiment-index.md). Its three-arm design was written before the reflection-capacity campaign closed; no partial outcomes were used to choose the initializer features or five admission criteria. The [capacity result](robot-reflection-capacity-results.md) is now closed and negative. The inherited reference roster, deadlines and diagnostic permutation below were finalized afterward. A separate committed source, data and qualification registration is required before fitting.

## Question and models

Test whether a cheap summary of the observed prefix improves initialization of the same compact transition. This is an information ablation, not a novel memory architecture. The [design note](robot-history-initialization-design.md) identifies the relevant history-encoder, observer and Kalman-network prior art. An innovation observer and online weight adaptation are deferred.

Use three primary arms around the unchanged `StructuredRobotTransition('dense_mlp')`:

| Arm | Extra initialization features | Parameters | Persistent numeric bytes |
| --- | --- | ---: | ---: |
| Last two | Original initializer | 590 | 2,600 |
| Local affine | Recent position, difference and preceding torque, plus fixed local square/tanh features | 962 | 4,088 |
| Temporal affine | Identical recent inputs, plus older position slope and older torque mean | 962 | 4,088 |

All arms retain twelve float32 forecast-state values. Bytes include learned parameters, state and four float64 normalization vectors; temporary workspace, request arrays and optimizer state are separate. No retained prefix or prepared-operator cache is allowed.

For a normalized 32-sample context, let `dq = q[31]-q[30]` and `c = concat(q[31], dq, u[30])`. The local features are `concat(c, dq*dq, tanh(u[30]))`. The temporal features are `concat(c, slope(q[0:30]), mean(u[0:30]))`. Compute each slope using weights `t-14.5` for `t=0..29`, divided by `2247.5`. Both feature vectors have thirty entries.

Both affine arms add `W*features+b` to the original twelve-value initializer. Their 12-by-30 weights and twelve biases start at zero. The shared transition parameters are identical for each paired seed, so all three initial functions agree. After training, the correction may alter all twelve latent coordinates, including the first six. Do not claim that the learned initial state preserves the final observed position exactly or represents a Bayesian posterior.

The two affine heads have equal stored parameter counts and common recent inputs. Their feature spaces and optimization conditioning differ. This is not a claim of identical effective function capacity. The local extra features are functional, with no padding or deliberately inactive weights.

No initializer uses `u[31]`: that torque first enters the forecast to predict `q[32]`. Scored forecasts receive no later position, innovation, teacher-forcing correction or reference-model state. Each request starts independently. Input and parameter nonfinites are errors, with no clipping or repair.

## Training and evidence

Use the same seven saved FIT recordings, two already-exposed DEV recordings, FIT-only normalization, three paired batch files, context32 and horizon128 as the closed structured study. No raw MAT decode, changed filtering, CONFIRM or official TEST access. The inputs remain realized measured torques, so this is conditional forecasting rather than verified command-driven control.

Fit all three arms concurrently as an experimental comparison, starting each from scratch. Execute fits sequentially on the host. Seeds are 8101, 8102 and 8103; learning rates are .001 and .003. This gives eighteen fresh attempts. Every attempt receives 4,096 updates, batch16, the original normalized position MSE, Adam betas (.9,.999), epsilon 1e-8 and gradient clipping at1. Use the original paired windows, including offset520000. Add no prefix reconstruction loss or feature statistics fitted on DEV.

Native suspend-aware deadlines are 1,800 seconds per fit and 14,400 seconds for the complete run, including preservation and evaluation. The external launcher cap is 60 seconds beyond the whole-run limit. If preservation crosses a fit cap after the underlying helper records success, retain that original receipt and mark the effective fit status failed. There is no retry or restart. Freeze the launcher, all source hashes, exact inherited payloads and original qualification receipts in the registration before fitting. Qualification uses fabricated inputs only. It must verify the slope independently, exact initial-function/common-parameter parity, causal torque alignment, gradients, finite guards, request isolation and complete storage accounting.

Retain all attempted fits, initial/final weights, optimizer states and update traces. Close all eighteen attempts before this campaign reads DEV. This barrier does not erase prior development exposure. Reuse the same22 windows per DEV file, all H64/H128 scores and all three seeds at both rates. Select one learning rate per arm using pooled H128 standardized SSE over both files and all seeds; break ties toward the lower rate. A failed seed makes that rate ineligible. Never select the best seed or average predictions into an ensemble.

Reuse all thirty fits from five inherited neural families at both original learning rates, with three seeds per rate:

| Inherited family | Closed source study | Source family |
| --- | --- | --- |
| GRU10 | `robot-transition-study-v1` | `gru_residual` |
| GRU32 | `robot-structured-study-v1` | `gru32` |
| Legacy instant scheduler | `robot-structured-study-v1` | `legacy_instant` |
| Bounded dense | `robot-structured-study-v1` | `dense_bounded` |
| Unbounded dense | `robot-structured-study-v1` | `dense_unbounded` |

These controls all used the same 4,096 updates, H128 training horizon, .001/.003 rates and paired batch offset520000. The earlier 1,024-update/H64 coupling-study GRU is not the GRU10 reference here. Retain all original recipe records and initial/final/optimizer/trace/receipt artifacts, authenticate their closed audits and manifests, and recompute both-rate predictions under the same selection rule. Cached fits receive zero new optimizer updates.

Keep both original causal-ridge banks, penalties1/100, frozen linear AR2 and persistence as additional references. Their coefficients and fitting times remain historical. Retime every selected control on the current host; do not silently refit or expand any tuning search. All declared comparison controls, including both ridge banks, enter the domination test. A missing or invalid declared control makes the decision ineligible rather than silently removing that competitor.

If all models remain finite, the ordinary evaluation contains48 fit records,104 forecast banks,208 H64/H128 score rows and28 timing rows. Selected-primary permutation diagnostics add18 forecast banks and36 descriptive score rows. These are expected complete counts, not permission to omit failed attempts.

## Development decision

The primary candidate is temporal affine. No observer or other model can substitute for it after seeing outcomes. Define each file's error as the mean H128 standardized RMSE across its three selected seed fits, then take an equal-weight mean of the two files.

Advance only if all of the following hold:

1. All three primary arms have complete finite selected recipes.
2. Temporal affine lowers the equal-file mean by at least5% versus both the local-affine and unaugmented controls. Each comparison requires positive baseline error; a zero-error tie is not a relative improvement.
3. Neither file's temporal-affine mean is more than2% worse than the better local-control mean on that file.
4. Its complete-request median latency is at most1.25 times the concurrent unaugmented control.
5. No declared comparison control is no worse in equal-file error, full-request latency and persistent numeric storage, with a strict advantage on at least one axis.

The 5%, 2% and25% margins are prospective development utility choices, not significance thresholds. This measures one fixed-exposure operating point, not a complete scaling frontier. A quality gain bought with extra latency is a tradeoff, not a speedup. Report per-file means, every paired seed difference, both rates, H64 errors and per-joint errors even though they are not dozens of separate pass conditions.

Timing uses batch1 on the first registered DEV window, three warmups and twenty repetitions per selected fit. Include normalization, conversion, prefix features, conditioning, validation, transition preparation, all128 forecast steps, denormalization and finite checks. Use the same deadline-callback policy, runtime and five single-thread settings for every arm. Aggregate each family's three fit medians by their median. Report raw durations, training time, shared-host identity and temporary versus persistent storage.

## Mechanism and continuation

The fixed paired position/torque permutation is `[29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,30,31]`. It reverses older indices0..29 and preserves indices30/31, the boundary torque and all future inputs. Publish all ordinary and finite permuted forecasts for the selected primary recipes. Retain any numerical failure of the temporal model on the corrupted prefix as a failed diagnostic attempt and error rows; it does not change the primary decision. The local arms must be exactly invariant; a numerical failure or mismatch invalidates the diagnostic implementation, not a new sixth scientific criterion. Structural or schema errors remain fatal. The temporal torque mean is order invariant in real arithmetic, although floating-point reduction order can change its last bits; its slope can change materially. Corruption is an off-distribution diagnostic, not an alternative way to pass the primary decision or proof of temporal reasoning.

The saved signals have undergone causal filtering. A useful history summary might recover preprocessing state, unobserved physical state or a task-specific statistical regularity; this experiment cannot identify which. A physical-memory claim would need a separately designed comparison with matched raw/filtered information and verified action semantics.

Independently replay saved checkpoints and reconstruct scores and the full decision before reporting an outcome. A pass only supports a separately frozen confirmation study. It does not establish architectural novelty, calibrated uncertainty, biological-wiring value, or an ICLR-ready result. A failure rejects this specific information contrast; adding more variants to the same DEV outcomes requires a new justified hypothesis rather than changing its rule.
