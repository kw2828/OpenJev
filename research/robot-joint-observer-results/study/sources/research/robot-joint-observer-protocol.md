# Joint observer and dynamics: exposed development protocol

Version `robot-joint-observer-study-v1`. This protocol is prospective before twelve fresh fits. The [position-only observer study](robot-position-observer-results.md) closed with all six fits complete and only three of five performance criteria passing. The present intervention allows its transition and correction gain to learn together. It preserves every earlier failure and uses no new confirmation data.

## One candidate and three fresh controls

All four arms start from the exact same archived last-two transition for each seed 8101, 8102 and 8103. That transition has 590 parameters and was previously selected at rate .001. Copy its weights only, with fresh empty Adam state in every fit. Do not use the archived temporal model as a better warm start.

| Family | Trainable values | Conditioning |
|---|---:|---|
| `joint_observer_long`, designated candidate | 662 | Thirty observed-prefix corrections |
| `joint_observer_short` | 662 | One correction from three recent positions |
| `continued_last_two` | 590 | Last two positions, unchanged initializer |
| `continued_temporal` | 962 | Existing thirty-feature temporal head, initially zero |

Both observers have the same unrestricted 12-by-6 gain initialized at `[I;0]` and the same trainable 590-parameter `dense_mlp` transition. The long observer starts at `z1 = concat(q1,q1-q0)`. For t=2 through 31, compute `prior=f(z[t-1],u[t-1])`, then `z[t]=prior+K(q[t]-prior[:6])`. Prepare the transition once for this conditioning request without detaching its gradients or storing a hidden cache.

The short observer starts at `z30=concat(q30,q30-q29)`, predicts once with `u30`, and corrects once with `q31`. It uses three positions, not two. Both forecast from state 31, consuming `u31` first to predict `q32`. Future position observations never enter the forecast. The continued controls preserve the qualified `HistoryInitializedDense` arithmetic, state dictionary and feature semantics. All arms use the same 12-state transition and direct position readout. The lower six coordinates are learned state, not supervised physical velocity.

A new joint-training implementation must enforce its own trainable-parameter contract. Do not modify the qualified frozen observer or disable its validation. Every parameter is trainable during fitting, with no fixed gain buffers. The two observer arms have equal parameter counts, but conditioning depth also changes their function classes, computational graphs and optimization. This study alone cannot attribute a difference exclusively to memory or observation information.

## Fixed fitting budget and numerical policy

Execute exactly twelve attempts in family order as listed above, then increasing seed within each family. Use the original per-seed 4,096 paired batches, batch size 16, CPU float32, H128 normalized position MSE, Adam betas .9/.999 and epsilon 1e-8, learning rate **.001 for every fresh fit**, and native gradient-norm cap 1.0. This rate is prospective and has no new search or selection. Each model sees the same additional training windows; equal update exposure is not equal compute. No auxiliary prefix loss, teacher-forced future rollout, learned readout, gain constraint or new latent dimension is introduced.

Keep the installed native `clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=False)` operation and its nonfinite-norm failure guard. All trainable gradients, including transition parameters, enter that single operation. Do not combine joint training with changed precision, overflow-safe clipping, optimizer-state continuation, early stopping, a rescue or a retry. Clear gradients between updates. Preserve failed attempts, partial traces and their last available parameter/optimizer states. Source, schema and I/O defects stop the campaign rather than becoming numerical losses.

Use only the seven inherited processed FIT recordings; inherit normalization, batches and reference coefficients without refitting. Reconstruct and verify paired batch identities from their original seed and geometry. No raw MAT decoding, official TEST access, new dataset or reference fitting is permitted. Complete all twelve attempts and save all 57 fresh/inherited model states before decoding any exposed evaluation arrays in this process.

Detached diagnostics observe each training attempt. Keep the five prefix fields `prefix_steps`, `max_state_abs`, `max_state_norm64`, `max_innovation_abs` and `max_innovation_norm64`; long and short normally report thirty and one corrections, while continued controls report zero and summarize their returned state. Record gradient-entry count, nonfinite-entry count, maximum absolute value and scaled float64 aggregate norm before native clipping. Save the native float32 norm separately. Diagnostics do not change forward values, gradients, optimizer arithmetic or decisions. A missing pre-backward capture is null, not a zero gradient. The last attempt retains raw available gradients by parameter name and native norm; earlier attempts retain scalar summaries. There are **no additional numerical diagnostic probes**.

Save every fresh initial/final checkpoint, Adam array, trace, per-attempt diagnostics, diagnostic aggregate and fit receipt. Verify each initial cell equals its seed's inherited last-two cell, gain/head initialization is exact, every parameter enters the fresh optimizer and no hidden state carries between windows. Final cell equality is not required because learning the dynamics is the intervention. Fit times include diagnostics within their declared scopes; no training speedup or diagnostic-overhead claim is planned.

## Retained comparisons and exact evidence roster

Copy all **488 parent score rows and 244 forecast-attempt records unchanged** under the parent's original closed run, agreeing independent audit and publication pins. Retain its eighteen complete selected families, including the position observer selected at .003. The older failed learned observer remains displayed but ineligible. Add one separately labeled eligible comparison, `frozen_position_lr001`, by aggregating the parent's already saved position-observer .001 rows. This same-rate frozen comparison does not duplicate, relabel or rescore the saved metric rows. Its timing identity must map explicitly to the original checkpoint key.

With four fresh families there are **24 displayed families and 23 eligible families**. The prospective candidate faces 22 controls. Keep all earlier declared rates and failures in the underlying records. No historical family is reselected or refit; the three continued controls are distinct fresh arms, not replacements for archived models.

Registration pins **69 inputs**: nine common payloads (normalization, linear coefficients, both causal-ridge NPZ/JSON pairs and three batch banks), eleven processed recordings, four immediate-parent JSON payloads (results, fits, resources and prediction attempts), and all 45 immediate-parent final model states. Copy the 58 nonmeasurement payloads opaquely into the child study. Resolve the four JSON payloads and all model states from the immediate position-observer study, not a nested grandparent directory. Common payloads and processed recordings retain their authenticated original descriptors.

Register all **63 source files**: the parent's 54 scientific sources, its publisher and publisher test, and seven new core/test/runner/test/protocol/auditor/test sources. Pin the launcher separately. Qualify these exact bytes with fabricated cases, preserve every qualification attempt and snapshot, then commit the registration and all sources before the original run. Parent admission and registration preflight hash bytes and read JSON only; they do not decode measurement arrays or call models.

## Forecasts, costs and decision rule

All four evaluation recordings are already exposed data from the same robot/day: original DEV `21H_54M` and `22H_10M`, former confirmation `22H_41M` and `22H_50M`. None selects a new rate. Keep causal preprocessing, decimation and FIT normalization unchanged. Each file has 22 windows at starts `64+160*k`. At start s, context is `[s:s+32]`, future torque `[s+31:s+159]` and target positions `[s+32:s+160]`. Realized future measured torque defines conditional system identification, not command-driven robot control.

The twelve new models produce **48 new forecast attempts and 96 H64/H128 rows**, making **292 attempts and 584 rows** with the complete parent evidence. Save every available new prediction bank; unavailable failed-training predictions remain explicit failures. Parent forecast banks are not regenerated. H64, per-joint errors and paired-seed outcomes remain descriptive alongside the primary H128 metric.

Time **61 current complete-request slots**: twelve fresh instances, the parent's 46 selected/reference instances, and three additional frozen-position .001 instances. Failed fresh fits remain explicit unavailable cost slots; they are not timed as successful recipes. Use the first original DEV1 window, three warmups and twenty retained timing samples. Set Torch and all five numerical thread variables to one. Include normalization, casts, all conditioning/preparation work, H128 rollout, finite validation, denormalization and the deadline callback. Disable detached training diagnostics for inference and rebuild prepared operators per request.

Charge every trainable/frozen weight, fixed buffer, caller-owned state and normalizer, with no cross-model sharing. The two fresh observers each use 2,888 persistent numeric bytes; continued last-two uses 2,600 and continued temporal uses 4,088. Include 48 state bytes and 192 normalizer bytes. Report request storage separately. Temporary/autograd workspace, loading and Python-object overhead are outside this logical persistent count and must not be represented as measured peak memory.

For each eligible neural family, average the three individual seed H128 RMSEs within a file, then weight the four file means equally. Deterministic references have one value per file. Family latency is the median of its instance medians. All five criteria are required:

1. Complete finite eligible forecasts at both horizons and complete finite timings for all 61 declared slots.
2. Candidate equal-four-file mean at most .95 times the best of all 22 controls, with a strictly positive control mean.
3. On every file, candidate mean at most 1.02 times the best of the three **fresh** simple controls: short observer, continued last-two and continued temporal.
4. Candidate full-request latency at most 1.5 times concurrent **continued last-two** latency.
5. No complete eligible control matches or improves candidate error, latency and persistent storage simultaneously, with at least one strict improvement. Missing eligible outcomes make the complete frontier fail closed.

The literal scientific result is `JOINT_OBSERVER_DEVELOPMENT_PASS` or `JOINT_OBSERVER_DEVELOPMENT_FAIL`. A bookkeeping/audit PASS does not alter this result. A pass justifies separately planned unexposed method transfer; it neither repairs the prior confirmation nor establishes novel architecture, biological wiring, Bayesian calibration, global stability or closed-loop control. If short conditioning or continued conventional models explains the gain, report that result without promoting long-prefix memory.

## Closure and independent audit

Each fit has a suspend-aware 1,800-second cap. The full original process has a suspend-aware 7,200-second cap, plus an external 7,260-second cap. Preserve partial outcomes and the original terminal receipt. Observation timeouts never authorize restarting a live handle; a closed failed campaign is not silently resumed.

After original closure, the independent auditor verifies sources, input provenance, the original process/qualification closure, 57 final states, twelve initialization/Adam/diagnostic bundles, all 48 scheduled new forecast outcomes, independently reconstructed scores/rules and 61 timing records. Initial cells and gain/head values must match the prospective warm-start contract; optimizer moments must correspond to all trainable values and the reported completed steps. Reconcile raw last-attempt gradients and native norms with their recorded diagnostics without new backward, clipping, training or timing calls. Recurrence replay may use qualified model arithmetic; metric/rule reconstruction must be independent of producer selection code.

Copy parent metrics and attempts exactly rather than recount them as new forecasts. Preserve every source snapshot, failed outcome and original process receipt. Keep measured arrays and target-window files local under hashes and separate dataset terms. Joint state estimation and dynamics learning have established precedents linked in the [design note](robot-observer-next-direction.md); this experiment tests the declared operating point and comparison rather than asserting novelty in the mechanism.
