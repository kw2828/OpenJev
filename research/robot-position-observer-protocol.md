# Position-only observer initialization: exposed development protocol

Version `robot-position-observer-study-v1`, prospective before the six new fits. The [frozen-dynamics observer study](robot-observer-results.md) is closed and failed its rule. This study changes its gain initialization from `[I; I]` to `[I; 0]`. It preserves the earlier failure and uses no new confirmation data.

## Intervention and controls

For each seed 8101, 8102 and 8103, inherit the identical 590-parameter transition from that seed's previously selected last-two checkpoint at rate .001. Every transition parameter remains frozen. The new learned gain is a full 12-by-6 matrix with 72 trainable values, initialized to `[I; 0]`; its bottom six rows may subsequently learn. The paired fixed control retains `[I; 0]` as an explicit 72-value buffer. Both have the same initial function and twelve persistent state values.

Keep the qualified observer recurrence: start at observed index 1 with `concat(q1, q1-q0)`. For indices 2 through 31, compute `prior = f(state, u[t-1])`, then `state = prior + K @ (q[t] - prior[:6])`. Prepare the frozen transition once for these thirty updates. The first forecast consumes `u31` and predicts `q32`; future observations never enter the forecast. Gradients must pass through the frozen transition without detaching state. No prefix cache or hidden trajectory is retained.

The intervention initially corrects only coordinates directly read out as position. The lower six coordinates are learned latent coordinates, not verified velocity. `[I; 0]` is a conventional observer initialization, not a calibrated Kalman gain or a stability guarantee.

Retain the complete parent evidence: all 416 score rows and 208 forecast-attempt records are copied without changing values. Freeze the parent's selected rates for its sixteen complete families: last-two, frozen-transition local and temporal affine heads, fixed `[I; I]`, zero gain, seven historical neural families, both causal ridges, frozen linear AR2 and persistence. The failed parent learned observer remains displayed with its failures, but has no eligible recipe and is not a comparator that automatically defeats the new experiment. There are nineteen displayed families and eighteen eligible families after adding the new learned and fixed position observers. Do not reselect or refit an old family.

## FIT-only diagnostic probes and training

Before any new fit, perform exactly seven diagnostic probes:

1. For each of the three seeds, evaluate the old `[I; I]` initialization and new `[I; 0]` initialization on that seed's first inherited FIT batch: six probes.
2. Evaluate the archived failed parent seed 8103/rate .001 final checkpoint on inherited batch index 22, using zero-based indexing: one probe. This is the saved failed checkpoint, not a reinitialized model or a resumed fit.

Each probe performs a public-input forward pass, the same H128 MSE backward pass, and the original native float32 gradient clipping operation. There is no optimizer step, parameter update, adaptive learning-rate change or retry. Clear gradients between probes and verify unchanged weights. Retain known numerical failures and partial diagnostics. A probe outcome cannot promote the model, remove a fit or change the six-fit schedule. Learning rate does not affect a forward/backward probe with no optimizer step, so initial probe differences cannot be attributed to learning rate.

Read diagnostics before clipping. Record gradient `numel`, `nonfinite_count`, `max_abs`, `norm64` and `all_finite`; compute diagnostic norms in float64 from the existing gradients. The diagnostic helper returns zero counts and magnitudes if invoked with no gradient entries. Probe and training records instead retain a null gradient field when backward did not complete and no summary was captured. With nonfinite entries, report their count and null magnitudes, not repaired gradients. Preserve the native float32 clipping result separately, including nonfinite outcomes. Each probe saves its available raw pre-clipping gain gradient and native norm in `preclip.npz`; it can be empty if failure precedes capture.

Optional per-call prefix diagnostics contain exactly `prefix_steps`, `max_state_abs`, `max_state_norm64`, `max_innovation_abs` and `max_innovation_norm64`. State maxima include the initial state and finite prior/corrected states; norms are maximum per-row float64 norms across the batch. `prefix_steps` counts successful corrections, normally thirty. A failed operation remains an explicit error even when only earlier finite diagnostics are available. Each fit saves per-attempt scalar summaries in `diagnostics.json`, aggregated counts and maxima in `diagnostic-summary.json`, and only its last attempt's available raw gradient and native norm in `last-gradient.npz`. No prefix tensors or trajectories are retained. Diagnostics observe the computation and do not repair, clip, rescale or detach its learning path. Their training overhead is included in fit time; inference passes no diagnostic collector.

Run six fresh learned-gain fits: three seeds by rates .001 and .003. Use the original 4,096 paired batches, batch size 16, H128 forecast MSE, CPU float32, Adam betas .9/.999, epsilon 1e-8 and native gradient-norm cap 1.0. Freeze all 590 transition values and allow optimizer state only for the gain. Keep every initial/final checkpoint, Adam array, loss/gradient trace, diagnostic aggregate and original fit outcome. No restart, rescue, extra fit, early stopping, ensemble or seed selection is permitted. Equal update exposure does not imply equal training cost.

Use only the seven already processed FIT recordings for fitting and probes. Inherit normalization and all reference coefficients without refitting. Verify the original paired batch arrays and their provenance. Finish all six new attempts and save their final checkpoints before numerical access to any of the four evaluation recordings.

## Evaluation and fixed development rule

The original DEV files `21H_54M` and `22H_10M` alone select one new learned-gain rate by summed H128 SSE over both files and all three seeds. A recipe must be complete and finite; the lower rate wins an exact tie. The exposed former-confirmation files `22H_41M` and `22H_50M` never select a rate. All four recordings are already exposed development data from the same robot and day.

Use unchanged causal filtering, decimation and FIT normalization. Each file has 22 starts `64 + 160*k`, context 32 and H128. At start `s`, context is `[s:s+32]`, future torque is `[s+31:s+159]`, and targets are `[s+32:s+160]`. Retain H64 and per-joint physical errors descriptively. Future torques are realized measurements, so this is conditional forecasting rather than command-driven control.

Evaluate the six fresh and three fixed new models on four files: 36 new forecasts and 72 score rows. Combined with the unchanged parent evidence, retain 244 forecast attempts and 488 H64/H128 rows. Parent forecasts are reused only under the original agreeing audit and byte pins. Do not regenerate parent forecast banks; parent model calls are limited to the declared probes and current request timing.

Average three seed H128 RMSEs within each file, then average the four file means equally. Deterministic references supply one value per file. All five criteria are required for a development pass:

1. Every selected forecast at both horizons is complete and finite, and all 46 planned cost slots have complete finite timings. No eligible new learned rate yields three explicit unavailable slots and fails this criterion.
2. The new learned observer has at least 5% lower equal-four-file mean error than the best of all seventeen eligible controls. The comparison error must be positive; a zero-error tie is not a relative improvement.
3. On every file, its mean error is at most 1.02 times the best of last-two, the parent's frozen-transition local and temporal heads, fixed `[I; I]`, and new fixed `[I; 0]`.
4. Its full-request family median latency is at most 1.5 times last-two.
5. Among the complete eighteen-family frontier, no control has error, latency and persistent numeric bytes all no greater, with at least one strict improvement.

The parent's ineligible learned observer is excluded only from these prospective comparisons, not from the retained records, figures or account of failure. Report every paired seed, both original DEV files and both exposed former-confirmation files. Five passing criteria would support further development and a separately designed new-environment evaluation, not restore the failed confirmation or establish novelty.

## Cost, admission and audit

Retain 45 evaluated neural model states: 36 inherited selected models, six fresh models and three new fixed models. The archived failed checkpoint used by one probe is additional diagnostic evidence. Time 46 slots: the parent's forty complete selected instances, three new learned instances or explicit unavailable placeholders, and three new fixed instances. Use the first original DEV1 window, three warmups and twenty retained samples, one Torch CPU thread and all five numerical thread variables set to one. Include normalization, casts, all prefix/preparation work, rollout, finite checks, denormalization and the native deadline callback. Family latency is the median of its three instance medians. Do not reuse prepared transitions across requests.

Charge all frozen/trainable parameters, fixed gain buffers, persistent state and normalizers. The learned model has 662 parameters and the fixed model has 590 parameters plus 72 buffered values; each totals 2,888 persistent numeric bytes in float32 including 48 state bytes and 192 normalizer bytes. Report request arrays separately. Temporary workspace, Python overhead and loading remain outside that logical storage count.

Authenticate the parent's original closed run, agreeing audit and publication before any numerical decode. Registration pins all 54 source files, the launcher, qualification evidence and 61 inherited input descriptors, including the four parent JSON payloads (results, resources, fits and prediction-attempts) and the archived `observer_learned-8103-lr0/final.npz` checkpoint used by the seventh probe. The exact paths, roles and descriptors must be explicit in the committed registration before execution. Preserve source snapshots and every qualification attempt. Neither raw MAT decoding, official TEST access, new normalization nor reference fitting is permitted; existing parent admission may hash historical media opaquely.

Each fresh fit has a suspend-aware 1,800-second cap. The whole original study, including probes, diagnostics, fitting, evaluation and preservation, has a suspend-aware 7,200-second cap and external 7,260-second cap. Retain partial evidence and the original terminal outcome on failure. Numerical failures remain scheduled failed rows; source, schema and I/O defects stop the campaign. There is no retry authorization.

After original closure, an independent auditor checks frozen-cell and gain-only optimizer evidence, all new saved forecasts and metrics, DEV-only selection, the five rules, timing samples and diagnostic structure/counts. Parent scores and attempt records must join their original audited values exactly; they are not rescored or replayed as new forecasts. New model recurrence replay uses qualified code, while scoring and rules are reconstructed independently. This does not replay every training gradient. Keep all measured arrays local under descriptors; preserve derived evidence and failed outcomes. Conventional state estimation, not biological plausibility, calibrated uncertainty, control success or architectural novelty, is the scope of this intervention.
