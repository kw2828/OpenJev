# Fixed-checkpoint confirmation of observed-history initialization

Prospective protocol, version 1. The [development study](robot-history-initialization-results.md) passes its five fixed criteria, including a 5.21% mean gain over the equally sized local initializer. That result authorizes this separate confirmation comparison under the active research goal. This protocol, source qualification and exact checkpoint/data registration must be committed before any numerical decoding of the reserved recordings.

## Question and fixed candidates

Does the selected temporal-affine recipe retain useful forecasting quality and cost on the two previously reserved recordings? This tests a selected recipe without further training. It is not a new architecture, another development search or the official benchmark evaluation.

Use all three existing seed checkpoints, 8101, 8102 and 8103, at each family's fixed development-selected learning rate:

| Family | Fixed training rate |
|---|---:|
| Last-two initializer | .001 |
| Local affine initializer | .003 |
| Temporal affine initializer, primary | .003 |
| Bounded dense | .001 |
| Unbounded dense | .003 |
| GRU32 residual | .003 |
| Legacy instant scheduler | .001 |
| GRU10 residual | .003 |

Load these 24 final checkpoints from the closed history study. Keep causal ridge penalties 1 and 100, frozen linear AR2 and persistence. Neither a favorable seed nor a different learning rate can replace a scheduled checkpoint. Do not refit coefficients, initialization heads, normalization or model parameters. No optimization, temperature fitting, ensembles, online weight adaptation or fallback recipes are allowed.

The common dense transition has 590 parameters; each affine initializer adds 372. Initialization is part of the trained model. The local and temporal models have equal parameter counts but different feature spaces. All models reset between requests and receive the same observed positions and measured torques. Forecast steps receive no later observation, target, innovation or correction.

## Reserved data and causal processing

Numerically open exactly `recording_2021_12_15_22H_41M.mat` and `recording_2021_12_15_22H_50M.mat`, in that order. Official TEST `22H_58M` stays closed and unsupported. No FIT or DEV raw recording is decoded during this study. Their already authenticated model coefficients and normalization arrays may be loaded.

The existing raw archive has SHA256 `9011509cf901dcb50a8de1e4a5a0fcf6cb8bed2f64945ba0fbbfb8e0b54886fe`. Extract only the two named members to a new input directory. This initial step may hash opaque bytes for registration but must not decode numerical values. Preserve the exact extraction command, archive/member hashes and original outcome. It does not count as a forecast observation. Do not extract the official test member.

Use the unchanged qualified `industrial_robot_data.load_recording` with explicit confirmation access. Validate raw field shapes, 90,881 timestamps at 250 Hz, finite selected measurements and recording-valid status. Position uses secondary encoders for joints 1-3 and motor encoders for joints 4-6; inputs are all six realized total measured torques. Exclude reference signals, feedback-only torque and measured velocities.

Filter each full recording independently with the same fourth-order 4 Hz Butterworth SOS filter and first-sample steady-state initialization. Decimate raw indices 0, 25, 50, ... to 3,636 samples. No zero-phase filter, cross-file filter state, per-window filter restart or new preprocessing fit. Normalize using the exact inherited FIT-only float64 means and scales.

For each file use the same 22 starts `64 + 160*k`, `k=0,...,21`. Each context has 32 observations. For start `s`, observed positions and torques are `[s:s+32]`, future torque is `[s+31:s+159]`, and the 128-step target is `[s+32:s+160]`. Thus the first future torque predicts the first unseen position. H128 is primary; H64 and per-joint physical errors are descriptive. Keep all 44 windows.

This is conditional prediction using realized future torque. It is not a verified issued-command control task. The two recordings share the same robot and day. Their windows and the three initialization seeds are not independent environments or independent dataset replications.

## Frozen decision and measured costs

Use the same five development criteria, with no new selection on confirmation outcomes. For each model, average the three individual H128 standardized RMSEs within each file, then average the two file means equally. Deterministic references have one forecast per file. All comparisons require complete finite cases rather than silently dropping difficult windows or failed seeds.

1. All three primary recipes are complete and finite.
2. Temporal affine improves the equal-file mean by at least 5% versus both last-two and local affine. Each baseline must have positive error; a zero-error tie is not a relative gain.
3. On neither file is temporal more than 2% worse than the better local control.
4. Its complete-request median latency is at most 1.25 times last-two.
5. The declared comparison set is complete, and no other model is no worse in equal-file error, full-request latency and persistent numeric bytes, with a strict improvement on at least one axis.

All five must pass for `CONFIRMED_HISTORY_INITIALIZATION`; otherwise report `DO_NOT_CONFIRM_HISTORY_INITIALIZATION`. The thresholds are practical continuation choices, not significance tests. A scientific failure can have a successful execution and agreeing audit. Do not change thresholds or discard controls after seeing outcomes.

Retime all 28 selected model/reference instances on the first registered window of the first confirmation file. Use batch 1, three warmups and twenty repetitions per instance. Charge normalization, casting, prefix conditioning, validation, transition/operator preparation, all 128 forecast steps, denormalization, finite checks and the same deadline callback. Keep raw durations; family latency is the median of the three fit medians. Set all five numerical thread environment variables and PyTorch CPU threads to 1. Record host/runtime details and retain the fixed model order. This shared-host measurement is not an isolated hardware speed claim.

Persistent bytes include all parameters, buffers, explicit recurrent state and normalization vectors. Report request arrays separately; temporary workspace, Python object overhead and model/disk loading are excluded. Retain historical training costs as historical. Confirmation adds zero training updates. No diagnostic permutation experiment is repeated here; the earlier order diagnostic remains separately reported.

## Admission, failures and evidence

Before the first array or MAT decode, authenticate the parent's successful original process, agreeing audit, exact source registration, complete parent evidence and development-selected mapping. Bind every selected checkpoint, normalization and reference payload to the parent's original manifest. Pin the new runner, tests, auditor, protocol and launcher, their relevant inherited source files, the opaque extraction receipt and exactly two raw member descriptors. Qualification uses fabricated inputs only. Commit the resulting registration before execution; save it and all source snapshots in the new evidence directory.

The original run has a suspend-aware 900-second whole-study limit and an external 960-second launcher limit, including preparation, evaluation, timing and preservation. There is one attempt, no automatic retry. Preserve the first failure and partial evidence on deadline, malformed input, source drift or infrastructure failure. A numerical model/reference prediction failure stays in its scheduled error rows and makes the affected completeness conditions fail. Do not repair nonfinite values or treat missing timing as zero. Structural or data-schema errors terminate the run.

Retain all 24 selected checkpoint copies, exact inherited reference/normalization copies, model identities, parameter/state checks before and after inference, all 56 forecast attempts and 112 H64/H128 score rows, all 28 planned timing cases including explicit failures, runtime, original process records and hashes. Successful complete execution has 56 prediction banks. Preserve both target-window files locally for audit with hashes; do not redistribute raw measurements or measured target payloads. Dataset terms remain separate from the repository license.

Independently reconstruct preprocessing, windows, metrics and all five decisions after original execution closes. Replay the qualified model implementation and independently reconstruct deterministic reference forecasts. This audit does not independently implement each neural recurrence or replay training gradients. Report those boundaries, zero optimizer updates and zero official TEST access. Preserve original audit and qualification attempts; an audit mismatch cannot be hidden by rerunning the scientific experiment.

## Continuation

A pass supports continued mechanism research and a new independent environment. It does not establish biological wiring, Bayesian state estimation, calibration, physical-state identification or ICLR-ready novelty. A failure rejects this fixed recipe's confirmation claim under the registered rule. These two recordings become exposed after this attempt and cannot serve as fresh confirmation for later tuning. Official TEST remains closed under either outcome. Future architecture changes require their own development comparison and new unexposed evidence.
