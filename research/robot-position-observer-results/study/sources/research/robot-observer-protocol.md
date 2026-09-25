# Frozen-dynamics observation correction: exposed development protocol

Version 1, prospective before fitting. The [earlier history recipe](robot-history-confirmation-results.md) failed reserved-recording confirmation. This experiment tests a more specific state-estimation mechanism. It does not reopen that failed claim or treat the four exposed recordings as new confirmation.

## Hypothesis and controlled models

For each seed 8101, 8102 and 8103, copy the exact 590-parameter dense transition from the completed last-two model at its previously selected learning rate .001. Freeze every transition parameter, including the gate, scale and input maps. The three new learned initializers use that identical per-seed transition:

- Local affine: the existing 372-parameter local feature head, initialized to zero.
- Temporal affine: the existing 372-parameter history feature head, initialized to zero.
- Learned observer: a constant 12-by-6 correction gain, 72 parameters, initialized to the vertical concatenation `[I; I]`.

Both affine heads retain their established feature definitions. For the observer, initialize at observed index 1 as `concat(q1, q1-q0)`. At observed indices 2 through 31, predict `prior = f(state, u[t-1])` and correct `state = prior + K @ (q[t] - prior[:6])`. Prepare the transition once for these thirty steps. Gradients may traverse the frozen transition to train the gain; no graph detach, prefix cache or transition update is allowed. The forecast then uses `u31` to predict `q32` and receives no later observations.

Include three fixed controls per seed: unchanged last-two initialization, fixed `[I; I]` gain, and zero gain. Fixed and zero gains are explicit 72-value buffers and count toward storage. Last-two forecasts must equal the saved parent model exactly on the first paired FIT batch. Learned/fixed observer forecasts start identically; the two zero affine heads start identically to last-two. These are two separate initial-function pairings, not one common initialization across all methods.

Retain all seven other previously selected neural families, without fitting: jointly trained local affine and temporal affine (renamed `joint_local_affine` and `joint_temporal_affine`), bounded dense, unbounded dense, GRU32, legacy instant and GRU10. Their original selected rates and three seed checkpoints stay fixed. Keep both causal ridge penalties, frozen linear AR2 and persistence. This supplies 17 selected families, rather than permitting a win only against weaker refitted heads.

## Training, data and selection

Run exactly eighteen new fits: three learned initializers, three seeds and rates .001/.003. Every fit uses the original 4,096 paired batches of size 16 and a 128-step forecast MSE. Keep Adam betas .9/.999, epsilon 1e-8 and gradient-norm cap 1.0. No extra fit, early stopping on development scores, restart, ensemble or seed selection. Count the actual complete training time; equal batches and updates do not imply equal computation. Adam may receive frozen parameters in its parameter list, but those must never receive gradients, optimizer state or changes. Preserve initial/final arrays, initializer-only Adam arrays, every loss/gradient trace and the original outcome.

Numerically read only the seven already processed FIT recordings for training. Inherit normalization, ridge and linear coefficients exactly, with no refit. Reuse the original paired batch arrays and verify them against their deterministic generator. Finish all eighteen attempts before reading any current evaluation arrays.

The original DEV files `21H_54M` and `22H_10M` select one rate per learned family by the sum of H128 SSE over all three seeds and both files. Select only a complete finite recipe; the lower rate wins an exact tie. The former confirmation files `22H_41M` and `22H_50M` do not select rates. They are now exposed development evaluation data. Report both original DEV and former-confirmation file results separately as well as their equal-four-file mean.

All recordings are from 2021-12-15 on the same robot. Use unchanged processed arrays, fourth-order causal 4 Hz filtering at 250 Hz, decimation by 25, inherited FIT scales, 22 starts `64 + 160*k`, context32 and horizon128. For start `s`, context is `[s:s+32]`, future measured torque `[s+31:s+159]`, and targets `[s+32:s+160]`. H128 is primary; H64 and per-joint physical errors remain visible. Realized measured future torque makes this conditional forecasting, not command-driven control.

No raw MAT numerical decoding occurs. Existing closed-evidence admission may hash the original archive/MAT bytes opaquely. Official TEST `22H_58M` is neither an allowed numerical input nor an extracted input. The two previously reserved recordings cannot become fresh confirmation again.

## Fixed development continuation rule

Average the three individual seed H128 RMSEs within each file, then average the four file means equally. Deterministic references contribute one value per file. All five criteria must pass:

1. Every selected forecast at both horizons and all 43 planned selected timing slots are complete and finite. A learned family with no eligible rate fails completeness and retains three explicit unavailable timing slots.
2. Learned observer has at least 5% lower equal-four-file mean error than the best of **all sixteen** other selected families. The comparison error must be positive. This includes the stronger previously trained models and references.
3. On no file is the observer more than 2% worse than that file's best of last-two, refitted local affine, refitted temporal affine and fixed-gain observer.
4. Full-request family median latency is at most 1.5 times last-two. This prospective allowance accounts for the thirty additional recurrent prefix updates; it does not alter the earlier confirmation rule.
5. The complete declared frontier contains no control with error, latency and persistent numeric bytes all no greater than the observer, with a strict improvement on at least one axis.

Report `OBSERVER_DEVELOPMENT_PASS` only for five passing criteria; otherwise `OBSERVER_DEVELOPMENT_FAIL`. These are practical continuation margins, not significance tests. Display all paired seeds and the distinction between refitted and historical models. A pass supports a new-environment experiment, not novelty or independent generalization.

Time all 43 selected instances on the first window of original DEV1 with three warmups and twenty repetitions. Include normalization, casting, all prefix work and preparation, all forecast steps, denormalization, finite checks and the same native deadline callback. Fix all five numerical thread variables and Torch CPU threads to one. No frozen transition sharing across requests or models. Retain raw timings. Family latency is the median of its three per-fit medians. Persistent bytes include frozen/trainable parameters, fixed gain buffers, twelve-state vectors and normalizers; report other request arrays separately and disclose that temporary workspace and Python overhead are not measured.

## Execution and evidence

Authenticate the closed history and confirmation studies, their original processes and agreeing audits, all 45 current source files, the launcher, fabricated qualification and exact 45 inherited payload descriptors before any new numerical decode. Freeze these in a committed registration before fitting. The active source roster also includes the independent auditor. Keep original source snapshots and all attempts.

There are 18 fresh fit attempts, nine fixed models, 21 cached models and four references, yielding 52 forecasts per file, 208 forecast attempts and 416 H64/H128 rows. All 48 neural final checkpoint files and all reference coefficients are retained. Record frozen transition hashes before/after fitting and complete model hashes before/after evaluation. Failed numerical attempts remain scheduled failed rows. No repair, replacement or retry. Structural/source/data defects terminate the study with partial evidence.

Each fresh fit has a suspend-aware 1,800-second cap. The whole original study has a suspend-aware 14,400-second cap and external 14,460-second cap, including evidence preservation. Preserve original terminal records. A timeout is a failure, not permission to restart. Treat a failed selection as unavailable, never zero latency; retain all 43 cost slots.

After the original process closes, independently reconstruct metrics, DEV-only selection and all five criteria, verify frozen cells and initializer-only optimizer states, and replay qualified forecasts. This is not an independent implementation of each neural recurrence or replay of all training gradients. No new timing, fitting or raw MAT decode is allowed in the audit. Keep all measured target arrays local under hashes. Publish derived forecasts, checkpoints, traces, timings, source/qualification/terminal evidence and the complete outcome under the dataset's separate terms.

State encoders and learned observers have established [prior art](https://proceedings.mlr.press/v144/beintema21a.html), including [KalmanNet](https://arxiv.org/abs/2107.10043). This constant-gain model is a mechanistic baseline. Biological wiring, Bayesian uncertainty, calibrated predictions, stability, robotic control and architectural novelty require separate evidence.
