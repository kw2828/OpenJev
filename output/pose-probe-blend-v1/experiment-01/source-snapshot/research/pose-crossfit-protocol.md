# Parent-excluded selector training

This development screen tests a training failure exposed by pose-coordination-v1:
the frozen GRU was better on its own training windows, while its ordering against
the recency-plus-Huber expert changed on the two exposed evaluation archives.
The previous recurrent selector learned almost always to choose the GRU.
Neither this diagnosis nor the present experiment establishes a new architecture.

## Paired training design

Use the original 30 training trajectories, with 24 windows each. Assign each
whole trajectory to group source_id modulo 3. Expert pair k trains on the 20
parents outside group k, or 480 windows. No window from an excluded parent
enters that pair's optimizer or preprocessing statistics.

Each pair contains the original learned-feature pose-adaptation model and the
original body-frame GRU. Keep their architecture, physical loss, CV1 rollout,
Adam learning rate .001 and gradient norm cap 1 unchanged. Train for 46 epochs
of 15 batches of 32, exactly 690 updates. Apply decay_huber3 support weighting
only when producing the fast expert's prediction cache, as in the earlier study.
Learned parameters start identically across folds within architecture and seed;
normalization buffers legitimately differ.

Compute each fold's action mean and population standard deviation over its 480
eligible windows and all 56 action blocks, independently for each of 40 ordered
torque coordinates. Use NumPy float64 reductions, a 1e-5 scale floor, then
float32 arrays. Recenter and rescale the prepared actions with these statistics.
This cancels the earlier global affine normalization up to float32 rounding.
Compute the four motion RMS scales only from eligible poses. Selector tokens
keep the original full-training motion scales, since the selector uses all 30
training parents. Tokens expose only 32 context poses and 31 completed actions.

For each seed 1101, 1202 and 1303, each of three expert pairs predicts all 720
training windows. Save these full caches before assembling two matched inputs:

- OOF: a row from parent group g uses expert pair g, which excluded that parent.
- IS: the same row uses pair (g+1) modulo 3, which included that parent.

Each pair supplies 240 rows per regime. Both regimes retain all 720 rows, their
ordering, targets and selector tokens. The cyclic comparison balances expert
usage but changes each case's training-parent composition. It does not isolate
prior exposure as the only causal difference.

Train unchanged summary and recurrent selectors in each regime. The summary
model has 3,302 parameters and the recurrent model 3,250. Paired IS/OOF fits use
identical initial weights and batch permutations. Each receives 30 epochs of
23 batches, or 690 updates. Use final checkpoints only. There are 18 expert fits
and 12 selector fits: 30 Adam fits and 20,700 optimizer updates in total.

## Stronger constant controls

Fit a two-coefficient constant mixture separately to IS, OOF and the original
full-30-parent training prediction caches for each seed, giving nine fits.
Position uses the exact clipped least-squares coefficient in [0,1]; equal
experts use .5. Rotation is not assumed globally convex. Project the three
cached rotation arrays to SO(3) with float64 SVD, and search alpha in [0,1]
with a deterministic binary interval algorithm. For center c and radius r,
the triangle inequality gives the objective lower bound
mean(max(0, distance_i(c) - speed_i*r)^2), where speed_i is the distance
between that example's two projected expert rotations.

Evaluate both endpoints, retain every interval and objective evaluation, and
break exact objective ties by the smallest evaluated alpha. Use the source's
explicit 1e-12 numerical angle and objective guards. The tolerance is 1e-7 rad²
with at most 4,095 objective evaluations per fit. This is a guarded numerical
certificate, not an outward-rounded interval-arithmetic proof. It concerns the
projected float64 objective, not the unchanged float32 deployment blend.
Report their discrepancy rather than claiming exact production optimality.

If any search exhausts its cap, retain its best observed coefficient and trace,
continue the scheduled diagnostic fits and evaluations, and prohibit continuation
success. Do not increase the cap, retry, or remove that control. All nine
certificates are an additional requirement, separate from the performance gate.

## Deployment, evaluation and accounting

Deploy every selector with the original six full-30-parent frozen expert
checkpoints. Thus both arms share a 20-parent-to-30-parent expert shift. Neither
arm refits deployment experts. Each expert privately forecasts all 25 steps;
blend outputs without feeding them back. Alpha is fixed over the horizon.

Evaluate fast, slow, is_constant, is_summary, is_recurrent, oof_constant,
oof_summary, oof_recurrent and full_constant, for three paired seeds and both
exposed panels: 54 new rows. Primary is oof_recurrent. Retain all 26 previous
canonical configurations and seven new ones, giving 33 configurations and 32
controls. The 12 fresh fast/slow rows are replay checks and replace their old
timings, not additional independent evidence. The combined archive contains
174 canonical rows after deduplication.

For each control, endpoint and panel require at least 10% lower pooled RMSE,
all three paired seed MSE values nonworse, at least eight of ten parents
nonworse, and strictly lower MSE after each of ten leave-one-parent-out removals.
Also require median complete forecast latency at most 1.5 times the freshly
timed GRU. Retain all 17 conjunction groups and 1,921 elementary comparisons.
All must pass, along with the nine optimization certificates. A failed earlier
study remains failed. No selection, sweeps, retries, replacements or exclusions.

Measure CPU one thread, three warmups and 20 individual windows per row.
Charge both experts, context processing, selector and blending when used.
Loading, normalization, metric calculation and artifact I/O are excluded from
forecast timing. Report all new fitting, cache and constant-search costs, plus
total execution wall time, separately. Historical expert training is inherited.
The selectors do not save expert computation.

## Independent evidence and limits

Bind 24 source files, prepared data, previous experiment/report and deployment
weights before training. Save 288 execution files, including all initial/final
weights, orders, losses, fold statistics, expert caches, regime caches, constant
traces, targets, predictions, selector weights and timing samples.
The saved-output auditor makes no neural calls and does not rerun optimization.
It independently reconstructs statistics, cache assignments, interval bounds,
physical errors and blends, then checks all continuation comparisons.

Action statistics must match their NumPy reductions exactly after float32
conversion. Motion-scale tolerance is rtol1e-4/atol1e-7; selector-token tolerance
is rtol1e-4/atol1e-4; blend tolerance is rtol2e-6/atol2e-6 with exact hard
endpoints; prior expert replay tolerance is rtol1e-6/atol2e-7. Proper rotations
and finite values remain required. Invalid evidence is not a performance result.

Both evaluation archives have repeatedly informed development. This is not fresh
confirmation, a real-robot control result, calibrated uncertainty, a new RL
algorithm, connectome efficacy or an ICLR-ready contribution. Applied torques
are recorded rather than authenticated issued commands, and Euler conventions
remain inferred. Raw/prepared data, training tokens and full prediction/target
arrays stay local under unresolved upstream licensing. Publish our code, weights,
numerical errors, optimization traces and receipts.

Cross-validated stacking is established prior art, including
[Wolpert's Stacked Generalization](https://doi.org/10.1016/S0893-6080(05)80023-1)
and [Ting and Witten's Issues in Stacked Generalization](https://arxiv.org/abs/1105.5466).
The contribution sought here is evidence about the failure mechanism and useful
forecasting behavior; this training correction alone is not architectural novelty.
