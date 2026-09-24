# Readout learning versus recurrent weight adaptation

This is a separate, prospective training ablation on **previously exposed
DEV paths**. It follows the negative Bayesian residual comparison. It is not
held-out confirmation, a new environment result, or a novelty claim. The two
closed screens stay closed; their reserved confirmation and old TEST are not
admitted. OpenJev remains the project name.

## Question and fixed comparison

Does ordinary joint continuation improve because of its action readout, its
prediction feedback, or adaptation of its recurrent weights?

| Arm | Trainable parameters | What can change |
| --- | ---: | --- |
| Action-residual readout only | 116 | Final action and shadow-prior readout |
| Both readouts | 232 | Those outputs plus the base prediction used in innovation feedback |
| Full joint | 6,112 | Both readouts and the GRU weights |

All arms start from the exact same-seed 80-epoch parent checkpoints, seeds
309000001, 309000002 and 309000003. Rerun all nine 40-epoch continuations. Do not
substitute historical full-joint checkpoints or select intermediate epochs.
Also evaluate each unchanged parent, yielding 12 views.

The base readout affects the error fed into the GRU. Freezing GRU weights does
not freeze its hidden trajectories when that readout changes. Preserve gradients
through the frozen GRU in the both-readout arm. The residual-only head does not
feed the innovation path. Test its hidden trajectory invariance before and
after a readout update using the same parameter flags. The qualified underlying
kernel can change floating-point rounding with parameter flags, so do not assume
bitwise cross-arm equality simply from identical weights.

## Data and training

Use only the original 54 TRAIN census paths and three pretrained checkpoints,
authenticated through the closed query-memory study's metadata lineage. TRAIN
file SHA256 is `14af1000e5c3c9a3a2b096e5e9745029e8c7e33e8b8dc005e7c0f041896b1b02`.
Use only the already-collected 18 residual-screen DEV paths, authenticated by
the completed native-runtime bridge. DEV file SHA256 is
`2bc1150574ac74e66d3823a147f768b33361c4178202e1fd244c184579d06883`.
The plan binds complete identities, source hashes, file descriptors and runtime.
No new teacher, simulator, Astra, confirmation or TEST calls occur.

Keep P4 observations, legal-centered nonquery MSE plus equally weighted all-four
prequery MSE, each in /64 score units. Use Adam 0.003, gradient clip 5, batch six
complete episodes, and 32-step carry detachment. A fresh optimizer and a restarted
PCG64(fit_seed) permutation stream are used for each branch. Each fit receives
40 epochs, 360 updates and 2,160 episode exposures. Total: 3,240 updates and
19,440 exposures. Freeze unused parameter bytes and retain exact optimizer masks.

All nine final checkpoints and their training journals must close before DEV
is decoded. Evaluation uses identical canonical frozen inference, batch one,
chunks of 32, and complete episode denominators for all 12 views. It is not a
bitwise comparison to the earlier study's batch-six evaluation. Retain every
view, fit seed, setting and collector regardless of outcome.

## Interpretation fixed before gradients

Full joint is the fixed mechanistic candidate. The primary diagnostic is later
case-weighted raw teacher-cost gap, lower is better. Continue the recurrent-weight
hypothesis only if full joint improves this gap by at least 5% against **each**
readout arm and the unchanged parent, in every one of the three paired fit seeds
and two settings. Require no full-episode gap regression against those same
comparators in each cell. Zero comparator gap requires an exact zero candidate
gap and provides no evidence of a positive relative gain; it cannot pass the
5% improvement condition. Preserve all mixed or failed cells.

Report readout gaps within 5% of full joint as descriptive proximity only, not
statistical equivalence or noninferiority. Report case-weighted agreement,
centered MSE, prequery MSE, complete support, all setting/collector summaries,
actual training time and inference time. Equal updates are not equal compute.
Joint beating readouts but losing to the parent is not a utility improvement.
No p-values or independent-sample claims are attached to three repeated fits.

## Qualification, resource limits and audit

Before registration, pass fabricated mask, gradient, state, update, source
parity and independent metric tests; lint new files; verify a metadata-only
handoff with the exact current runtime. Fabricated capacity uses the historical
TRAIN episode lengths from authenticated JSON metadata, nine batches per arm,
and no empirical arrays or checkpoints. Admission requires the projected cost
`2 * 40 * 3 * (sum of three one-epoch arm times) + 120` at most 4,050 seconds.
The new supervisor has a fixed 5,400-second train/evaluate cap, 4 GiB RSS and
512 MiB output. Independent saved-output audit has 600 seconds, 2 GiB RSS and
128 MiB output. Both use the original suspend-aware supervisor and process-group
cleanup. Register all source/data/runtime hashes before the first gradient.

The audit reconstructs histories, parent/frozen tensor witnesses, training
orders and counters, complete metrics and the diagnostic from saved outputs.
It performs no model, optimizer, teacher or simulator calls. Actual optimization,
causal model inference and timings remain source-tested producer evidence.
A result is official only after both original processes close successfully.
Any technical failure ends this attempt with its partial artifacts; no retry,
replacement seed, cap extension or post-result threshold adjustment rescues it.
A passing diagnostic permits a new proposal, not automatic held-out access.
