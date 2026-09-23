# Query-written history memory: fresh fixed-path comparison

Status: prospective implementation protocol. No scientific collection or fitting
is admitted until the corresponding source-bound plan is published. The previous
protected-readout experiment remains closed at FAIL 15/29. This is a distinct
online-memory hypothesis, not a retry of that candidate.

## Question and limits

Can supervised error-driven memory improve a recurrent predictor's action-score
forecasts beyond simple last-error correction and ordinary recurrent training?
Does mixing recent hidden features help relative to instantaneous features?

The primary outcome is the teacher's legal score gap on fixed paths after the
first correction opportunity. A full-path gap guard prevents improvement in that
scope from hiding worse overall choices. Neither is realized game return. A pass
admits a fresh autonomous, total-compute comparison. Architecture novelty,
biological learning and a second-environment result require further evidence.

## Fresh collection

Use the unchanged released OTTO environment, teacher weights, native/public
adapter, 31 public features and three collectors: analytic, neural and
period-four held teacher. Each originating case has all three collectors in
global-case-index rotation. Initial hit is `1 + case % 3`; horizon is 2,188.

| Split | Setting | Seed range | Cases | Paths |
| --- | --- | --- | ---: | ---: |
| TRAIN | lambda3 | 303000001-303000009 | 9 | 27 |
| TRAIN | lambda4 | 304000001-304000009 | 9 | 27 |
| DEV | lambda3 | 305000001-305000003 | 3 | 9 |
| DEV | lambda4 | 306000001-306000003 | 3 | 9 |
| TEST | lambda3 | 307000001-307000006 | 6 | 18 |
| TEST | lambda4 | 308000001-308000006 | 6 | 18 |

All 108 paths receive a complete teacher-score census, exactly one teacher call
per recorded state. An annotation never changes an already selected action or
refreshes the held collector's score cache. There is no deferred public replay,
window selection or sampling seed. Preserve paired cases, every final public
update and the original episode durability boundaries.

Collection retains its original period-four age feature and correction metadata.
A separate projector sets the forecasting age to `(absolute_step % P)/2188`
for P=4 or P=8. Only actual P-scheduled answers enter the predictor or memory.
All other scores remain targets. The other 30 features stay unchanged. Changing
the observation schedule on these same paths does not simulate an autonomous
period-eight controller or establish saved physical teacher calls. Full census
annotation costs remain reported.

## Predictors, memory and controls

Fit seeds are 309000001, 309000002 and 309000003. For each seed, train one fresh
ordinary scheduled GRU for 80 epochs with period-four inputs. Its eight tensors
contain 6,112 parameters, including the 116-parameter static action residual.
All eight tensors are eligible for the common pretraining objective.

Fork the final checkpoint into five 40-epoch branches:

| Branch | Slow parameters | Additional optimized parameters | Runtime correction |
| --- | --- | ---: | --- |
| joint_aux | All 6,112 trainable | 0 | None |
| instant_delta | Frozen | 224 | Normalized delta, instantaneous key |
| trace_delta | Frozen | 224 | Normalized delta, history-conditioned key |
| trace_additive | Frozen | 224 | Additive write, history-conditioned key |
| trace_scrambled | Frozen | 224 | Delta write, rotated past feature |

The candidate is `trace_delta`; its identity never changes after results.
Also evaluate the common `pretrained` checkpoint, its fixed `last_error` view,
and a no-write view of the trained `trace_delta` branch. There are 18 optimized
fits and 24 final evaluation views. No-write shares its parent projection and
training cost; it is an intervention, not an independent trained fit.

Each memory branch starts with the identical nonzero seeded, bias-free 28-to-8
linear projection. Its 224 parameters are the only optimizer parameters in
memory branches. No-memory and last-error controls do not construct or execute
an unused projection. Preserve the original slow model's required parameter
flags, but frozen slow forwards run entirely without gradients. Validate the
slow/fast carry joins before each call.

At each active key step, normalize the key with `norm.clamp_min(1e-6)`.
The trace is `0.75 * previous_trace + 0.25 * key`, then normalized for reading
and writing. Matrix decay is 1, step size is 0.25 and the write denominator is
`1e-6 + ||cue||^2`. Read before any write. A later query supplies the centered
residual `(answer - complete_slow_shadow_prior)/64`; the delta update subtracts
the current memory read from that residual. Additive writes omit this subtraction.
The complete shadow prior includes the static action residual. Exact query
outputs remain the supplied answer; the first query performs no error write.

Last-error memory overwrites with that normalized residual at each later query
and decays by 0.75 at every key step. The rotated-past control maintains the
ordinary chronological trace but rotates its past vector by
`1 + absolute_step % 7` before mixing the read/write cue. This changes the feature
representation, not temporal order. It does not identify temporal credit.
No clipping, replacement key or numerical recovery is permitted.

## Fixed objective and optimization

All optimized models use the same **full-forecast AUX** objective: legal-centered
nonquery MSE plus all-four centered prewrite query-forecast MSE, both in score/64
units and with coefficient one. Ordinary models use their complete shadow prior;
memory models add the prewrite correction to that shadow prior. This is explicitly
a new matched objective, not the old experiment's base-prior-only AUX recipe.
Teacher targets and frozen slow forecasts are detached. Projection gradients may
flow through cues, traces and writes within a chunk. Initial zero memory does not
imply a zero-initialized projection: that would prevent useful initial gradients.

Use Adam at 0.003, weight decay zero, default betas/epsilon, and global effective
trainable-gradient norm clip 5. Each branch uses a fresh optimizer and restarts
NumPy PCG64 at its fit seed, giving identical 40-epoch episode orders. CPU float32,
deterministic Torch algorithms, one intra-operation and one inter-operation thread.
No accelerator, simulator or teacher call occurs in training.

Batch six complete episodes in chronological 32-step chunks. Keep all parameters
fixed until the whole batch finishes; detach slow and fast carry at chunk edges.
Then take one optimizer step. Updating the projection between chunks would mix
coordinate systems inside the carried matrix and is prohibited. Explicit zero
gradients for otherwise missing effective-parameter gradients preserve the fixed
Adam update schedule. Frozen parameters receive no optimizer state.

Each episode's nonquery weights sum to 1/54 when supported; its later-query prior
weights separately sum to 1/54 when supported. Unsupported episodes contribute
zero with the full denominator retained. Multiply batch loss by 54/actual batch
size. There are no sampled-window weights or realized-support renormalization.

There are nine batches per epoch: 720 pretraining updates and 360 per branch,
totaling 7,560 optimizer updates and 45,360 episode exposures. These counts are
not equal-compute claims. Record actual recurrent/projection/memory operations,
forward/backward/no-gradient chunks, setup, serialization, inference, optimizer
time and peak memory. This version executes ordinary slow forwards explicitly;
no unqualified feature cache or uncharged computation reuse is admitted.

## Frozen evaluation and one continuation rule

All final checkpoints, forks, optimizer counts and TRAIN predictions must be
durable before DEV arrays are decoded. No best-epoch, best-seed, family or
hyperparameter selection occurs. Run one DEV evaluation with P=4, then close
the training/DEV producer and its original supervisor. A separate saved-output
audit must establish the DEV gate before a separately supervised TEST evaluator
can decode TEST. Its admission must verify the original DEV-audit supervisor's
successful closure as well as the audited result. Otherwise preserve TEST as unused and terminate this study
without a replacement run. TEST uses the already frozen checkpoints and has no
optimizer. Its producer is followed by a separate saved-output audit.

For each of the two settings at DEV P=4, and then separately for all four
setting/period combinations at TEST P=4/P=8, require all of:

1. Genuine technical completion, finite preserved outputs and independent audit.
2. At least two supported originating cases on DEV, four on TEST, for the later
   nonquery scope.
3. Candidate mean later teacher-score gap is at most 90% of, and strictly below,
   the lowest control mean among `pretrained`, `last_error`, `instant_delta` and
   `joint_aux`. A zero best-control gap cannot be beaten and fails this condition.
4. Candidate mean full-nonquery gap is no higher than the lowest corresponding
   control mean.
5. Each of the three paired fit seeds has candidate later gap no higher than the
   lowest same-seed control gap.

Every condition is a conjunction, without tolerance, pooled rescue or post-result
exception. Aggregate equally over fit seeds and originating cases, retaining all
three collector paths and zero-support episodes. Full nonqueries exclude actual
queries; initial scope is steps 1 through P-1; later scope is nonqueries from
P+1 onward. Report every age, collector, case and setting. Agreement is secondary
and cannot substitute for the gap requirements. Additive, rotated-past and no-write
results are mechanism controls, not alternative candidates.

P4/P8 scopes differ. Also report common nonquery support `step % 4 != 0`, with
common initial steps below 8 and common later steps from 9, when comparing the
two schedules directly. The primary gate compares methods within each schedule,
not unlike row sets across schedules. Three seeds and these small case sets are
a mechanism screen, not a statistical or paper-level performance claim.

## Admission, bounds and preservation

Bind the scoped seed reservation, successful fabricated qualification, protocol,
sources, native checkpoint and runtime in the published collection plan before
any native call. The subsequent training plan must bind the collection's genuine
successful original supervisor closure before any training array is decoded.
New training and audit code need their own fabricated qualification and source
freeze; collection success alone never admits model training.

Collection: 7,200 suspend-aware seconds, 4 GiB RSS, 2 GiB output, 108 resets and
at most 236,304 native steps and teacher/TF value calls. No warmup or replay.
Training including canonical TRAIN rescoring and DEV evaluation: 21,600 seconds,
4 GiB RSS, 2 GiB output. A bounded fabricated capacity check must precede training
admission; historical GRU timing does not qualify this memory implementation.
Conditional TEST evaluation: 1,800 seconds, 4 GiB RSS, 2 GiB output, no fitting.
Each independent saved-output audit: 600 seconds, 2 GiB RSS, 256 MiB output, with
zero model, optimizer, teacher or simulator calls.

Use the qualified detached launcher and unchanged supervisor, exclusive paths,
source/runtime/input hashes, original launch/exit/deadline/reaping evidence and
an absent worker process group. No scientific retries, replacement seeds, cap
extensions or synthetic terminal records. Preserve failures and unstarted phases.
Report all methods and paid computation, including shared pretraining and the
gate itself. Autonomous quality-versus-total-compute performance remains a later
experiment even if every condition here passes.
