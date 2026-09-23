# Protected recurrent forecasts and decision adaptation

Status: implementation-stage protocol. Scientific collection and fitting are not yet admitted. Source hashes, the engineering receipt and a fresh seed reservation must be bound in a published collection plan before any collection begins. The subsequent training plan must bind that collection's genuine successful supervisor closure before any training array is decoded.

## Question and scope

Can an action-only residual improve decisions when its recurrent predictor is protected from adaptation? The previous fixed SPO+ recipe worsened later teacher-score gaps; its separate diagnostic leaves the original interrupted study ineligible. It supplies motivation, not a diagnosis of damaged memory or a reusable validation set. No earlier study is resumed or retried.

This is a fresh fixed-path teacher-imitation screen on ordinary GRU models. It is not an autonomous search result, a new RL algorithm, a biological-wiring comparison or an architecture-novelty claim. Both sensing settings occur in training; neither is an unseen shift. Passing permits a fresh confirmation and quality-versus-total-compute experiment, not a paper-level claim. An architectural contribution still requires strong ordinary-model controls, an unseen shift and a second environment.

## Fresh data and allocation

Use the unchanged released OTTO native environment, teacher checkpoint, public-feature builder, action rule and three collectors: analytic, neural and period-four held teacher. Each originating case has all three collectors, in global-case-index rotation. Initial hit is `1 + case % 3`; horizon is 2,188 steps. All paired trajectories retain their case identity.

| Split | Sensing setting | Seeds | Cases | Paths |
| --- | --- | --- | ---: | ---: |
| TRAIN | 3 | 297000001 through 297000009 | 9 | 27 |
| TRAIN | 4 | 298000001 through 298000009 | 9 | 27 |
| VALID | 3 | 299000001 through 299000006 | 6 | 18 |
| VALID | 4 | 300000001 through 300000006 | 6 | 18 |

There are 54 TRAIN and 36 VALID paths. TRAIN window-selection seeds are 302000001 through 302000054 in TRAIN path order. Select `min(8, ceil(length/4))` windows uniformly without replacement after a complete path. Annotate their union with every genuine period-four query row. Unselected query labels support recurrent inputs and prior supervision, not additional sampled nonquery targets. VALID annotations are a census.

Only the established 31 public features and genuine scheduled query scores enter the model. Skipped teacher labels never enter recurrent state. Query times are absolute steps 0, 4, 8, and so on; virtual last-query and age features retain the same convention. An annotation cannot change an already chosen action or refresh held state. Preserve complete compressed work journals and episode durability boundaries.

## Models, forks and output units

Fit seeds are 301000001, 301000002 and 301000003. Each seed produces one ordinary `innovation_gru` AUX pretrain with 5,996 parameters, then four adaptation branches. Their names are `pretrained`, `frozen_aux`, `frozen_spo`, `joint_aux` and `joint_spo`, giving 15 final models including references.

Every branch strictly copies all six backbone tensors from its own final pretrained checkpoint and starts the same new linear residual at zero. The four branches have 6,112 parameters: the original 5,996 plus 116 action-readout parameters. Frozen branches permit gradients only into the residual; joint branches permit all parameters to train. No optimizer state transfers from pretraining or another branch.

The qualified raw-unit adapter remains unchanged. A separate normalized-unit adapter supplies `64 * (W h + b)` to its existing four-coordinate centering operation. Thus the action residual is mathematically `64 * centered(W h + b)`, with float32 scale-before-centering order. This matches the backbone readout's normalized units and allows the same fixed learning rate. It changes active nonquery action scores only. It never changes the predictor's returned base scores, prior, correction signal or recurrent carry by direct feedback.

## Fixed optimization

Pretrain for 80 epochs with AUX. Adapt each branch for 40 epochs with a fresh Adam optimizer, learning rate 0.003, weight decay zero, default Adam betas and epsilon, and global trainable-gradient norm clip 5. CPU float32, deterministic Torch algorithms, one intra-operation and one inter-operation thread. No accelerator, teacher call or simulator call occurs in training.

Use six complete episodes per batch and chronological 32-step chunks. Keep parameters fixed across each episode batch, detach carry at each chunk boundary, accumulate gradients, then take one Adam step for the whole batch. Paired branches use the identical stage-two episode order and initial pretrained tensors. All epochs and final checkpoints are retained in the journal; there is no best-epoch or best-seed selection.

Each branch restarts NumPy PCG64 at its fit seed, giving the four branches identical 40-epoch orders. A trainable parameter that receives no gradient in a complete batch is assigned an explicit zero gradient before the scheduled Adam step. This preserves the declared update count and permits existing Adam momentum to evolve; it does not create a new data gradient. Frozen parameters receive neither gradients nor optimizer state. Record the trainable parameter names and optimizer step counters.

There are nine batches per epoch: 720 updates per pretrain and 360 per adaptation branch, totaling 6,480 updates and 38,880 episode exposures. Record actual forward, differentiable, backward, no-gradient and skipped-backward chunks. In frozen branches a prior-only chunk can have no differentiable loss, so it must not trigger an unconditional backward call. Matching data and optimizer-update schedules does not establish equal compute.

AUX is the unchanged eligible-centered nonquery MSE plus all-four prior-query MSE. Nonquery scores and targets are divided by 64; prior scores and targets use the same scale. The sampled nonquery weight is `W/(k * 54 * M)`, where W is the full window count, k the selected count and M the episode's complete eligible nonquery-row count. Prior-query weight is `1/(54 * K)`, where K counts later query rows. Multiply each batch loss by `54 / actual_batch_episode_count`. No realized-sample renormalization occurs; zero-support terms contribute zero.

Nonquery MSE and SPO+ use the deployed action scores. Prior MSE uses the unmodified base prior. Both prior and SPO coefficients are one. Frozen prior MSE remains in the reported objective but cannot update frozen parameters. SPO+ is added only for the two `_spo` branches. It uses exact legal teacher minima, a uniform reference over tied minima, and first-max subgradients. The established deployment rule remains float32 legal near-minimum selection at tolerance `1e-10`; no claim transfers the exact-oracle theory automatically to this task.

## Evaluation and preservation

Save every final checkpoint, final TRAIN prediction and weighted loss. All 15 checkpoints must be durable and their complete barrier recorded before VALID decoding. No tuning, stopping, selection or replacement seed may use VALID.

All final TRAIN rescoring and VALID inference use canonical frozen-backbone evaluation under `torch.no_grad()`. Pretrained references have six `requires_grad=False` backbone tensors. Adaptation models use fresh frozen-mode clones with those same backbone flags; their residual retains the immutable adapter's required `requires_grad=True` flags, but the surrounding no-gradient context prevents evaluation graphs. Record construction and copying time. Preserve evaluation metadata and strict full-state load checks.

Save action predictions, base predictions, priors and prior masks for every family and seed. Frozen branches must preserve pretrained backbone tensors bitwise. Their canonical base predictions and priors must equal the same-seed pretrained results bitwise on the same TRAIN and VALID paths. Also retain query/padding invariants, the checkpoint/fork identities and every failed attempt. Tiny differences caused by changing parameter flags in the unchanged reference are engineering behavior; they cannot be presented as an adaptation mechanism.

Use the established complete-episode metric denominators: average eligible rows within each path, then average over every declared path including zero contributions from unsupported paths. Report full nonquery, initial steps 1-3 and later nonquery absolute steps at or after 5, including age, collector, setting, case and seed scopes. The primary gap is the teacher's legal score difference, not realized environment regret.

## One 29-condition development gate

The eleven common conditions are genuine technical completion, and for each setting: at least four supported originating cases for initial decisions; at least four for each later age 1, 2 and 3; and positive later gap for the held-score reference.

For each setting, `frozen_spo` must satisfy nine more conditions:

- Mean later gap must be at most 90% of and strictly below each of `frozen_aux`, `joint_aux`, `joint_spo` and `pretrained`: four separate conditions.
- Mean full agreement and mean initial agreement must each be no lower than the largest corresponding control mean: two conditions.
- Each paired seed's later gap must be no greater than `frozen_aux` for that same seed: three conditions.

All 29 conditions are required, without numerical tolerance or post-result exceptions. All three fit seeds contribute equally. The technical condition remains false in the training producer and can only be qualified through complete independent saved-output checks plus genuine successful original phase closures. Report negative and incomplete outcomes separately. No selected scope or agreement improvement substitutes for the primary comparisons.

## Bounds and independent audit

Collection cap: 7,200 suspend-aware seconds, 4 GiB RSS, 2 GiB output, 90 resets and at most 196,920 native steps. Teacher and TensorFlow value-call cap is 138,708. No warmup or extra collection occurs.

Training cap: 14,400 suspend-aware seconds, 4 GiB RSS and 2 GiB output. Historical capacity evidence covers the unchanged backbone only; it does not measure this residual, two-stage schedule or total runtime. Actual work and costs are recorded. Saved-output audit cap: 300 seconds, 2 GiB RSS and 256 MiB output, with no model, optimizer, teacher or simulator calls.

Use the qualified detached launcher and unchanged supervisor. Exclusive phase paths, source/runtime/input hashes, original launch identity, exit status, elapsed bounds, reaping and absent worker process group are required. A missing original terminal record never authorizes a synthetic replacement. No scientific retries, continuation after terminal failure, replacement seeds or cap extensions are allowed.

The independent reader reconstructs targets and importance weights, independently evaluates scalar losses and metrics, checks all forks and stage exposure, verifies the frozen-backbone invariants and the complete VALID barrier, and independently recomputes all 29 conditions. Tests use fabricated data before the scientific freeze. Presentation may later read the closed saved results, without fitting or changing the acceptance rule.
