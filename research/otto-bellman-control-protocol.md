# Proposed controlled continuation with observation-backup targets

**Protocol fixed in source; results pending.** Each execution requires a prospectively published machine-readable plan binding its exact source, input and runtime identities. The full study additionally requires successful completion of the separate TRAIN-only qualification and its original supervised process. Writing this protocol performed no scientific training, target generation, model inference or native episode. All previous failures, including the scalar-return study's **0/54** decision, remain unchanged.

## Question and scope

Does delayed observation-backup learning produce competent autonomous search and improve on continued Monte Carlo fitting from the same existing ordinary MLP? This is a learning-procedure control with full exact public belief, not a new architecture or compact-memory experiment. The first sixteen saved parity prefixes motivated a consistency question; they do not establish its cause or determine the new evaluation cases.

Use the fixed width-eight biased ReLU `mlp8`, 11,028 features, stored float32 `c0`, known kernels and explicit sixteen-branch deployed calculation from [the completed scalar-return protocol](otto-return-value-protocol.md). Neither the architecture nor the public filter changes. The [learning design](otto-bellman-control-design.md) supplies the objective rationale. The qualified [target primitive](../src/openjev/research/otto_bellman_targets.py) constructs a numerical backup; it is not evidence of convergence or efficacy.

## Authenticated inputs and six continuation fits

Bind the completed `otto-return-value-v1` plan SHA `92df71d5e20ca48d8485bfa0a5f50bdf0e6e7a276c8e0c097bccd8e1c095aee2`, its worker receipt, successful parent terminal, independent audit, complete 44-payload manifest and inherited source/runtime closure. Authenticate checkpoint bytes before deserialization. The three starting models are fixed:

| Original fitting seed | Starting checkpoint | SHA-256 |
| --- | --- | --- |
| 10101 | `final-mlp8-10101.npz` | `47d662f8c85c6ae9c957accd599189c8c2d26d19e6f55e5a40fd8b275a36d8f7` |
| 10102 | `final-mlp8-10102.npz` | `393fc8f14eb393ac1be8a50f2a3382b2c35f6ca8ae9b6b79ac655ba831c4d526` |
| 10103 | `final-mlp8-10103.npz` | `445b7028d0294ffdc907f0590e4395b67ae16889cc63752e1ea963a05cfab47d` |

For each seed, create `continued_mc`, `observation_backup` and `unchanged` identities. Machine-readable arm names are `mc@seed`, `backup@seed` and `reference@seed`, respectively. Only the first two train, yielding six fits. The unchanged reference retains its original checkpoint and baseline byte identities; it receives no optimization. Restore both learners from exactly the same starting parameter/buffer arrays, save their hashes, and use fresh, independent Adam state. The original Adam states were not saved; this is weight continuation with a fresh optimizer, not optimizer resumption.

TRAIN remains all **5,589 pre-action prefixes from 192 naturally completed teacher episodes**, with no sampling, episode weights, DAgger states or new collection. Bind original row ordering, reconstructed public posterior hashes, features and scalar targets. Use uniform row MSE. The stored baseline `c0` remains fixed; do not recenter it on backup targets. VALID remains all **1,109 prefixes from 48 separate teacher episodes**, descriptive only. Neither VALID nor any old or new EVAL state may enter target generation, fitting, schedule choice or checkpoint selection.

## Fixed equal-update schedule

Each continuation trains for **40 additional epochs**, using CPU float32, one numerical thread, deterministic Torch operations, Adam learning rate 0.001, betas `(0.9, 0.999)`, epsilon `1e-8`, zero weight decay, batch size 128 and gradient norm cap 5. Retain the short final batch and average loss over its actual rows. The frozen last checkpoint, epoch 40, is the only deployed continuation checkpoint.

For each seed, initialize an independent `numpy.random.default_rng(seed + 30000)` per arm. Both arms receive the identical 40 permutations in the same order. Record every permutation hash. There are **44 updates per epoch, 1,760 per fit and 10,560 across all six fits**. Optimizer moments continue across all 40 epochs within a fit and are never reset at a target refresh. No hyperparameter search, early stopping, validation selection, retry or adaptive extension is allowed.

- **Continued MC:** retain each fixed float32 target `(T-t)/64` throughout all 40 epochs.
- **Observation backup:** before epochs 1, 6, 11, 16, 21, 26, 31 and 36, snapshot the current learner into an immutable target checkpoint. The first snapshot is the original epoch-80 checkpoint. Materialize one target per TRAIN state from that snapshot, then reuse the target array for exactly five epochs. The next snapshot follows five completed optimization epochs. Never update target weights within materialization or a five-epoch block.

For raw branch product `u[a,h]`, use `w[a,h]=max(sum(u[a,h]),1e-10)` and `z[a,h]=u[a,h]/w[a,h]`. Evaluate physical target values as `C_bar(z)=64*f_bar(z)` in the existing float64 deployed arithmetic. The normalized target is

`y_backup(b) = min_eligible_a [1 + sum_h w[a,h]*C_bar(z[a,h])] / 64`.

Use the **true eligible minimum** for the target. The deployed policy still selects the first eligible numeric action strictly within `1e-10` of the minimum. These are distinct operations; the chosen near-tie cost must not silently replace the exact target minimum.

Preserve all four raw action scores and all four hit branches, including blocked-action calculations, zero/subfloor branches, signed values and biased zero-input outputs. Do not renormalize weights, clamp values, add a discount, fuse the homogeneous shortcut or impose a learned terminal-value override. Found continuation is omitted through the existing kernel-origin convention; this does not imply that a biased model's value at an all-zero input is zero. Every actual found or horizon-censored public update remains assimilated.

There are **24 complete refreshes**, **134,136 TRAIN-state backups** and **2,146,176 evaluated scalar branch rows** before adding separately recorded qualification or parity work. Stream bounded batches of at most 32 current states, preserving the target primitive's one sixteen-row forward per state. Do not retain all dense branch features. Save all 24 target checkpoints, float64 target arrays, float32 training casts, row identities, branch masses/values/costs and target diagnostics. These are self-generated targets, not common labels or observed action returns.

Record final-checkpoint TRAIN/VALID diagnostics for all six continuation fits, with no selection. No intermediate validation calls or checkpoint choices are made. Final MC-return prediction diagnostics use the same original targets for both arms; a backup training loss and an MC training loss have different labels and are not comparable measures of policy quality. Record target growth, sign, cast error and refresh-to-refresh drift without repair. A nonfinite target, loss, gradient, weight or deployed value is a technical failure; finite poor behavior remains in the study.

## Numerical qualification before admission

The new target/continuation pipeline requires a separate frozen, bounded **TRAIN-only** engineering qualification before the scientific run is admitted. Its row indices are exactly `numpy.linspace(0,5588,64,dtype=int64)`. Disposable copies of original checkpoint 10101 train once in each mode for six epochs, batch 32, with the unchanged learning rate, clipping and five-epoch refresh interval. This exercises the second target refresh at epoch six. It performs 24 optimizer updates, two refreshes and 128 sixteen-row target readouts. The unchanged initial model and two final qualification exports each receive parity on the first sixteen full TRAIN rows: 96 NumPy and 96 Torch forward calls, plus three parity restorations. Independent replay checks all 128 saved target rows. No VALID diagnostic or native episode is run. Qualification caps are 180 suspend-inclusive seconds, 4 GiB RSS and 128 MiB output; the exact interpreter, package closure and supervisor are pinned. These weights are discarded for study initialization. Fix its rows and operations in advance, authenticate all inputs, and record branch, forward and optimizer attempts/returns. Any qualification updates use disposable copies; their weights or Adam states cannot initialize the scientific fits. The qualification may estimate execution cost, never choose methods, targets or evaluation seeds from outcomes.

Before autonomous evaluation, require unchanged-checkpoint restoration parity and all six final exports and all three unchanged references to match a Torch double copy on the predeclared first sixteen TRAIN prefixes. Compare current normalized scalar values, all sixteen physical branch values, four float64 costs and eligible selected actions. Preserve the prior `atol=rtol=1e-10` numerical bound and exact action agreement. Retain all comparisons and count their work. A failed qualification stops the attempt before EVAL, without widening tolerances, switching arithmetic or replacing a model.

The native environment/filter path is unchanged, so this protocol adds no repeated mechanical native episodes. Three separately counted template resets verify exact saved kernels and initial-hit mixtures for sensing lengths three, four and five. This does not authorize an unqualified new native path.

## Fresh paired evaluation: 144 cases, 1,440 policy episodes

Use **48 cases per setting**, not 48 episodes across the entire experiment. Every case is evaluated by all ten policies: the six continued fits, three unchanged checkpoints and `analytic_inbounds`. This yields **144 paired cases and 1,440 policy episodes**, including 1,296 learned-policy episodes. A larger 288-case comparison would require 2,880 episodes. The smaller choice doubles the previous per-setting case count while retaining all seeds and eight paired blocks; it is a bounded positive-control screen, not a power calculation or significance claim.

| Setting | Fresh evaluation seeds, inclusive | Cases | Role |
| --- | --- | ---: | --- |
| lambda3 | 12100001-12100048 | 48 | TRAIN-supported known kernel |
| lambda4 | 12200001-12200048 | 48 | TRAIN-supported known kernel |
| lambda5 | 12300001-12300048 | 48 | Unseen supplied kernel on the same grid |

Template seeds are 12500001, 12500002 and 12500003, respectively. No efficacy case uses a template seed. These proposed ranges were absent from the inspected local OTTO source/protocol seed declarations and 14 existing OTTO plan JSON files when drafted. Final plan assembly must repeat the disjointness check against the complete known local collection/evaluation ledgers. This establishes local nonreuse, not independence from an external model's unknown training corpus.

For case `c=0..47`, set `block=c//6` and `initial_hit=1+(c%6)//2`: eight blocks, two cases per initial-hit category in each block, 16 cases per category overall. Arm order is seed-outer, then `(continued_mc, observation_backup, unchanged)`, followed by analytic control. Rotate this ten-arm list left by `(setting_index*48+c)%10`.

Keep N=53, four hit categories, R_dt=2, Euclidean sensing and the 2,188-move horizon. Recreate each arm's environment with the same case seed and conditioned initial hit. Require matched source and initial public packet and matched uniforms by random-channel/index, without claiming that different trajectories obtain identical observations. Native source, posterior, seeds and draw records remain evaluator-only. Every controller maintains its own exact public posterior and selects only in-bounds movement. There is no fallback, forced exploration, anti-reversal rule or intervention.

Finding or reaching the full horizon ends an episode. Retain every failed search at 2,188 moves and assimilate the final observation in either case. Keep complete public transitions, posterior hashes, branch values/masses, all four raw costs, masks, choices, native evidence and cost journals. Do not replace cases, inspect partial quality to stop early, or rerun a favorable cohort. Preselected illustrative replays use case zero in all three settings and show all ten arms.

## Fixed continuation rule: all 42 conditions

First compute each arm's conditional mean within each initial-hit stratum, then apply the authenticated setting-specific initial-hit mixture. Family means equally average all three fitting seeds. A block mean first averages its two cases within each stratum, applies the same mixture, then averages fitting seeds. Pair each continuation with the unchanged checkpoint of the same original seed. Raw found counts, every seed, every stratum and all eight blocks must also be reported.

Technical completion of all six fits, three references, numerical parity, 1,440 episodes and the independent audit is required separately. The candidate is **observation_backup**. All following conditions are required; there are no substitutions, averages across settings or favorable-seed selections.

**Competence: 18 conditions.** For each of three backup fitting seeds in each of three settings:

1. Mixture-weighted success is at least 0.95.
2. Mixture-weighted capped moves are at most 1.05 times analytic control on the same fresh cases.

These absolute analytic anchors prevent a pass obtained only by improving on failed old models. Neither low training loss nor a large percentage gain over an incompetent checkpoint can satisfy them.

**Paired procedure improvement: 24 conditions.** In each setting, against each of the `continued_mc` and `unchanged` families:

1. Backup family success is no lower.
2. Backup family capped moves are at most 0.95 times the comparator.
3. At least six of eight paired blocks have a strictly positive capped-move gain, averaged across matched fitting seeds.
4. Backup family complete controller seconds per search are no greater.

Use full-precision deterministic aggregation with no decision epsilon. The move-gain threshold is inclusive; positive blocks are strictly positive. The cost comparisons concern complete deployed control, not training-cost parity. Passing all 42 conditions admits a separately frozen stronger objective comparison only. It does not establish a new architecture, biological mechanism, compact memory, optimal value or general robotic advantage. If both continuations become competent but the backup lacks its required paired gain, report that outcome and keep continuation false.

## Compute accounting and declared amortization

This is an **equal-update comparison**, not a compute-matched comparison. The backup arm pays for 24 target refreshes, branch construction, feature formation, target-checkpoint export/load, all scalar readouts, target casting/storage and ordinary optimization. Continued MC performs the same 1,760 updates per fit without artificial extra forwards or idle work. A future compute-matched experiment must freeze a different MC update schedule using TRAIN-only timing qualification before any evaluation; it cannot be inferred or retrofitted from this run.

Record physical totals for preparation, each fit, each refresh, export/parity, common setup, deployment setup, native resets/steps, and complete evaluation. Training loss/target production is counted within its actual enclosing phase; do not double count nested work. Controller time includes actor initialization, branch construction/copies, scalar features/readout, cost reduction/masking and every public update. Exclude only measured nested artifact I/O. Keep simulator and fitting time separate. Each of the nine deployed heads is physically loaded once, including separate unchanged references; allocate its load over **144** evaluation episodes. Allocate shared deployed model/branch-module setup over **1,296** learned-policy episodes. Retain any inherited analytic-table cost even when unused by the learned readout.

Report additional **incremental continuation amortization** at exactly **H=1, 100 and 10,000 searches per model**. For each continued model, let `L` be its complete continuation phase, including all its target refreshes; allocate common TRAIN preparation equally across six continuations as `P/6`. Let `D` be its deployment load plus one ninth of shared deployed-module setup, and let `U` be its measured mixture-weighted controller time excluding that deployment setup allocation. Report

`amortized_seconds_per_search(H) = U + (L + P/6 + D)/H`.

For unchanged references, incremental continuation and TRAIN-preparation charges are zero; their actual deployment setup remains charged. Their original training is a common prior investment, not free lifetime training. Also publish the original physical preparation/training/setup totals so amortized views cannot hide them. The three H values are accounting scenarios, not repeated executions or estimated deployment demand. Do not select a favorable H, claim break-even beyond the observed evidence or merge these quantities with the frozen 42-condition decision. Analytic control's one-time/per-search setup must be split consistently when showing the same accounting scenarios.

## Frozen execution and independent readback

Root-approved prospective worker caps are **10,800 suspend-inclusive seconds, 8 GiB RSS, 8 GiB output, 1,443 native resets, 3,150,720 native steps and 10,560 optimizer updates**. Resets are three templates plus 1,440 evaluation episodes. No native qualification steps are allocated. The fixed main target-generation budget is 134,136 sixteen-row target forwards; qualification and final parity counts are separately fixed in the admission plan and charged in full. The study performs 288 NumPy and 288 Torch parity forwards across nine models, nine parity restorations, and 40,188 final scalar diagnostic forwards across the six continued models and the full TRAIN/VALID caches. Independent target replay makes another 134,136 sixteen-branch readouts, recorded separately from target generation. No remote model service or installation is allocated.

Use the qualified native-clock supervisor and isolated worker group, exclusive outputs, immutable source/input/runtime pins, strict deadline checks and durable attempted/returned operation journals. Preserve nonfinite failures, pending calls and all partial evidence. No retry, replacement cohort, late checkpoint choice, tolerance change or resource extension follows a failure. A completed worker alone does not authorize a scientific claim without successful supervisor closure and independent readback.

Before launch, bind the new protocol, runner, tests, numerical target primitive, independent auditor, qualification receipts, the original study's full source/payload/parent/audit chain and actual interpreter/distribution manifest. The independent full-study audit is separately capped at 5,400 suspend-inclusive seconds, 4 GiB RSS and 256 MiB output. Its exact source, command, inputs and output contract must be frozen before scoring. The in-worker independent target replay occurs after fitting, outside fit-wall accounting; it must agree on every saved refresh before EVAL begins. The saved-only audit should independently check complete dataset/target identities, all refresh checkpoints and target arithmetic, exact update-order/count witnesses, final values/branches/actions, public episode joins, full costs, all 42 conditions and the declared amortization. Count saved-checkpoint replay calls honestly. Optimizer trajectories, native randomness and timing truth remain authenticated execution evidence, not independently repeated training or simulation.
