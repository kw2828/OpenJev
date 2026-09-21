# Fixed student-coverage control

**Prospective design; not executed.** This document fixes the scientific contrast and Stage 1 collection/selection rules. A source-, input- and runtime-bound machine plan must precede collection. Stage 2 additionally requires a complete independent collection audit, separately qualified implementation and a separately frozen training/evaluation plan. Writing this protocol admits no training or evaluation by itself. No closed result or source is modified.

## Question and existing evidence

Does replacing approximately half of a fixed teacher-state dataset with fresh states visited by the unchanged students improve ordinary MLP control under the same delayed observation-backup procedure?

The [previous Bellman comparison](otto-bellman-control-results.md) passed **19/42** conditions and failed its overall rule, including all 18 competence conditions. The [capacity screen](otto-capacity-results.md) passed **6/12** conditions: every TRAIN-excess criterion passed, every required VALID improvement failed. Width 128 reduced mean VALID MSE by 7.93%; the deeper model increased it by 1.37%. These findings motivate a coverage control without establishing the cause of poor actions. Stable fit to self-generated targets is not evidence of correct action ranking.

Student-state training is established prior work and has already been attempted here. The [earlier symmetry/action-head study](otto-symmetry-head-results.md) pooled student trajectories, then passed 2/54 conditions and failed. This comparison changes state exposure within the fixed width-eight scalar-value model and Bellman objective; it neither introduces a new architecture nor establishes that coverage was generally untested. The [OTTO learning method](https://auroreloisy.github.io/papers/Loisy2023a_EurPhysJE_drl-benchmark.pdf), Algorithm 1, already uses explored transitions, replay and delayed targets.

## Bound inputs and unchanged controllers

Authenticate the original scalar study's completed plan, worker, successful parent terminal, independent audit, full 44-payload manifest and inherited source/runtime closure before decoding arrays. Bind the completed Bellman and capacity records as the decision context without using their EVAL arrays for collection, selection, targets or fitting.

The input teacher population is exactly the original **5,589 TRAIN pre-action states from 192 episodes**. Its six strata are `(lambda3|lambda4, initial_hit=1|2|3)`; original row counts and order are authenticated metadata. No VALID, earlier student collection or any old EVAL row enters the new mixture. Existing 1,109-row VALID diagnostics, if retained by the Stage 2 implementation, remain descriptive and cannot select schedules, targets or checkpoints.

All three collectors and future fit starts are the original `mlp8` checkpoints, with fitting seeds **10101, 10102 and 10103**, from `output/otto-return-value-v1/run-01/final-mlp8-<seed>.npz`. Restore their exact float32 parameters and `c0`; do not substitute capacity-screen or Bellman-continuation checkpoints. All use the unchanged full public posterior, 11,028 scalar features, known kernel, sixteen explicit float64 branches, mass floor `1e-10`, signed values and in-bounds action selection. Actual native state, source and RNG records remain evaluator-only. No policy fallback, exploration override, anti-cycling rule or post hoc checkpoint choice is introduced.

## Stage 1: fixed fresh TRAIN collection

| Setting | Episode seeds, inclusive | Cases | Collectors per case | Trajectories |
|---|---|---:|---:|---:|
| lambda3 | 13100001-13100012 | 12 | 3 | 36 |
| lambda4 | 13200001-13200012 | 12 | 3 | 36 |

For zero-based case `c`, `initial_hit=1+(c%3)`, giving four cases per hit stratum per setting. The collector list is ascending seed `[10101,10102,10103]`; rotate it left by `(setting_index*12+c)%3`. Each collector is reset on the same case seed and initial hit. Require matched initial public packets and sources and matched random uniforms by channel/index; different actions need not produce identical observations.

Use the unchanged 53x53 native task, four hit categories, `R_dt=2`, Euclidean sensing, supplied sensing lengths three/four and horizon **2,188**. Two separately counted template resets use seeds **13500001/13500002** and verify the existing kernel/initial-mixture identities. This allocates exactly **74 resets** and at most **157,536 native steps** across all 72 collection episodes. Root must verify local seed nonreuse against completed and admitted plans before freezing the executable plan; local nonreuse is not a claim about external pretraining.

Retain every full public trace, action, posterior witness, found/censored status, selected evaluator source, compact random-channel records and paid cost. Assimilate every final found or horizon-censored observation. The selected candidates are pre-action states `t=0..T-1`, where `T` is the episode's actual step count. A final packet is preserved but is not an extra decision state. Short successful trajectories and full-horizon failures both remain in the eligible population. Collection outcomes do not change the fixed budget, quota or selection rule.

Worker caps are **900 suspend-inclusive seconds, 8 GiB RSS, 2 GiB output**, CPU with one numerical thread, no remote calls or installation. Record actual attempted/returned native and readout work, template/head setup, public updates, complete controller time, measured artifact I/O and physical worker time. A failure preserves partial outputs and pending calls; it does not authorize new seeds, replacement trajectories, retries or extension.

## Deterministic 5,589-row mixture

Let `M_s` be the authenticated original teacher count in stratum `s`, summing to 5,589. Fix **2,790 student rows and 2,799 teacher rows**, with final count `M_s` in every stratum.

1. Start student quota `Q_s=floor(2790*M_s/5589)`. Allocate the remaining student rows by descending exact integer remainder `(2790*M_s)%5589`, breaking ties lexicographically by `(regime, initial_hit)`. Teacher quota is `M_s-Q_s`.
2. Within each stratum, divide `Q_s` evenly across the three collector seeds. Give one residual row each to ascending collector seeds until the remainder is exhausted.
3. Teacher identity is `teacher:<original_row_index>`. Student episode identity is `train:<regime>:<episode_seed>:mlp8@<collector_seed>`; student row identity appends `:<preaction_index>`. Integers use ordinary decimal notation without zero padding.
4. Rank candidates within each fixed quota by `(SHA256(UTF8('otto-coverage-v1|'+identity)).hexdigest(), identity)` and take the smallest ranks. Selection is without replacement by row identity. Equal-feature states remain distinct rows; do not deduplicate or filter by success, value, loss, action or source proximity.
5. Final order is all teacher strata lexicographically, each in selected hash order, then student strata lexicographically, collector seeds ascending, each in selected hash order. Assign new contiguous `mixture_index` values; preserve original teacher indices or complete student addresses separately.

Complete all 72 trajectories before finalizing selection. If any fixed stratum/collector quota has insufficient eligible rows, fail selection without borrowing quota, adding episodes or choosing another salt. Publish candidate counts, every quota/remainder, selected identities/hashes, original and final stratum counts and dataset/source hashes. All fitting seeds receive this one immutable mixture.

Uniform row MSE is deliberate: neither dataset receives episode weights. Original stratum totals are preserved, while the student quota fixes collector representation. Hash selection from each pooled stratum/collector naturally gives longer trajectories more candidate rows; this tests the declared occupancy-weighted exposure, not equal episode influence or optimal diversity.

There are **no Monte Carlo labels for student rows**. Any zero array required by the existing continuation interface is explicitly an unused `mc_targets` placeholder, never a terminal value or training label. Both future arms use backup mode; the first complete target refresh must replace the placeholder before the first optimizer call, and the saved audit must verify that ordering. Do not derive censored return labels or add zero tails.

The separately capped saved-output audit is **900 seconds, 4 GiB RSS, 128 MiB output**. It must bind the original successful parent and complete payload hashes, reconstruct every public trajectory and selected state, verify all collector/checkpoint/case identities, quotas and exact ordering, and independently check saved readout/choice witnesses. Its numerical replay is counted separately, not described as zero computation. No simulator or training is rerun. A complete worker without a successful independent audit does not admit Stage 2.

## Stage 2: matched continuation, separately admitted

For each original seed, restore two independent copies with fresh Adam state:

- **Teacher only:** all original 5,589 TRAIN rows in original order.
- **Fixed mixture:** the selected 5,589 rows in the order above.

Both keep `c0` fixed and use 40 epochs, batch 128 including the short final batch, Adam learning rate 0.001, default betas `(0.9,0.999)`, epsilon `1e-8`, zero weight decay, gradient norm cap 5, CPU float32 and deterministic one-thread operations. Each uses independent `default_rng(seed+30000)` with the same 40 permutations of row indices. This matches index-order randomization, not state contents. Optimizer moments continue through all 40 epochs. Only the final checkpoint is deployed; there is no early stopping or validation selection.

Before epochs **1,6,11,16,21,26,31,36**, snapshot that arm's current model and materialize all its targets. For each state, retain every raw mass `m`, floored weight `w=max(m,1e-10)`, successor `z=u/w`, physical value `64*f_bar(z)` and all four costs. The target is the true minimum eligible cost divided by 64, with one final float32 cast. Deployed near-tie selection remains the first eligible numeric action strictly within `1e-10` of the minimum. No clipping, discount, fused shortcut, branch reweighting, terminal-value override or likelihood repair is allowed. Preserve biased zero-input values and subnormalized inputs.

Six fits yield **10,560 optimizer updates, 48 target refreshes, 268,272 sixteen-row target forwards and 4,292,352 scalar branch rows**, before separately declared qualification, parity, diagnostics or independent audit. These are equal row/update/refresh counts, not a total-compute match. Each arm generates its own targets: changed targets are part of the exposure intervention, not shared externally correct labels. A smaller self-target loss alone cannot support a coverage claim.

Before any Stage 2 execution, separately freeze its implementation, TRAIN-only disposable qualification, exact numerical parity coverage/tolerances, complete operation/payload counts, runtime/caps and independent audit. Qualification weights never initialize scientific fits. Previous successful engineering does not permit an unqualified data-path change. Signed finite targets are retained; nonfinite numerical work fails technically without repair. Undiscounted bootstrapping can diverge and is not a convergence guarantee.

## Fresh evaluation and fixed 42-condition rule

Retain all six continuations, three unchanged original checkpoints and `analytic_inbounds`: ten arms on **144 fresh cases, 1,440 policy episodes**. Proposed seeds are **14100001-14100048**, **14200001-14200048** and **14300001-14300048** for lambda3/4/5. Final plan assembly must confirm local nonreuse. Lambda5 is an unseen supplied-kernel setting on the same grid, not an unknown observation model. No case from any older EVAL is reused.

For case `c=0..47`, use `block=c//6`, `initial_hit=1+(c%6)//2`, eight blocks and two cases per hit per block. Arm order is seed-outer, then `(teacher_only, mixture, unchanged)`, then analytic; rotate left by `(setting_index*48+c)%10`. Keep horizon 2,188, all failures at the full cap, every final public update and common source/uniform channels. Policy inputs remain public. No teacher labeling or interventions enter student evaluation.

Apply authenticated setting-specific initial-hit mixture weights after within-stratum means; family means equally average all three fitting seeds. Block means apply the same mixture to two cases per stratum and average paired fitting seeds. Report every arm, stratum, seed, block, raw count and failed case. Use unrounded deterministic values with no decision epsilon.

The candidate is **mixture**. Technical completion and independent agreement are separate prerequisites. All **42** conditions are required:

- **18 competence conditions:** for each candidate fitting seed and each setting, weighted success is at least **0.95** and capped moves are at most **1.05** times analytic control.
- **24 exposure-improvement conditions:** for each setting against both the teacher-only backup family and unchanged family, candidate success is no lower; capped moves are at most **0.95** times the comparator; at least **6/8** paired blocks have a strictly positive move gain; and complete deployed controller time is no greater.

Passing is evidence for this fixed exposure recipe under these conditions, not architecture novelty, optimality, general robotic efficacy or recurrence/connectome superiority. Failing preserves the failed decision; it neither admits a different study automatically nor proves coverage, capacity, spatial processing or memory irrelevant. Further experiments require separately specified hypotheses and freezes.

## Complete costs and interpretation

Publish collection physical cost once, with template/head loading, all 72 controllers, native resets/steps, reconstruction, selection and audit separated. Charge complete collection cost **C/3 to each mixture continuation** in incremental amortization, since the shared collection serves the three mixture fits; teacher-only continuations receive no collection charge. Do not divide C across all six fits or omit unsuccessful trajectories. Report the actual physical total alongside allocations so accounting cannot imply collection happened three times.

Both arms pay all target construction, checkpoint publication, readouts, casts and optimization. Separate common preparation, dataset-specific preparation, diagnostics, deployment setup, measured artifact I/O and evaluator costs without double counting nested phases. Each of nine deployed heads is loaded once and allocated over its 144 searches; shared deployed setup is allocated over 1,296 learned-policy episodes. Complete controller cost includes initialization, branches/features, readouts, reduction/selection and final public updates. Explicitly show any unchanged inherited analytic setup cost.

Retain amortization scenarios **H=1,100,10000 searches per model**. Add C/3 only to each mixture model's one-time numerator; allocate genuinely shared preparation once across its actual consumers, with dataset-specific work assigned to its own family. State the allocation algebra in the frozen Stage 2 plan and publish unallocated physical totals. The three H values are accounting scenarios, not new executions or an efficacy gate. Prior checkpoint training is a common historical investment, not free lifetime training.

The decisive comparison is fresh autonomous control with an absolute analytic anchor. Capacity and spatial readout remain competing explanations if this recipe fails; new recurrence or biological wiring would need its own opportunity and matched controls. This study intentionally isolates a data-exposure intervention within one ordinary learning procedure.
