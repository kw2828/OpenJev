# Autonomous control after fixed input conditioning

**Prospective protocol; autonomous execution has not occurred.** This fixes the outcome-blind [design recommendation](../output/otto-conditioning-v1/autonomous-design-review.md) as a separate experiment. Qualification and full evaluation each require frozen implementation, source/input/runtime identities, exclusive outputs and a supervised plan. Full evaluation additionally requires completed deployed-branch qualification and independent saved qualification replay. No training is part of either mode.

## Question and fixed models

Does the fixed gain-53 training parameterization improve autonomous search relative to gain 1, while meeting an absolute analytic-control anchor? This is an ordinary optimization control with the same width-eight model class and full public belief, not a new architecture, memory mechanism or connectome comparison.

The [completed scalar screen](otto-conditioning-results.md) passed **3/6** conditions and failed overall: all three TRAIN-excess conditions passed and all three required VALID improvements failed. That result is unchanged. The autonomous design was recorded before those outcomes and is executed after technical qualification regardless of scalar success or failure. Smaller scalar error does not establish better action ranking, and the failed [coverage collection](otto-coverage-collection-results.md) leaves its separate hypothesis unanswered.

Bind the completed conditioning plan, worker, original successful parent terminal, independent audit and full payload/source closure before decoding checkpoints. Retain all six `final-gain1-<seed>.npz` and `final-gain53-<seed>.npz` checkpoints at seeds **10101, 10102 and 10103**. Their exact float32 weights, fixed `c0` and gain metadata are immutable. Do not substitute older scalar, Bellman or capacity weights. Add the unchanged qualified `analytic_inbounds` controller. No additional fitting, calibration, clipping, fallback, checkpoint choice or seed selection is allowed.

## Separate deployed-branch qualification

Qualification is a **zero-native-call** mode. Authenticate the existing scalar cache and metadata, then use exactly its first eight TRAIN and first eight VALID states, including float64 beliefs, public positions, sensing lengths and eligible actions. These 16 mechanical prefixes are existing coverage, not representative validation or a fresh performance test.

For every one of the six final checkpoints, restore one Torch double copy and load one independent immutable NumPy float64 head. Each state produces one NumPy sixteen-branch forward and one Torch-double sixteen-branch forward. There are exactly **six Torch restorations, six NumPy loads, 96 head/state comparisons, 96 NumPy forwards and 96 Torch forwards**, evaluating 3,072 scalar branch values across both routes. Compare 1,536 paired scalar values, 384 paired action costs and 96 selected actions. There are no extra current-state scalar forwards, optimizer updates, native resets or simulator steps.

Each route receives identical qualified explicit branches. Require every finite branch value and raw action cost to satisfy `abs(numpy-reference) <= 1e-10 + 1e-10*abs(reference)`. Require exact agreement of selected eligible action IDs; numerical near ties do not permit a tolerance relaxation. Each parity row binds its head ID/checkpoint hash, split/original row index, public position, sensing length and exact posterior hash; retain all sixteen raw masses, floored weights, values from both routes, four costs, masks and choices. Previously qualified synthetic zero/subfloor/boundary behavior remains source-bound; these mechanical prefixes do not claim exhaustive branch coverage.

Qualification caps are **180 suspend-inclusive seconds, 4 GiB RSS and 128 MiB output**. Its separate independent saved audit has the same caps, reconstructs all fixed branch arithmetic and saved NumPy readouts, and counts its actual replay computation. Recorded Torch forwards remain authenticated execution evidence. A failure preserves the attempt and prevents full evaluation without repair or widened tolerances.

The full plan binds the successful qualification, original parent terminal, independent qualification audit and unchanged weights/source/runtime. It **does not rerun parity**. It physically loads the six deployment heads afresh and charges those loads. Thus qualification work and deployment work remain separately visible.

## Fixed fresh evaluation cohort

Use the unchanged 53x53 sampled-source environment, four hit categories, `R_dt=2`, Euclidean sensing and horizon **2,188**. The fixed settings and inclusive seed ranges are:

| Setting | Episode seeds | Cases | Arms | Episodes |
|---|---|---:|---:|---:|
| lambda3 | 15100001-15100024 | 24 | 7 | 168 |
| lambda4 | 15200001-15200024 | 24 | 7 | 168 |
| lambda5 | 15300001-15300024 | 24 | 7 | 168 |

For zero-based case `c`, set `initial_hit=1+(c%3)` and `block=c//3`. Every setting has eight fixed blocks, with one case from each hit stratum per block. The 72 environmental cases produce **504 episodes**. Multiple fitting seeds on a case are paired evaluations, not additional independent environmental samples.

The arm list is `[gain1@10101, gain53@10101, gain1@10102, gain53@10102, gain1@10103, gain53@10103, analytic_inbounds]`. Rotate it left by `(setting_index*24+c)%7`. Pair sources and random uniforms by channel/index across all seven arms in a case, without forcing identical observations along different action paths. The three settings use distinct case seeds, so their differences are not paired interventions on the same source realization.

Exactly three separately counted template resets use **15500001, 15500002 and 15500003** for lambda3/4/5. Require exact agreement with the already-authenticated saved kernels and initial-hit mixture weights. The unchanged native/filter implementation needs no additional native qualification episodes. Any implementation change invalidating that reuse must be separately qualified before the plan is frozen, not repaired during evaluation.

The [seed-ledger review](../output/otto-conditioning-control-v1/seed-ledger-review-02.json) records the actual inspected plan, declaration and episode-ledger scope, hashes and overlaps. It includes the stopped coverage collection and preserves the first review's current-study declaration classification error. It certifies nonoverlap only within those inspected local records, not global freshness or external-model corpus independence. Reserved coverage EVAL starts 14100001/14200001/14300001 remain unused by this experiment.

## Public controller and transfer semantics

Lambda3/4 are training-supported settings. Lambda5 is an unseen sensing length with its exact observation kernel supplied to the actor on the same grid. Its feature value and observation law differ, while task family, actions, objective and geometry remain fixed. This tests supplied-model parameter transfer, not unknown-kernel adaptation, grid-size transfer or general distributional robustness. No lambda5 data enters fitting or policy selection.

The actor owns its public posterior. Actual source coordinates, RNG state and native hidden state remain evaluator-only; require exact public/native posterior agreement at every reset and update. End only on finding the source or reaching the horizon. Retain and assimilate every final found or censored packet, and charge its update. No stuck-based stop, trajectory replacement or anti-oscillation override is introduced.

Deploy float64 upcasts of the fixed float32 parameters. Internally multiply only the spatial feature block by its fixed gain; retain raw mass for `c0` and unchanged mass-scaled position/lambda context. Do not fold the gain into weights or alter reduction order after qualification. Construct all four actions and four nonfound hit branches explicitly. Preserve native kernel zeros, raw masses `m`, weights `w=max(m,1e-10)`, successor input `u/w`, signed physical values `64*f(u/w)` and biased values at zero inputs. Do not normalize weights, clip values or introduce a terminal-value override.

Compute all four raw costs `1+sum_h(w*physical_value)`, including blocked stay-and-observe directions. Restrict selection to in-bounds actions and choose the first numeric ID strictly within `1e-10` of the eligible minimum. Both gains and analytic control use the same public information and allowed action set. No analytic teacher calls enter learned choices.

## Fixed thirty-condition screen

First average equally across the eight cases in each initial-hit stratum, then apply the setting's authenticated initial-hit mixture. Family means equally average all three fitting seeds. Each block applies the same mixture to its three cases, then averages paired fitting seeds. Failures contribute the full **2,188 capped moves**. Report all arms, settings, seeds, strata, blocks, raw successes and failures; do not pool away lambda5 failure.

Gain53 is the fixed candidate. All **30** direct, unrounded conditions are required, with no decision epsilon:

- **18 competence conditions:** for each candidate seed and setting, weighted success is at least **0.95** and mean capped moves are at most **1.05** times analytic control.
- **12 paired conditioning conditions:** in each setting, candidate-family success is no lower than gain1; capped moves are at most **0.95** times gain1; strictly positive paired move gains occur in at least **6/8** fixed blocks; and complete controller seconds per search are no greater than gain1.

Publish the same success and 1.05-times-analytic competence thresholds and outcomes for **all nine gain1 seed/setting cells** as descriptive controls, without silently adding them to or removing them from the 30-condition rule. Publish analytic outcomes too. Relative gains between incompetent families cannot pass the overall rule.

These are practical small-cohort screens, not calibrated significance tests or population success guarantees. A pass supports stronger separately frozen confirmation of this optimization recipe, not architecture novelty or optimality. A failure preserves the failed decision and does not rule out all conditioning, coverage, spatial processing or recurrent mechanisms. The earlier scalar failure is never reversed by a new outcome.

## Full costs, limits and saved audit

Use the unchanged pinned CPU runtime and one numerical thread, with no installation, remote model service or training. Full-worker caps are **5,400 suspend-inclusive seconds, 8 GiB RSS, 6 GiB output, 507 native resets and 1,102,752 native steps**. Resets comprise three templates plus 504 policy episodes. At most **945,216 learned decision forwards** occur across 432 learned episodes, with sixteen scalar branch rows per forward. Full-mode parity calls and optimizer updates are zero. Hard caps are not guaranteed completion times.

Each of six independent deployment heads is loaded once and its complete load/validation cost allocated over **72** searches. Allocate learned module setup over **432** learned episodes. Complete controller cost includes actor initialization, any inherited analytic-table construction, branch arrays/copies, centering/context features, gain scaling, prediction, reduction/masking/selection and every posterior update. Retain first measured calls and all outliers. Exclude only measured nested artifact I/O; publish raw instrumented time and excluded I/O separately. Native template/reset/step work, qualification, original fitting and full worker wall time remain separate disjoint scopes. One rotated run does not establish repeated latency behavior.

Publish already-paid per-head fitting costs without rerunning training. At descriptive amortization horizons `H=1,100,10000`, report steady per-search controller cost plus `(head fit cost + common fitting preparation/6 + deployment load + learned shared setup/6)/H`. Show unallocated physical totals, qualification and diagnostic costs separately. The scalar worker's original preparation/pairing/scoring and nested parent wall are not silently duplicated. These accounting views do not add an efficacy gate; analytic setup must be treated consistently when compared.

Save complete public transitions, posterior witnesses, raw masses/floors, sixteen values, four costs, masks/actions, evaluator-only source/random-channel evidence and attempted/returned operation/cost journals. Preserve pending calls, partial outputs and technical failures. Do not retry, add cases, switch routes, replace checkpoints or extend limits after launch.

Only after successful full worker and original supervisor closure, run the separately frozen saved-output auditor under **1,800 seconds, 4 GiB RSS and 128 MiB output**. It authenticates all source/input/output joins, independently reconstructs all 504 public trajectories and saved learned readouts, and verifies costs, cohort counts and all 30 conditions. Count numerical replay work honestly. Native randomness, historical optimization and timing truth remain authenticated execution evidence rather than newly simulated facts. Audit disagreement prevents a completed scientific claim.
