# Full-belief action scoring with symmetry and student-state training

21 September 2026. The [boundary control](otto-boundary-control-results.md) establishes a competent but expensive existing reference on its new cohort, with no boundary-restriction benefit. Its teacher-consistency rule still fails. The [earlier action head](otto-action-head-results.md) failed even with full belief, so compact memory is not yet an isolated explanation. This new pilot trains our own small scorer against an analytic teacher and tests architectural symmetry against ordinary augmentation and inference averaging.

## Question and attribution

Can a small direct action scorer match the analytic controller's autonomous quality at lower complete computation? Does sharing an action-centered scalar network improve on an ordinary four-output network given the same data, augmentation exposure and comparable active parameters?

Student-visited data aggregation follows the established motivation of [DAgger](https://proceedings.mlr.press/v15/ross11a.html): actions change the distribution of later observations. This pilot uses one fixed pooled aggregation round across students; it does not claim the original algorithm's guarantees. Symmetry sharing and frame averaging are established methods ([group-equivariant networks](https://proceedings.mlr.press/v48/cohenc16.html), [frame averaging](https://arxiv.org/abs/2110.03336)). Neither applying them here nor passing this pilot establishes a novel architecture. The longer-term question remains whether compact recurrent evidence can preserve a competent scorer's decisions with a useful quality/computation tradeoff.

Keep the exact public posterior fixed across all learned policies. No learned world model, compact memory, biological wiring, RL update, proprietary Jev reproduction or previous-failure reversal is part of this experiment.

## Shared public features and three inference families

Every learned scorer receives the same **2,836 float32 features**, extracted from its own exact public belief and supplied observation kernel:

- All 2,809 values of `53*sqrt(p)` on the fixed 53x53 grid, with no posterior normalization, floor, clipping or support repair added by feature extraction.
- Two public coordinates scaled by `coordinate/26 - 1`.
- Four indicators for directions that move within the grid.
- The known sensing length divided by five. Training includes lengths three and four, so this input is not constant.
- Twenty local forecasts: for each of four directions, posterior mass at its clamped successor and the four raw joint nonfound hit masses there. Compute them from the supplied kernel and current posterior without constructing normalized successor posteriors, entropy, a value model or an analytic objective.

Past hits, elapsed time and an extra zero-support mask are omitted because this test retains the full public posterior. The actor receives no sampled source, native belief, seed, evaluator draw or future observation. Fixed scaling replaces fitted preprocessing. Include all feature extraction, kernel cropping and transforms in decision cost.

The shared candidate uses a **2,836 -> 32 -> 16 -> 1 Tanh MLP**, with **91,329 parameters**. For each action, transform the complete input through both square symmetries taking that action to north and average the two scalar costs. The network evaluates all eight square rotations/reflections once per decision. Coordinates, action masks and forecast rows transform with the grid; sensing length stays invariant.

The ordinary network is **2,836 -> 32 -> 16 -> 4**, with **91,380 parameters**. Train on all eight transformed views of each state, transforming target action probabilities and masks consistently. Average the eight view losses before applying the state weight. This approximately matches eight first-layer evaluations per training state; exact work, parameters and time are still reported separately.

Evaluate that ordinary checkpoint two ways: `dense` uses one untransformed pass; `dense_ensemble` averages all eight inverse-permuted raw cost vectors. These are the **same weights**, not separately trained fits. The ensemble provides an eight-view inference control; matching view counts is not a claim of exactly matched total computation. The candidate's mathematical score equivariance is checked within floating-point tolerance; deterministic first-index tie-breaking is not claimed to be equivariant at symmetric states.

Train both architectures with paired seeds **9101, 9102, 9103**. There are six trained models, six initial and six final checkpoint artifacts, and nine learned evaluation arms. No favorable seed or ensemble is selected after outcomes.

## Native setup and fresh split identities

Use the unchanged pinned OTTO source, sampled-source adapter, exact public filter and analytic in-bounds objective. Keep N=53, Nhits=4, Ndim=2, R_dt=2, Euclidean sensing and the **2,188-move horizon**. Directions blocked by the boundary are legal stay-and-observe actions in native OTTO; all policies in this pilot deliberately restrict selection to moving directions.

Construct native templates at sensing lengths three, four and five with seeds **1000101, 1000102, 1000103**. Save their kernels and positive-initial-hit mixtures. Kernels at three and four must exactly match the authenticated previous kernels. Check the length-five kernel against the native formula, and qualify public/native filtering on eight **length-five** prescribed trajectories with seeds **1000001-1000008**, initial hit `1+i%3`, repeating actions `[0,2,1,3]`, at most 32 moves each, stopping on finding. No replacement after an early find. These are mechanical qualification cases, not effectiveness evidence.

All collection and evaluation episodes run to finding or the full horizon. Assimilate every final public packet, including a found-source point mass or a final censored observation. Check exact public/native posterior equality at reset and each update. The native draw log remains evaluator-only.

| Split | Length three | Length four | Length five | Cases per listed setting |
| --- | --- | --- | --- | ---: |
| Initial teacher TRAIN | 910001-910096 | 920001-920096 | none | 96 |
| VALIDATION | 930001-930024 | 940001-940024 | none | 24 |
| Student TRAIN, each of six initial fits | 950001-950012 | 960001-960012 | none | 12 |
| Final EVALUATION | 970001-970024 | 980001-980024 | 990001-990024 | 24 |

Initial hit is `1 + case_index % 3` throughout. The same student-TRAIN seeds are intentionally paired across fits; their collector identities remain distinct. No validation/evaluation state enters fitting or data aggregation. These seed ranges were absent from the inspected local OTTO sources and protocols before implementation. This does not establish their absence from any third-party model's training.

## One fixed data-aggregation round and optimization

First collect 192 analytic-teacher TRAIN trajectories and 48 analytic-teacher VALIDATION trajectories. Use the exact analytic moving-direction policy with no random-action mixture. After each complete trajectory, retain at most 64 pre-action prefixes at evenly spaced indices `round(linspace(0, length-1, min(64,length)))`, including both endpoints. Preserve the full public trajectory and sampling indices. Do not sample the terminal state as an action decision.

Targets come from the same analytic costs for both models. On permitted directions subtract the minimum, divide by `max(max-min,1e-8)`, negate, divide by temperature **0.25**, then apply softmax. Other directions have zero target mass. These are relative heuristic preferences, not optimal Q-values or calibrated probabilities. Obtain teacher labels without committing the teacher's chosen action when the actual behavior belongs to a student.

Give each retained episode equal total training weight. For a dataset of N rows from E episodes, a row in an episode with n retained prefixes has weight `N/(E*n)`. Apply these fixed weights to per-row cross entropy before the batch mean. Validation uses the same episode-balanced convention. This is balanced conditional-episode training, not the evaluation's initial-hit mixture.

Train all six initial models for **40 epochs**. Each then runs its 24 student-TRAIN episodes independently, with no teacher fallback. Pool the selected prefixes from **all 144 student trajectories** with the initial TRAIN data. All six models receive this identical pooled dataset; there is no method-specific training set. Continue each model's weights **and Adam state** for exactly **40 more epochs**. This is one data-aggregation round. There is no subsequent rollout round, hyperparameter search or evaluation-informed update.

Use CPU float32, one numerical thread, deterministic Torch operations, Adam learning rate **0.0003**, batch size **128**, gradient-norm clipping at five and paired per-seed row permutations across architectures. Validate every fifth epoch in both stages, producing 96 validation records. Keep the final epoch, regardless of validation or autonomous results. Save initial/final weights and all training/validation records. No checkpoint search is performed.

Initial TRAIN has at most 12,288 rows, student TRAIN at most 9,216, pooled TRAIN at most 21,504 and VALIDATION at most 3,072. At most **63,360 optimizer updates** are allocated. Qualify exported NumPy costs against Torch with absolute and relative tolerance **2e-5** on fixed saved validation rows. NumPy defines the deployed policy; cost-tolerance agreement is not exact action parity at a numerical tie. Every deployed choice uses the first permitted action within strict 1e-10 of its minimum recorded float32 cost.

## Autonomous evaluation and full computation

Run all nine learned arms and the analytic moving-direction controller on each of 72 cases: **720 episodes**. Settings three and four are training-supported; five is an **unseen supplied-kernel setting**, with the same grid size. Do not relabel length four as unseen. Every case has the same sampled source and initial public state across arms. Pair categorical uniforms by channel/index; differing actions can yield different observations. Rotate the ten arms by global case index modulo ten.

There are eight paired blocks per regime, each containing one case of each initial hit (`block = case_index//3`). Compute stratum means then apply the saved regime-specific positive-initial-hit mixture. Family means average all three fit seeds. Report each fit, family, stratum, paired case and block. Failures contribute all 2,188 moves. Preserve every censored search and make no replacements.

Physically load nine separate immutable runtime heads before evaluation, including separate loads of the same dense checkpoint for single-view and ensemble arms. Allocate each complete checkpoint loading/validation/construction time over that arm's 72 episodes. Separately measure the model module import and shared D4-table construction, allocating that one physical cost across all 648 learned evaluation episodes. Torch imports belong to training setup; NumPy inference does not require Torch. Charge per-episode belief/feature-map initialization, every feature/crop/transform/readout/mask operation and every posterior update. Measure nested artifact writing separately and exclude only measured nested I/O from controller cost; retain raw intervals. Never add a nested inference time twice. Analytic label computation is not available to a learned policy during evaluation.

Report collection, validation, data aggregation, training and setup time separately from evaluation; save complete process time. Mutable state, immutable kernels/model arrays, temporary work and process RSS are distinct. Equal parameters or equal view counts do not substitute for measured full cost. Timings are one rotated CPU run, not repeated deployment estimates.

## Frozen scientific rules

The candidate is always `shared`. All **54 conditions** below must pass for pilot continuation; no post hoc weighting or threshold change is allowed.

**Competence, 18 checks:** for each of three fit seeds in each of three regimes, weighted success is at least 95%, and capped moves are at most 105% of analytic control.

**Compression, 12 checks:** in each regime, family success is no lower than analytic control; family capped moves are at most 105% of analytic; family complete controller cost is at most 80% of analytic; and every fit's complete controller cost is strictly below analytic.

**Architecture comparison, 24 checks:** in each regime against each of `dense` and `dense_ensemble`, shared family success is no lower; family capped moves are at most 95% of the comparator; at least six of eight blocks have strictly positive move gains after averaging paired fit seeds; and family complete controller cost is no greater than the comparator.

Publish every condition with its values and threshold. A failed full-belief competence rule rejects this readout/data recipe without diagnosing compact memory. A pass would motivate untouched confirmation and then a compact-memory comparison; it would not establish novelty, statistical significance, biological learning or paper readiness. Every prior failed result remains unchanged.

## Execution boundary and independent readback

Before the first native call, commit and push the protocol, exact plan, new/reused sources, tests, runtime identity and engineering evidence. Bind the completed boundary-control plan, worker, parent and independent audit and their inherited source closure. Use the existing `.venv` Python 3.12.13, NumPy 2.5.3, SciPy 1.18.1 and Torch 2.14.0 without installation or changes. No TensorFlow, external model API or paid service is allocated.

Run once in an exclusive output directory under the suspend-inclusive supervisor. Limits are **5,400 seconds, 8 GiB RSS, 2 GiB output, 1,115 native resets, 2,415,808 native steps and 63,360 optimizer updates**. The step bound is `(192+48+144+720)*2188 + 256`; three setup resets add no step budget. Stop on integrity failure or cap, preserve partial artifacts and unresolved attempts, and do not restart the scientific run or extend its budget.

Completion requires all planned fits, data stages, evaluation rows and returned work, unchanged sources/runtime/inputs, successful worker and supervisor closure, and an absent process group. The independent saved-output reader has **1,800 seconds, 8 GiB RSS and 512 MiB output**, with no training or simulator calls. It checks split identities, exact sampling, pooled exposure/weights, checkpoint and history coverage, public filtering/features, saved-checkpoint readout values, selections, work/cost totals and all 54 rules. Count local checkpoint evaluations explicitly rather than claiming no model computation. Independently reproduced costs use declared floating-point tolerance, while recorded selected actions must obey recorded raw costs. Optimizer trajectories, native random execution and timing truth remain inherited evidence and must be disclosed.

Any replay uses preselected case zero from all three regimes (970001, 980001, 990001), retains all ten arms and labels playback timing illustrative. No effectiveness claim is published before completed independent readback.
