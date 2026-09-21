# Autonomous comparison of the released OTTO value policy

21 September 2026. The [released original TensorFlow policy](otto-released-native-qualification-results.md) now passes public/native integration, while the earlier learned action heads failed autonomous competence. This experiment measures a stronger existing learned reference before investing in another compact readout. It does not test a new architecture, train a model, amend earlier failures or reproduce the published benchmark's full conditions.

## Question and three fixed controllers

Does the released full-belief value policy provide materially better autonomous search decisions than strong analytic control, under our two known sensing models? Is any apparent gain explained by how the policies treat blocked directions?

Use exactly three arms:

1. `released_tf`: the authenticated 13,390,849-parameter checkpoint, original TensorFlow value model and original RLPolicy, including eight-way symmetry averaging, sixteen action/observation branches, original branch-mass floors and first-action tie rule. All four directions are scored.
2. `analytic_all4`: the published space-aware infotaxis objective evaluated on the same actor-owned exact posterior, extended to score blocked directions as stay-and-observe actions.
3. `analytic_inbounds`: the same posterior and analytic objective, preserving the upstream policy's exclusion of blocked directions. This separates the action-choice restriction from the analytic objective. It is not claimed to reproduce the earlier local planner's floating-point implementation or recorded timings.

All actors use the qualified public belief initialization/update, including its normalization threshold, zero-likelihood handling, found-source point mass and consecutive-packet lifecycle. They receive only the applicable known kernel and public position, hit, done, step and in-bounds movement metadata. A blocked action stays in place, consumes one move and obtains an observation. The metadata describes movement, not an environment prohibition. Native posterior, source, random seed, random state and draw records remain evaluator-only. Every reset and completed update is checked against the evaluator's native posterior without supplying it to the actor.

The analytic all-four version is an explicit extension of the heuristic. Before autonomous work, qualify both analytic readouts against unchanged upstream heuristic arithmetic on the eight previously recorded mechanical prefixes, including boundaries. The all-four comparison changes only the reference view's movement-permitted Boolean, preserving the resulting position. Require exact score and selected-action agreement. This uses saved public history and pure numerical functions, not new simulator calls, neural forwards or sampled performance cases. Bind that engineering receipt into the prospective study plan.

## Fresh local cohort

Use a 53x53 grid, two dimensions, four hit categories, Euclidean sensing, R_dt=2 and a 2,188-move horizon. The baseline has sensing length three; the shifted regime has length four. Each actor receives the regime's exact kernel. The fixed-grid known-kernel shift is not unknown-model identification or the automatically sized upstream lambda4 task.

Each regime has 96 cases. For case `c=0..95`, use block `c//12` and initial hit `1+(c%12)//4`: eight blocks, four cases per hit per block, and 32 per hit overall. Baseline seeds are **850001-850096**; shifted seeds are **860001-860096**. These have not been used in the local OTTO studies inspected. The released checkpoint's training-condition membership is not known, so do not call them proven unseen training environments.

Run all three arms on all 192 cases, **576 episodes**. Rotate arm order left by the global case index modulo three. Within a case, reconstruct the environment independently with the same seed and conditioned initial hit. Require identical initial source and public state. The separate initial/source/hit random channels are paired; different paths can produce different observations. There is no outcome-based replacement, fallback, teacher correction, early numerical-criterion stopping or favorable arm selection.

Every actor starts fresh each episode. Stop on public source finding or after 2,188 moves. Incorporate the final observation even on a censored episode, and preserve the real terminal point mass on finding. All failures contribute the complete horizon to capped moves. Record blocked-stay counts, all raw successes, conditional-hit outcomes and each paired block. Use each regime's previously qualified positive-initial-hit mixture for primary weighted means; report all three strata and unweighted counts alongside it.

## Prospective interpretation

These rules screen the usefulness of a released comparator. They are practical engineering margins, not significance tests, a new-model result or permission to revise old continuation rules.

**Reference competence** requires all six conditions: in each regime, weighted success at least 95%, and weighted capped moves at most 105% of each of the two analytic controls. Passing this permits the term competent local reference, not improved model.

**Promising teacher candidate** requires all twelve conditions: for each analytic control in each regime, no lower weighted success, weighted capped moves at most 95% of that control, and positive weighted paired-block move gains in at least six of eight blocks. Both controls and both regimes must pass. Passing does not establish that a student can learn the improvement; a separate distillation experiment would still be required.

Separately report a descriptive utility/computation comparison against each analytic control in each regime: no lower success, no more capped moves, no more complete controller computation, and at least one strict improvement in moves or computation. A slower high-quality reference may be useful as a teacher while failing this computation comparison. Do not report an architecture advantage for either outcome. Preserve every failed condition and every episode regardless of the aggregate result.

## Complete cost and storage

Measure actor initialization, every posterior update and every complete choice, including symmetry expansion, all action/observation evaluations, output validation and tie selection. Record raw instrumented intervals and separately measured nested evidence-writing overhead; primary controller time excludes artifact serialization/journal writes, not model or algorithm work. Preserve both quantities and their arithmetic. Do not use model-forward time alone as controller cost.

Measure model-specific framework import, graph construction, checkpoint loading, tensor validation and immutable model setup once, then allocate that actual setup cost equally across the **192 released-policy episodes**. Report this allocation and warm execution separately. Include the first actual forward's cold cost in its episode. Analytic actor setup is measured per episode. Shared evaluator/kernel authentication, native initialization/steps, posterior comparisons and evidence writing are separately reported rather than silently attributed to one algorithm. This is one rotated instrumented CPU run, not a repeated deployment-latency benchmark or a hardware-independent compute estimate.

Report evolving posterior bytes, immutable kernel/distance-table storage, released-model tensor bytes/parameter count and whole-process peak RSS separately. Low mutable state alone is not low total model memory. Actual execution journals count model construction, graph build, loading, each real model forward, native construction/step, actor construction/choice/update and any saved-only qualification work separately. Preserve attempted and returned operations and an unresolved call on failure.

## Freeze and bounded execution

Use the unchanged, already-qualified `.venv-otto-released-native` runtime, all 43 pinned distributions, original weights and upstream sources. Authenticate the successful native qualification and its independent saved-output audit, complete prior source/input closure, new controller/runner/tests/protocol and the new analytic engineering receipt before importing numerical code or constructing objects. Keep the failed NumPy port unqualified.

Commit and push the complete plan before the first autonomous native call. Run once under the existing suspend-inclusive supervisor with limits **1,800 seconds, 4 GiB RSS and 512 MiB output**. There are at most **576 native resets**, **1,260,288 native steps** and **420,096 neural forwards**. Previous mechanical work remains a separate completed run; no additional native qualification calls are allocated here. No training updates or remote model calls occur. A timeout, cap violation, unresolved call or source change preserves a failed/partial attempt; there is no rerun or budget extension under this plan.

Preserve all episode rows, public transitions, four action costs and choices, neural successor values/branch masses, belief hashes, evaluator-only sampled sources and compact categorical draw witnesses, timing components and complete operation counts. Do not save the large per-decision model input tensors. Keep raw nonfinite analytic blocked costs explicit as unavailable/null with the in-bounds mask; never silently serialize NaN or infinity as numeric JSON. Require finite costs for every permitted action.

Completion requires all 576 episodes, complete expected work, unchanged sources/inputs/runtime, successful worker and supervisor closure, absent process group and no late-failure marker. An independent saved-output reader must recompute cohort coverage, public transitions, paired initialization, action selection, aggregates, cost arithmetic and every rule before publishing effectiveness. Model/simulator execution and timing truth remain authenticated producer evidence. Any visual replay must use preselected case zero in each regime, show all three arms and label playback timing illustrative.
