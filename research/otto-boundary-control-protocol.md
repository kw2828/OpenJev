# Prospective movement-restriction control for the released OTTO policy

21 September 2026. The [completed reference comparison](otto-released-reference-results.md) found a baseline gain and a shifted failure. A saved-only diagnosis traced the censored case to 1,928 legal stay-and-observe choices at a corner while the posterior remained normalized. This motivates a simple policy control before proposing a new memory architecture. This experiment does not change the old cohort, rescue its failed gates or train a model.

## Question and four fixed arms

Does restricting the released value policy to directions that change position improve autonomous search on fresh cases? Does the restricted policy meet the already specified competence and teacher-quality margins against strong analytic control?

1. `released_tf`: unchanged original TensorFlow model and RLPolicy, all four choices.
2. `released_inbounds`: exactly the same model, four raw float32 costs and single original policy forward. Only final selection is restricted to the current public packet's in-bounds directions. Preserve the first-action near-tie rule with float32 subtraction and strict epsilon 1e-10. Keep all four raw costs; do not replace excluded neural costs with infinity or rerun the model.
3. `analytic_all4`: the qualified space-aware objective scoring all four directions, including blocked stay-and-observe actions.
4. `analytic_inbounds`: the same analytic objective restricted to directions that change position, retaining its explicit null convention for excluded scores in saved JSON.

The restriction is a changed policy, not a correction of an illegal environment action. Both restricted arms must make zero blocked moves as an implementation invariant. That property alone does not establish improved outcomes. Every actor receives the applicable exact kernel and only public position, observation, done, step and movement metadata. All four maintain their own exact qualified posterior. Source, native posterior, seed and random draws remain evaluator-only. Verify exact public/native posterior agreement at reset and every update, including the final censored observation or found-source point mass.

Before autonomous execution, qualify the new selector against all sixteen previously saved neural score vectors at eight public prefixes. Use a clearly labeled fake policy returning those recorded scores, with one fake call per comparison and no neural or simulator invocation. Require unchanged raw bytes, original four-action selection validation, an independently computed restricted choice and the correct pending-action lifecycle. Synthetic tests must also cover blocked minima, ties among permitted actions, malformed masks/scores and final/reset semantics. Bind the completed preflight to the new plan.

## Fresh fixed cohort

Keep the qualified 53x53, two-dimensional environment, four hit categories, Euclidean kernel, R_dt=2 and 2,188-move horizon. The baseline sensing length is three and the shift is four; each actor receives the correct kernel. This is a fixed-grid known-kernel shift with an unchanged value network, not unknown-model identification or the automatically resized upstream task.

Use 96 cases per regime. Case c=0..95 has block c//12 and initial hit 1+(c%12)//4, giving eight blocks and four cases per hit per block. Baseline seeds are **870001-870096**, shifted seeds **880001-880096**. These seeds were absent from the inspected local OTTO source/protocol sets before this plan. Their absence from the released checkpoint's training is not established.

Run all four arms on every case: **768 episodes**, including **384 neural-policy episodes**. Rotate the four-arm order left by global case index modulo four. Reconstruct each environment independently with the same case seed and conditioned initial hit. Require identical sampled source and initial public state within every four-arm case, and paired categorical uniforms at every shared random-channel index. Different actions may lead to different observations.

For each original/restricted neural pair, retain and compare the common public-history prefix. Every shared-prefix posterior and all four raw float32 costs must match exactly. The first selected-action divergence may occur only when the unrestricted policy chooses a blocked direction. Both actions must otherwise agree while the histories remain identical. Publish one paired-prefix record per case. This checks that the experiment isolates the intended selection change; it does not assert parity after histories diverge.

There are no replacements, outcome-dependent stopping, additional tuning cases or reruns under this plan. Stop each episode on finding or the full horizon. A failure contributes all 2,188 moves. Primary means use each regime's authenticated positive-initial-hit mixture. Preserve all strata, raw found counts, paired cases and eight weighted block means.

## Prospective interpretation

The **descriptive restriction-benefit screen has five checks**: in each regime, the restricted policy has no lower weighted success and no greater weighted capped moves than the unrestricted model (four checks), plus strictly higher success or strictly fewer capped moves in at least one regime (one check). This is a descriptive paired-cohort screen, not a significance test, a novel architecture result or permission to infer a general rescue rate from a rare failure. Report all paired changes and block differences, including ties and regressions.

Independently apply the unchanged prior quality rules to `released_inbounds` against both analytic controls:

- **Competent reference, six checks:** in each regime, weighted success at least 95%, and capped moves no more than 105% of each analytic control.
- **Promising teacher candidate, twelve checks:** for each analytic comparator in each regime, no lower success, capped moves no more than 95% of the comparator, and positive weighted move gains in at least six of eight blocks.
- **Utility/computation, sixteen checks:** for each analytic comparator in each regime, no lower success, no more capped moves, no more complete controller computation, and a strict improvement in moves or computation.

All conditions in a rule must pass. An unchanged original-policy result on the earlier cohort remains failed regardless of this outcome. Even passing the teacher screen establishes only a promising existing reference under this protocol; distillation usefulness and novel learned-model performance require separate experiments. No training pilot is automatically admitted by the five-check restriction screen.

## Complete cost, storage and evidence

Keep the original qualified checkpoint, TensorFlow/Keras configuration, eight-way symmetry averaging, sixteen action/observation branches, original score arithmetic and exact 43-distribution native runtime. Each decision of either neural arm must execute exactly one original model forward. The NumPy port remains unqualified.

Measure actor initialization, full choices and updates. Include the restriction's selector/mask work within choice cost. Physically perform model-specific framework/source import, construction, graph build, checkpoint loading and exact tensor validation once, and allocate its complete measured setup equally across all **384 neural episodes**. Keep common imports, added restricted-actor class import, evaluator/native work and kernel loading separately accounted. Preserve first-forward cold cost in its actual episode. Report raw instrumented timing and measured nested artifact-I/O exclusion; never add model-forward time twice. These are single rotated CPU timings, not repeated deployment latency or architecture-independent compute estimates.

Retain mutable posterior storage, immutable kernels/distance tables, model tensor bytes and process RSS separately. Save all four raw neural costs, allowed masks, selected actions, public transitions, posterior hashes, actual neural branch masses/values, sources and compact draw witnesses for evaluator use, episode rows and nested attempted/returned operation logs. Excluded analytic costs are null, with masks; all four neural costs stay finite even when selection is restricted.

The new runner may reuse hash-authenticated frozen reference helpers in separate namespaces. Prior authentication must retain its original constants. Configuring the execution namespace or deferring a single setup-metadata write must not change model inputs, weights, score computation or prior files. Bind every reused and new source in the plan. The independent reader may reuse the prior independent arithmetic helpers, but may not use the producer's metric/rule functions or execute policies or models.

## Freeze, caps and publication

Authenticate the completed prior reference plan, worker, supervisor and independent audit, their full inherited source/input closure and the new selection preflight before numerical imports. Commit and push the new plan, sources, tests, protocol and engineering receipts before the first autonomous call. Use exclusive output directories and unchanged frozen sources throughout execution.

Run once under the suspend-inclusive supervisor with caps of **1,800 seconds, 4 GiB RSS and 512 MiB output**, at most **768 resets**, **1,680,384 native steps** and **840,192 neural forwards**. Zero training updates and zero remote model calls are allocated. Preserve partial/failed receipts and unresolved attempts on any cap, crash or mismatch; do not replace seeds, restart the scientific run or extend its budget.

Completion requires all 768 episodes, all 192 paired-prefix checks, complete returned work, unchanged sources/inputs/runtime, successful worker/supervisor closure and absent process group. An independent saved-output audit must reconstruct beliefs, selections under the correct masks, pairing, full costs and every five/six/twelve/sixteen-check rule before effectiveness is published. Actual model outputs, runtime and timing remain authenticated execution evidence, not independently rerun facts.

Any replay must use the preselected first case of each regime, seeds 870001 and 880001, show all four arms and label playback timing illustrative. A boundary case may be described as an outcome-selected diagnostic, never substituted for these prospective replays or omitted from aggregates. Preserve previous failures and report a simple restriction's effect as such; no recurrent, biological or post-transformer novelty follows from this control.
