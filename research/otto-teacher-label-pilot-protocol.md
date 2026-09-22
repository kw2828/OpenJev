# Teacher continuation costs: six-anchor feasibility pilot

22 September 2026. **Prospective and not admitted for collection.** This fixes a small feasibility question before fitting: what capped action-cost panels, censoring, ties, paired variability and complete generation cost arise at six existing learner-visited TRAIN anchors? It has no statistical-confidence, label-quality, policy-efficacy or architectural pass rule. A technical completion is not a positive learning result and does not admit training or evaluation.

Before any rollout, independently qualify the extractor and collector sources, freeze the exact six anchor identities and their public-state/kernel hashes, establish the serialization/resource allocation below, and bind successful native qualification plus its independent saved-output review. This document alone is not that freeze. No empirical anchor is selected or reconstructed by writing it.

## Existing TRAIN provenance

Use only the completed, audited [symmetry study](otto-symmetry-head-results.md), whose [protocol](otto-symmetry-head-protocol.md) declares six initial student collectors and a distinct student-TRAIN phase. Its 144 student trajectories yielded 4,596 retained prefixes. Bind its [plan](../output/otto-symmetry-head-v1/plan-01.json), [worker receipt](../output/otto-symmetry-head-v1/run-01/receipt.json), [original supervisor terminal](../output/otto-symmetry-head-v1/run-process-01.terminal.json), [independent audit receipt](../output/otto-symmetry-head-v1/audit-01/receipt.json) and complete closed source/payload manifests before decoding inputs.

The allowed metadata is [dagger-rows.jsonl](../output/otto-symmetry-head-v1/run-01/dagger-rows.jsonl). Its recorded SHA-256 is `9935d48827e3a9b2e714383a2b3a87866bff0309205ac1316d75a0d5fdd5ce7b`; it contains public packets and posterior witnesses, not exact posterior arrays. The reconstruction source is [collection-transitions.jsonl](../output/otto-symmetry-head-v1/run-01/collection-transitions.jsonl), recorded SHA-256 `4224b39e4825cbec243c5d9e6fff323ccea336b7fb758f5cd54f705cebd3ed74`. The public observation inputs are the same run's [kernel-lambda3.npz](../output/otto-symmetry-head-v1/run-01/kernel-lambda3.npz) and [kernel-lambda4.npz](../output/otto-symmetry-head-v1/run-01/kernel-lambda4.npz). Live bytes must match their closed descriptors at extraction and collection.

The collection journal also contains teacher TRAIN and VALIDATION. Establish an exact allowlist of selected `dagger:<regime>:<episode_seed>:<collector>` IDs before decoding transition values. The extractor must filter authenticated raw JSONL records by the exact allowlisted episode identifier before numerical decoding, then strictly validate the selected record's complete identity and schema. Nonmatching bytes may be hashed for closure but are not scientific inputs. Do not decode VALIDATION/EVALUATION rows or arrays. Never consume `source_evaluation_only`, native posterior arrays, saved draw truth or rewards to reconstruct an anchor or generate its continuations. Ignore unrelated fields in a selected public record; extraction exports only the required public packet, action, posterior witness and provenance.

Do not invert `dagger-data.npz` float32 features to recover beliefs and do not load any learned checkpoint. The 5,589-row scalar-return TRAIN cache contains teacher-visited states and is not the learner pool for this pilot. The later failed coverage collection is not an alternative input or replacement pool.

## Fixed deterministic anchor selection

Select exactly one anchor in each row below, using the original retained student-TRAIN metadata only. Eligible metadata rows have the specified regime, initial hit and collector, `stage='dagger'`, a nonterminal public packet, and `prefix_index >= 1`. Among them choose the first lexicographically ordered `(episode_seed, prefix_index, row_index)`, where `episode_seed` is the metadata's integer `seed`. Preserve the corresponding episode identifier. The seed ranges are 950001-950012 for lambda3 and 960001-960012 for lambda4; each original initial hit is fixed by its case index.

| New anchor ID | Regime | Initial hit | Original collector |
| ---: | --- | ---: | --- |
| 0 | lambda3 | 1 | shared@9101 |
| 1 | lambda3 | 2 | dense@9102 |
| 2 | lambda3 | 3 | shared@9103 |
| 3 | lambda4 | 1 | dense@9101 |
| 4 | lambda4 | 2 | shared@9102 |
| 5 | lambda4 | 3 | dense@9103 |

This represents all six sensing/initial-hit cells and all six historical collectors once. It is deterministic feasibility coverage, not random population sampling or a comparison of collector families. Original retained-prefix sampling depended on episode length; that limitation remains. Do not rank by posterior mass, action gaps, eventual success, censoring or expected continuation cost. Do not deduplicate or substitute anchors. An empty cell is an explicit preparation failure.

Freeze the resulting six exact `(row_index, episode_id, prefix_index)` identities before posterior reconstruction or label generation. Reconstruct each selected episode's public belief from its original center prior and reset hit, then apply the saved actions/public observations through the selected pre-action prefix. Prefix `t` means the state after `t` updates and before action `t+1`. Check the original float64 posterior hash, mass and public position at reset and every reconstructed step, including the selected metadata witness. Never reset the continuation to a fresh native source or assimilate the anchor's last hit twice.

All six anchors must validate before the first source draw: finite nonnegative float64[53,53], total mass within the existing categorical tolerance `1e-10` of one, exactly zero current-cell mass, nonterminal packet, exact public-step continuity, exact geometric eligibility and the authenticated unchanged kernel. Copy the accepted entries without normalization, clipping or support repair. A missing row, hash mismatch, empty/subfloor belief or other unsupported anchor fails preparation, with no replacement, retry or reduced denominator. Record validation through six explicit `TeacherSnapshot` constructions; this is separate from the sampler's own validation snapshots.

## Fixed paired sampling

Use the unchanged [qualified rollout component](otto-teacher-rollouts-component.md), [snapshot initializer](otto-teacher-snapshot-component.md) and [cost reducer](otto-teacher-cost-component.md). Require the strict [native qualification protocol](otto-teacher-sampler-qualification-protocol.md), its [frozen plan](../output/otto-teacher-sampler-qualification-v1/plan-01.json), [worker receipt](../output/otto-teacher-sampler-qualification-v1/run-01/receipt.json), [original terminal](../output/otto-teacher-sampler-qualification-v1/supervision-01.terminal.json) and completed independent review in the final pilot input closure. Native parity is a mechanical prerequisite, not evidence that these labels improve learning.

The sampling seed is the fixed uint32 value **19000001**. Use anchor IDs **0 through 5** in table order and replicate IDs **0 through 15**, ascending. Do not derive randomness from filenames, row ordering, Python hashes or future outcomes. Each local generator is exactly `Generator(PCG64(SeedSequence([0x4F54544F, 19000001, anchor_id, replicate_id, channel])))`, where source channel is 0 and hit channel is 1. All entropy components are uint32 integers. These identities must remain unchanged after inspecting anchors or labels.

At each anchor require every geometrically eligible first action, ascending, for all sixteen replicates. Draw one source per replicate from the anchor's public belief and reuse it across the first actions. Each action receives a fresh teacher snapshot and a fresh hit generator with the same replicate entropy, giving common observation uniforms at the same local step. Position-dependent hit probabilities can produce different hits from the shared uniform. Newly generated sources and draws are evaluator-only evidence; they never enter the teacher's public interface. No original trajectory source is reused.

The horizon is exactly **H=2188**, including the forced first action. There is no teacher choice before that first move. Then follow the unchanged in-bounds analytic teacher. Source discovery after movement immediately returns the found sentinel with no odor draw. Incorporate every packet, including a success at H and a final nonterminal packet at H. Preserve the qualified filter's arithmetic on subsequent beliefs; do not insert normalization or a cycling stop. A failure or interruption is missing work, never a censored label. No learned model, native reset or native simulator call is required by the collector; this is explicit public-model continuation generation.

Let `A_i` be the number of eligible actions at anchor i, `C=16*sum(A_i)` the continuation count, `S` the total completed moves and `F` the number of found continuations. All 53x53 positions have 2, 3 or 4 eligible actions, so `C <= 384` and `S <= 2188*C <= 840192`. On successful completion, exact counts are:

| Operation | Count |
| --- | ---: |
| All-anchor validation snapshots before generation | 6 |
| Sampler validation snapshots / per-action snapshots | 6 / C |
| Total snapshot constructions | 12 + C, at most 396 |
| Source generators / source draws | 96 / 96 |
| Hit generators | C |
| Forced first actions / completed continuation records | C / C |
| Movement operations / public updates | S / S |
| Analytic teacher choices | S - C |
| Hit draws | S - F |
| Completed panels | 6 |
| Native constructions, native steps, optimizer or learned-model calls | 0 |

The current component emits `198 + C + 4*S - F` attempted/returned operation pairs plus forced-action, completed-record and panel-completion events. This is `402 + 4*C + 8*S - 2*F` component events, at most **6,723,474**, before the collector's separate validation, timing and artifact records. Reconcile actual counts from the complete ledger rather than infer work from successful labels alone. Preserve returned work and pending attempts on failure.

## Outputs, costs and interpretation

Retain all declared anchors, actions and replicates regardless of their costs. For each anchor/action publish capped mean moves, success and censoring fractions, the centered mean cost and every individual capped cost. For each eligible action pair retain replicate-matched differences, their mean and descriptive paired standard error. Report exact per-replicate ties, all-censored paired ties separately, and equal action means. Success on move H remains success even though its capped cost equals a censored rollout's cost. Report anchor eligibility and all denominators. Do not aggregate the six deliberately chosen anchors into a claimed population success rate or compare the collector families.

Retain the reducer's fixed per-anchor Hoeffding output at `familywise_alpha=0.05` if emitted, clearly labeled as a conditional bound under IID replicate-vector assumptions, not a calibrated winning label or a guarantee across anchors. The [analytical budget check](otto-teacher-cost-budget.md) already shows that N=16 has a very wide range-only bound. Neither that bound, paired SE, gap size nor censoring is a pass/fail gate. Do not increase replicates, shorten H, remove uncertain anchors or fit a model after seeing the labels.

Measure complete elapsed intervals for authentication, extraction/reconstruction, all-anchor validation, each sampler panel, reduction, serialization and worker completion. The panel interval includes generator/snapshot setup, analytic choices, categorical sampling, posterior updates, resource checks, logger callbacks, copying and durable output I/O; subtract none of these to claim cheaper label generation. Optional operation timings are nested diagnostics and must not be summed again into total wall time. Report output bytes and peak RSS. Source-generation, source hashing and any separate saved-output review remain explicitly counted work.

The provisional envelope is **1800 seconds, one numerical thread, 4 GiB RSS and 4 GiB output**, including preparation and all generation. It is **not yet a validated or admitted allocation**. Before freeze, use fabricated maximal serialized records to bound every attempt/return, per-panel result, duplicated draw evidence, anchor/kernel arrays and receipt at the maximum C/S above. Bind the byte projection and collector enforcement to the plan. If this envelope is insufficient, stop preparation and revise the still-prospective allocation before any real labels; never extend it mid-run. The approximately 6.7 million possible events make logger cost and size material.

Historical analytic-only call costs of roughly 0.175-0.289 milliseconds imply approximately 147-243 seconds for the maximum choice count, before the new sampler's other costs. This is a projection from prior recorded timings, not a measured rollout throughput or a promised upper bound. Current source selection, logging and full generation costs must be measured here.

Use an exclusive output and the existing suspend-inclusive supervisor. Persist durable attempts and returned operations, fixed anchor provenance and copied public arrays, all continuation records, all generated source/hit evidence, reductions, complete cost accounting and a completed or failed receipt. Preserve partial artifacts and the original terminal state. No retries, extra sampling, anchor replacement or silent result repair are authorized. A completed worker must subsequently join its original successful supervisor terminal; it cannot certify that future terminal itself. Independently check saved identities, streams, public updates, complete pair coverage, reductions and cost/count accounting under a separately declared saved-review budget before publishing a completed label panel. That review performs no fresh rollouts.

Only technical completeness is assessed. A subsequent [matched action-learning comparison](otto-action-cost-followup.md) needs its own frozen training and fresh autonomous-evaluation design. This pilot cannot establish useful supervision, policy competence, recurrence benefits or architectural novelty.
