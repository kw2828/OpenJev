# Proposed 16-versus-64 continuation-target comparison

22 September 2026. **Design review only; not frozen, admitted or executed.** This follows the regression branch of the [prior decision note](otto-teacher-learning-next-decisions.md). It tests finite-sample target precision with the same ordinary controller, not a new architecture. The completed [matched study](otto-teacher-learning-results.md), its scientific failure, and both audit attempts remain unchanged. Root will decide whether to proceed after the separately authorized saved-noise diagnostic.

The review used source, protocols and closed accounting JSON only. It did not decode empirical arrays, generate labels, fit a model or evaluate a policy. The existing study binds 558 anchors, not 548: 398 have four actions, 107 have three and 53 have two, totaling 2,019 eligible anchor/action pairs.

## Fixed intervention and inheritance

Use every existing anchor, public belief, kernel, feature row, mask and episode weight unchanged. Authenticate the previous empirical plan `7c155b10b630ee6fb7b16e302accfc0883b4a39ff7a0cb6f94fe4b6ffb0bec90`, worker receipt `f55f19acc8b8b22ca936b78a1023835556b1ecb80b9bb84bbaa1b3dc661a0bf3`, original terminal `4dd9a826c8bfd0747e429ac0f3471b5e1614b03fe78f16f87e6a7834e19ebdf7` and successful V2 audit receipt `3ef858eb393a2896a48e9ec7bd9a9feb4600a3c4c3111b8ec3484829a56826d0`, including their complete source/input/payload closure. The inherited full run has 583 payloads. Hash its historical evaluation files as evidence, without decoding them as training input.

Read the original replicate 0-15 records from authenticated `panels.jsonl`, joined to each frozen anchor and closed panel-file descriptor. Generate **only replicate IDs 16-63**, using unchanged `sample_teacher_panel`, namespace `0x4F54544F`, seed **19000002**, original anchor IDs 0-557 and horizon 2188. Keep every eligible first action, tie and censor; no replacements or adaptive sampling. The full R64 panel contains 129,216 records, including 32,304 reused records. Join by `(anchor_id, replicate_id, first_action)` and reject gaps or duplicates. Compute R64 means from the combined integer step sums divided by 64, not an unweighted average of the 16- and 48-replicate means.

The R16 arm must retain the original continuation target arrays byte-for-byte. R64 uses eligible-action-centered mean capped moves and the **same original R16 global TRAIN scale**, read and pinned from `training-data.json` (`1.9131089760854865` in the closed study). Do not recompute an R64 RMS, normalize each panel, apply analytic range scaling, or change episode weights. Save raw means, float64 centered values and float32 casts. Use the qualified four-slot NumPy centering order and the V2 auditor's exact-recipe plus rational original-unit checks. The prior analytic-centering audit failure is preserved, not repaired in place.

## Exact added sampler allocation

For N=558 anchors, R=48 new replicates, C=96,912 new continuations, actual moves S and discoveries F, the unchanged sampler emits

`E = (4R + 3)N + 4C + 8S - 2F = 195*558 + 4C + 8S - 2F`.

The constant includes one validation snapshot and final panel event per invocation, plus two source operations per replicate. Forced first actions and completed records are included.

| New work | Count or bound |
| --- | ---: |
| Panel invocations / validation snapshots | 558 |
| Source generators / source draws, each | 26,784 |
| Action snapshots / hit generators, each | 96,912 |
| All snapshots, including validation | 97,470 |
| Movement / teacher update, each | S, at most 212,043,456 |
| Teacher choices | S - 96,912 |
| Hit draws | S - F |
| Maximum sampler events | 1,696,844,106 |
| Maximum raw event bytes at 512 bytes | 868,784,182,272, about 809.12 GiB |

Use the same buffered gzip level 1, durable panel-start/end boundaries and failed-incomplete-panel semantics. A compressed disk cap does not guarantee capacity for this raw maximum. Recheck the actual serializer projection for the larger operation IDs before freezing; never truncate oversized events or silently omit evidence.

## Smallest safe implementation surfaces

| Existing surface | Prospective adaptation |
| --- | --- |
| [Study runner](../scripts/study_otto_teacher_learning.py): `parent_closure`, `make_plan`, `bind`, `execute` | New version and files; bind the completed R16 study and V2 audit, freeze replica ranges, two target names, seeds and bounds. Preserve exclusive output, durable pending work, final source/input hashes and original supervisor joins. |
| Runner `setup`, `collect` | Load the same owned anchors and kernels. Read the old complete records once, sample only 16-63, retain the new event files separately, then persist both R16 and merged R64 reductions. Count reused and newly executed work separately. |
| Runner `training_data` | Reuse original features/masks/weights/R16 targets exactly. Build R64 continuation centering with the fixed original scale. **Do not call `targets(R16, R64, ...)`: its first argument receives analytic range normalization.** |
| [Training helper](../src/openjev/research/otto_teacher_learning.py): `_bundle`, `train_pair` | A narrow new helper copy should change only arm names and admitted seed constants, retaining the update body, paired initial exports, fresh Adam, epoch orders, checkpoint callbacks and parity. Existing arm/seed constants are hardcoded; do not monkeypatch them or mislabel R16 as analytic. Reuse unchanged model and regression-loss components. |
| Runner `checkpoint`, `train`, `evaluation_setup`, `episode`, `evaluate`, `aggregate` | Carry the reviewed behavior into a new runner with explicit R16/R64 identities. Preserve saved-checkpoint byte verification, complete timing, union of paired random prefixes, final updates and durable episode acknowledgment. Avoid subclassing old `Run` methods that resolve old module globals. |
| [V2 saved auditor](../scripts/audit_otto_teacher_learning_v2.py) | New audit version authenticates both old and added panels, checks disjoint replica ranges and exact integer merges, fixed scale and all checkpoints/work/metrics. Retain the declared shared-source and non-replayed score/filter/gradient limitations. |

Six fresh heads use three paired seeds, 80 epochs, batch 128, Adam 0.0003 and clip 5: **400 updates per fit, 2,400 total, 2,142,720 augmented row presentations**. Fixed final checkpoints and first-sixteen TRAIN export parity remain unchanged. No checkpoint or gain selection. A minimal new runner plus narrowly renamed helper is clearer than another generalized execution framework.

## Prospective seeds, evaluation and rule

Suggested fresh fit seeds: **20101, 20102, 20103**. Suggested evaluation ranges: **1050001-1050024**, **1060001-1060024**, **1070001-1070024** for lambda3/4/5. On this review, exact integer-token searches, including underscore-separated literals, found no mentions of these proposed values across **188** local OTTO research Markdown, scripts and research-module Python files. This is a finite source/declaration check, not proof of global or checkpoint-training independence. Before admission, root should bind a separate check of actual known episode ledgers and computed seed ranges. None of these seeds is reserved by this document.

Evaluate all six new heads and analytic control on the same 72 fresh cases: **504 resets/episodes**, no template calls, at most **1,102,752 native moves** and **945,216 learned readouts**. Preserve the eight three-hit blocks, known-kernel lambda5 shift, seven-arm rotation and complete 2188-step outcomes. Retain the previous **33 checks** with R64 as candidate and R16 as comparator: three analytic-control anchors, eighteen per-fit R64 competence checks, and twelve family-relative checks using the same 95%, 1.05, 0.95 and six-positive-block thresholds. Report every R16 fit's absolute performance too. Root must freeze these semantics separately before any new calls.

## Measured projection, costs and limits

The closed [collection accounting](../output/otto-teacher-learning-v1/run-01/collection.json) records 908,718 moves, 7,371,738 events and **342.199729 seconds**, including sampling, reduction, gzip, hashes and fsyncs. Six fits with paired setup took **7.896493 seconds**. Evaluation took **1,210.948968 seconds** over 677,488 moves. Worker wall was **1,565.370112 seconds**, peak RSS **454,443,008 bytes**, and all payloads totaled **1,042,905,604 bytes**; panel gzip files contributed **182,768,981 bytes**.

At the same per-replicate workload, 48 added replicates project to about **1,026.60 seconds** of collection and **548,306,943 bytes** of new panels. Replacing the old collection phase with this estimate gives about **2,249.77 seconds, or 37.5 minutes**, and **1,408,443,566 bytes** of new payloads before added lineage/merge metadata. These are simple point estimates, not admission guarantees. At the old evaluation time per move, all 504 episodes reaching H would cost about 1,971 seconds. At the old collection time per move, the theoretical added-label maximum would cost about **79,850 seconds**, exceeding the previously proposed 10,800-second ceiling. Rare long continuations, filesystem cost and different learned paths can therefore cause a genuine bounded-run failure.

The earlier proposal of **10,800 seconds, CPU1, 4 GiB RSS and 32 GiB output** appears plausible for a prospective single run, but requires root's explicit budget freeze and retains failure at either cap. No guarantee follows from observed compression or the all-found R16 sample. Auditing and publication have separate paid budgets.

Charge added generation and merging to R64, allocated across its three fits; report old R16 labeling as common inherited work for both methods. Show current physical cost separately so historical cost is not double-counted. Preserve actual head restore/72 and model-module setup/432, actor/features/scoring/filtering costs, native costs and single-pass timing limitations. Reused modules are not a cold-start benchmark.

## What this would establish

The two label sets are nested, so their errors are correlated. More replicates can improve estimation of the same capped analytic-teacher continuation value, not remove teacher mismatch, censoring bias relative to uncapped cost, distribution shift or representation limits. Three fitting seeds measure optimization variation, not three independently sampled label datasets. Sixty-four samples do not guarantee correct action ranking, particularly near ties; retain all precision diagnostics without filtering rows or choosing a favorable replicate budget.

Better target stability without better fresh control would reject sample precision as a sufficient practical remedy for this recipe. Improved control still needs the absolute analytic competence anchor and full paid-cost comparison. Neither outcome establishes architectural novelty, optimal values or general robotic competence. No additional collection, training or evaluation is admitted here.
