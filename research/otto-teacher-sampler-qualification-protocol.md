# Public teacher sampler: strict native generation qualification

22 September 2026. Prospective mechanical qualification only. This document fixes the cases and acceptance rule before execution; it does not authorize execution, real TRAIN anchors, label collection, fitting or evaluation. Passing the existing synthetic component tests for the rollout sampler, snapshot initializer and cost reducer is a prerequisite, as is independent source review of the new qualifier. Bind those component sources/tests, their test receipts, the qualifier and this protocol in a frozen plan. Additional harness tests are not required merely to repeat its implementation.

The question is whether the new source-conditioned kernel sampler reproduces the existing seeded native observation generator, including exact categorical boundaries and source discovery. This is transition/generation qualification, not policy efficacy, a statistical distribution test or empirical-study admission.

## Reference and isolation

Use the existing `.venv/bin/python` under the complete Python/package manifest inherited from the frozen spatial-control plan. Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` and `NUMEXPR_NUM_THREADS` to `1` before numerical imports. Import only the authenticated native `sourcetracking.py`, NumPy/SciPy and the public sampler/snapshot dependencies. Do not import TensorFlow, Torch, a learned model, historical checkpoints or empirical trajectory arrays. No runtime installation or upgrade is allowed.

Construct exactly three `seeded_environment(SourceTracking, seed, config, initial_hit=1)` objects: sensing lengths 3, 4 and 5, in that order, with mechanical seeds 0, 1 and 2. Every configuration has `Ndim=2`, `R_dt=2`, `Ngrid=53`, `Nhits=4`, `norm_Poisson='Euclidean'`, `draw_source=True`. These are integration identifiers, not fresh evaluation seeds. Save the three freshly constructed float64 kernels, unchanged, as shared public inputs to the helper. Require finite nonnegative entries, shape `(4,107,107)` and exactly zero origin. Each construction consumes one native source draw and no initial-hit draw. Retain those initialization draws; none supplies a fixture source or a continuation label.

For each mechanical step, explicitly install evaluator-owned native state rather than call `restart`: position, prescribed source, copied anchor belief, nonterminal observation, fresh hit map, cumulative-hit counter and the native previous-position/stuck bookkeeping. Preserve RNG draw counters/logs across installations. Supply only the copied public packet, belief and kernel to a fresh `TeacherSnapshot`. The anchor has absolute step 7, hit 1, `done=False` and exactly the in-bounds action list. At position `q`, form the anchor from `raw[x,y] = 1 + ((17*x + 29*y + 7*x*y) % 97)`, set `raw[q]=0`, cast to float64 and divide by its float64 array sum. Do not assimilate hit 1 again.

Replace the evaluator's `_public_rngs['hit']` with a one-value uniform tape. An evaluator-only `SourceTracking` subclass must also install a transparent logging wrapper around the already-bound seeded adapter `_draw` before invoking the native `SourceTracking.__init__`, so initialization source draws are journaled too. Capture the original bound `_draw`, emit a durable attempt, delegate to it exactly once with unchanged arguments, then emit its returned result and original draw evidence before returning unchanged. Do not change its probability arithmetic, RNG access, counters, log or selected index, and do not perform an extra draw. The adapter's `_execute_action` and native `step`, movement and filtering methods remain unchanged. An extra uniform request must fail. This controlled substitution tests inverse-CDF mechanics; it does not qualify the sampler's PCG64 namespace or independent replicate streams, which require their own synthetic tests.

## Fixed 408 native steps

For each sensing length, process the following twelve nonfound geometries in table order. Actions 0/1 change the first coordinate by -1/+1; 2/3 change the second coordinate by -1/+1. Boundary moves clamp to the current position.

| Fixture | Anchor position | Action | Prescribed evaluator source |
| --- | --- | ---: | --- |
| 0 | (26,26) | 0 | (24,26) |
| 1 | (26,26) | 1 | (29,27) |
| 2 | (26,26) | 2 | (52,52) |
| 3 | (26,26) | 3 | (0,0) |
| 4 | (0,0) | 0 | (52,52) |
| 5 | (0,0) | 2 | (1,2) |
| 6 | (52,52) | 1 | (0,0) |
| 7 | (52,52) | 3 | (51,52) |
| 8 | (0,0) | 1 | (0,52) |
| 9 | (0,0) | 3 | (2,1) |
| 10 | (52,52) | 0 | (52,0) |
| 11 | (52,52) | 2 | (50,51) |

Run eleven independently reinstalled one-step probes per geometry. Probe zero uses uniform `0.0`; retain the original adapter's recorded probability vector. Construct its reference CDF using explicit left-to-right float64 scalar additions, starting from zero, then divide each cumulative entry by the final cumulative mass. Do not use Python `sum`, which may use compensated summation. Let `U = nextafter(float64(1), float64(0))` and let `b0,b1,b2` be the first three normalized boundaries. Remaining probes, in order, are `U`, then for each boundary `b`: `min(U,nextafter(b,0))`, `min(U,b)`, `min(U,nextafter(b,1))`. This bounds uniform probes only; it never changes probabilities. Keep repeated probes, including zero bins or boundaries equal to one. The first native probe is part of the eleven, not an extra step.

For each probe at moved position `q`, compare the helper's untouched vector `kernel[:,53+sx-qx,53+sy-qy]` against the original native vector, and compare both normalized CDFs and their final unnormalized cumulative mass. Independently select the first bin whose normalized cumulative value is strictly greater than the uniform, using a scalar loop. Require the same supported category as the helper and native adapter. Save every original value and exact-equality decision.

This gives `3 * 12 * 11 = 396` nonfound native steps. The four blocked directions are deliberate native transport fixtures. They do not authorize the public continuation sampler to accept blocked first actions; its full panel must still reject them before sampling.

After the nonfound geometries for each regime, run four immediate-found fixtures in action order 0,1,2,3 from `(26,26)`, with source equal to that action's successor. Install a tape that rejects any hit draw. Require native return `(-2,1,True)`, terminal public step 8, no valid actions and a point-mass posterior at the successor. The sampler's movement/found helper must agree; no kernel-origin categorical call is valid. Its full-continuation found-before-odor ordering is separately covered by the prerequisite synthetic tests. This gives twelve found steps, for **408 native steps total**. Every fixture is one forced continuation move, whether found or nonfound; there is no teacher choice or trajectory-level early stopping.

Update each fresh public snapshot with the returned action and packet, including the last nonterminal observation. Require exact native/public belief and public-position equality. Preserve the terminal sentinel as a terminal operation, never as a likelihood index.

## Direct categorical and source checks

On the first native object only, call the logging-wrapped adapter `_draw('hit', probabilities)`, which delegates to the unchanged original method, for four vectors in this order: `(0,.25,0,.75)`, `(.25,0,.75,0)`, `(0,0,0,1)`, `(1,0,0,0)`. Each receives the same eleven-probe rule, independently computed from its original vector. Compare the public categorical helper and scalar oracle. These are **44 additional hit draws and zero native steps**; zero-probability categories must never be selected.

Also make nine direct native `_draw_a_source` calls on the first object: three public belief fixtures, each with uniforms `0`, `.5`, `U`. At anchor `(26,26)`, use (1) unit mass at `(0,0)`, (2) mass `.25` at `(0,52)` and `.75` at `(52,0)`, and (3) the asymmetric anchor defined above. Replace only the source uniform tape, assign the declared belief, and compare the actual source helper and native source selection with an independent C-order flat-index calculation. Require exact probability/CDF mass and source identity; the source cannot be the zero-mass current cell. These checks consume no native steps or public updates.

Complete successful work is therefore:

| Operation | Exact count |
| --- | ---: |
| Native constructions, each including its normal reset | 3 |
| Additional native restart calls | 0 |
| Evaluator state installations / native steps | 408 / 408 |
| Fresh public snapshots / public updates | 408 / 408 |
| Hit draws: 396 transition + 44 direct | 440 |
| Source draws: 3 initialization + 9 direct | 12 |
| Initial-hit draws / teacher choices / learned-model calls | 0 / 0 / 0 |

Journal helper operations separately from native counts. All **452 native draws** must have their own attempted/returned records, including the three initialization source draws. Draw events nest inside construction, source-selection or step events where applicable; a direct categorical call is counted once, not again as a second native draw. Preserve the operation nesting on failure: if a draw returns and later step filtering raises, that completed draw remains returned while the enclosing step remains pending. If the draw or return publication fails, retain its unresolved attempt; never fabricate a return or retry. Nested intervals are not disjoint time and must not be added as separate total work. Do not call `sample_teacher_panel` on these fixtures: full-panel RNG coupling, continuation choice and horizon accounting are synthetic-component obligations and remain distinct from this native one-step test.

## Strict acceptance, bounds and evidence

Require original probability-vector, CDF, selected category, source, packet and posterior equality with no numerical tolerance. In particular, native sampled generation computes the tail with `maximum(0,1-sum)`, whereas native grid-kernel construction uses the raw remainder; scalar and grid Poisson calculations also execute separately. Any discrepancy is a failed qualification. Do not clip the helper vector, change the CDF rule, drop probes, relax equality or claim native equivalence because differences are small. Retain every planned comparison where execution remains possible; record numerical mismatches as failed rows. Structural or resource failures retain the partial ledger and fail completion. A corrected mechanism requires a separately frozen version and preserves this attempt.

Use the existing suspend-inclusive supervisor with **120 seconds, 4 GiB peak RSS and 64 MiB total output**, one numerical thread. The parent timeout is also at most 120 seconds. Create an exclusive output directory and started record before native construction. Persist attempted/returned construction, draw, step and public-update events, every fixture/probe record, unchanged shared kernels/anchors, both native/public resulting beliefs, summary and completed or failed receipt. A completed worker receipt requires exact counts, no pending operations, all comparisons passing and unchanged source/runtime identities. It binds the actual supervisor launch, command and deadline and records `requires_successful_original_supervisor=true`; the still-running worker cannot authenticate its future parent terminal. Overall qualification acceptance additionally requires the original successful parent terminal, no timeout, an absent process group and exact launch/worker/terminal joins. Recorded source/step costs are mechanical work, not gameplay or label-generation throughput. No retry is authorized by this protocol.

## Source identity

The review used checkout `011774c1c4b9bd7247805146a51a9ab8d987239f`. The frozen plan must hash the complete import closure, sampler and existing component tests, new qualifier, this protocol and prerequisite test receipts before execution. These reviewed reference bytes must remain unchanged:

| Reference | SHA-256 |
| --- | --- |
| `src/openjev/research/otto_public.py` | `438631a18005493e0cafa0777315e2158cac97cc7d0e71ad9fd6402284615b3d` |
| `tmp/otto-source-review-01/isotropic/classes/sourcetracking.py` | `1057f129fa8a3249a7297a9ef2c15059d37430b12e42a2f1d43dfcdffe3e76f1` |
| `src/openjev/research/otto_teacher_snapshot.py` | `c318bd1d743b1589153c360c4bd1a2afdec5df34e7b769be0b2c3a26a87e9c58` |
| `src/openjev/research/otto_released_policy.py` | `8abac34fef2a6517d43209d5ca27514633363248f66a249e2f4d6615aa0a0d45` |
| `src/openjev/research/otto_reference_control.py` | `21e5f6ece09426d26c4a3214d95f0f4803a8efd4bacb9ca347c1d896407c5893` |
| `src/openjev/research/suspend_clock.py` | `cac9077db3c66f5ef43d9412b732f5b869c1004c4bac98589816780c120f6124` |
| `scripts/supervise_dialogue_observation_v2.py` | `610a2fcd2d4d55eb35bce92e61942d5169491dee9d05e2f47f65e211fdf94144` |

Prior evidence: [public snapshot engineering](otto-teacher-snapshot-component.md), [native/public integration protocol](otto-released-native-qualification-protocol.md), and [conditional action-cost proposal](otto-action-cost-followup.md). Their existing receipts and frozen sources are not changed by this qualification.
