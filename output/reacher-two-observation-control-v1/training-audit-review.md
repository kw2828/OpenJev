# Independent static review of two-observation training audit

Reviewed on 2026-09-18. No remaining material blocker was found in the corrected component and its focused tests. This is a source review, not a scientific run or an independent test execution.

## Reviewed versions

| File | SHA256 |
| --- | --- |
| `src/openjev/research/reacher_two_observation_training_audit.py` | `57e17139ae90df588477341ec28803d4f2d318c23b8f1842daaeff550e998ee0` |
| `tests/test_reacher_two_observation_training_audit.py` | `1540f1593d05381006e68daa850949ea5722720a621eba951f4f4f67b852beab` |

The author reports 86 handbuilt synthetic tests passing in 1.26 seconds and Ruff clean. The reviewer did not rerun tests, construct a model or optimizer, call a native environment, or read production data or outcomes.

## Compatibility and evidence checks

The auditor's exact checkpoint, log, anchor-metric and optimizer schemas agree with `reacher_two_observation_training.py` and the unchanged memory/world-model training conventions. It checks the actual class and configuration, caller-supplied initialization/data/order/provenance/source/runtime bindings, complete `ceil(episodes / batch_size)` coverage including a short final batch, saved episode indices, final cursor, and update log hash chain. The checkpoint expectation refers to the sealed body digest; the order expectation refers to the canonical hash of the entire sealed order payload.

The isolated CPU generator reconstructs the supplied historical minibatch permutations and intermediate epoch RNG identities. This is verification of the declared old order, not allocation of a fresh scientific stream or mutation of the ambient RNG. Adam parameter names, IDs, settings, tensor shapes, finite moments, nonnegative second moments and final step counters match the trainer's actual ownership and recipe.

Public data validation covers the permitted packet/command/reward fields, initial visibility, static targets, valid ages, sanitized missing angles, command bounds and every retained-history overflow boundary. Saved mask counts are independently derived from the actual public batches. Loss composition, deterministic zero KL, clipping flags, work formulas, completed hook counters, final attempt and timing relationships are checked. The accounting includes all 12 observation-update and 11 full transition calls for each real reconstruction, including padding, plus the unchanged anchor rollout work. Optional deployment weights must equal the checkpoint's final student tensors.

Two narrow compatibility issues identified during review are corrected in the hashes above:

1. Reconstructed model configuration now normalizes `dt` and `noise_std` to float, matching the actual constructors even when caller settings contain integers.
2. The clipping flag uses the trainer's float32 tensor-versus-scalar comparison, rather than a Python double comparison that can disagree near a threshold rounding boundary.

The added handbuilt tests exercise a full integer-settings checkpoint and the `nextafter(10, 0)` clipping edge. Neither correction changes the intended `.02`, `.05`, and `10.0` study settings.

## Limits and integration obligations

The component does not load files or import a learned model, trainer or optimizer. Saved losses, gradient norms, moments and hook counts are evidence supplied to it. Schema and arithmetic checks do not establish that a neural forward/backward computation or numerical Adam transition actually produced those values. Its output states these limits and reports zero new model, optimizer and native calls.

The caller must authenticate file bytes and the original paired initialization, sealed order, data, source/runtime and pair provenance. A matching tensor digest alone does not establish that an initialization came from the required original file. This component verifies one complete fit; the future enclosing audit must enforce cross-fit pairing, exact artifact membership, scientific source and stream lineage, native control verification and the prospective study's qualification rules. A passing component audit is not launch authority or an efficacy result.
