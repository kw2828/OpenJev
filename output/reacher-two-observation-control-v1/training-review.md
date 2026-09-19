# Independent two-observation trainer review

No material blocker found in the bounded read-only source review. This is an engineering component review, not a scientific result, protocol freeze, study authorization or confirmation of production readiness.

## Exact reviewed sources

- `src/openjev/research/reacher_two_observation_training.py`: SHA-256 `89d4f40d1d0730a7d48ffb775092d50aa50fd8507d4e491ccba746fe51d33791`.
- `tests/test_reacher_two_observation_training.py`: SHA-256 `3455bd892fc9dbc20b899b49c8de283539714829cb3626d993d13cbaa09bf512`.

The review also read the unchanged parent `reacher_memory_training.py` and its `sequence_loss`/anchor-work dependencies, and the new two-observation model's reconstruction and accounting contracts. No source was edited. The reviewer made no model, optimizer, native-environment, training or network calls and did not rerun the synthetic tests. The author's reported 63 passing synthetic tests in 1.57 seconds and clean Ruff are separate author-supplied validation claims; this review assessed their source coverage.

## Pairing and actual-class identity

Construction admits exactly `TrainingSettings` and `TwoObservationHistoryGRUWorldModel`, then verifies scalar settings, complete model configuration, CPU float32 finite tensors, train mode, absence of registered buffers, named parameter identity and exact Adam parameter ownership/order. It does not modify a frozen model registry or admit the original residual-GRU class through a shape-only fallback.

Initial named tensors, the complete sealed order payload and the complete public training tensor mapping each require independent expected hashes. Inputs are cloned before use. Initialization loading checks exact tensor keys, dimensions, device, dtype and finiteness. The temporary constructor is isolated at seed410 and all its model tensors are overwritten by the supplied initial state. The setup scope restores ambient CPU RNG even on failure.

A zero-update resume additionally requires student tensors to equal the original initial tensor hash, no prior log chain and no last attempt. Resume after updates restores the exact new class rather than relabeling an old fitted model.

The component deliberately does not prove that caller-supplied tensors are historically original initial weights. The future runner must authenticate original paired initialization files and the corresponding data/order/source/runtime receipts. Supplying an independently verified digest is essential; tensor-schema similarity or self-reported provenance is insufficient. The module documents this boundary clearly.

## Objective, optimizer and restoration

Training dispatches to the unchanged parent `MemoryTrainer.train_next`, which calls the existing `sequence_loss` with the existing anchor keyword settings. The objective remains masked angle prediction, native reward MSE, existing valid-root rollout terms and the existing KL interface. There is no added auxiliary loss, teacher, detach policy or alternative optimizer in this adapter.

Adam settings are imported from the existing recipe, including its exact option fieldset. Live checks bind parameter ownership and group order. Resume authenticates an externally supplied canonical checkpoint digest, exact checkpoint membership, model semantics, original initialization/data/order bindings and source/runtime/provenance. It checks optimizer slot coverage, parameter IDs, scalar step counts, finite CPU float32 moment shapes, nonnegative second moments, and equality after loading to reject silent coercion.

Successful-update and optimizer-step counts must agree and remain within the configured total. The saved next cursor is recomputed from those counts. The final completed attempt must identify the prior cursor and correct partial-batch work. Failed checkpoints cannot resume. Save uses exclusive file creation. A future file-based runner must also authenticate checkpoint file bytes; the current API's expected checkpoint digest is the canonical sealed-body digest.

## Public data, order and RNG boundaries

Only public packets, issued commands and native reward labels are admitted. Shape/device/dtype/finiteness, initially visible observations, masked missing angle channels, clipped issued actions, static public targets and integer-clock-derived ages are checked. The complete dataset is scanned before model construction to reject unsupported twelve-packet suffix overflow, including gaps that would discard the required older observation.

The immutable full order is independently regenerated with a local seeded CPU generator, checking exact permutations, shape/dtype and initial/final RNG states. Its externally bound canonical hash also includes role/version metadata. The logical current order RNG is derived from the update cursor using one consumed permutation when an epoch's first batch has occurred; it is not an unbound ambient generator. The cached RNG-state sequence is included in the live metadata binding.

Each update checks the complete training-data and order hashes before forward execution. The tests include externally mismatched bindings and resealed-but-semantically-invalid cursor, order, RNG, initialization, optimizer and configuration payloads. CPU ambient RNG is checked around deadline callbacks and restored by the surrounding isolated scope.

## Failures and paid work

Each real assimilation reconstructs all twelve observation updates and eleven complete transitions, including masked startup padding. Anchor-work counts include causal prefix advances and every valid-length open-loop rollout call. The adapter converts those interface samples into full reconstruction work, counts GRUCell and Linear samples through actual forward hooks, and requires observed calls to equal the analytical accounting. The tests cover a partial final minibatch and epoch-boundary resume. The component explicitly reports that counts are not total FLOPs and that compute is not matched.

Forward/backward/clipping/optimizer timings remain nested within batch/training wall time; final cleanup and bookkeeping remain charged in total training time. Interrupted work records completed neural calls and wall time, not an invented count of an interrupted internal operation. Gradients and temporary hooks are cleared. Any exception marks the trainer terminal, and subsequent training or restoration from its failed checkpoint is rejected. If a failure occurs after an optimizer update, the recorded counters and failed state preserve that paid attempt rather than claiming rollback or silently retrying.

The synthetic suite's central split/resume fixture compares subsequent losses, indices, model tensors, Adam state, cursor and RNG identity across an epoch boundary. Separate tests inspect pre-forward deadline/interrupt/unexpected-RNG failures, failure after paid forward without an optimizer step, original-weight alias isolation, unsupported history, and exclusive checkpoint saving. Those tests provide a meaningful engineering design; they do not establish full-scale training effectiveness or causal scientific superiority.

## Remaining boundary

Before a scored experiment, the enclosing runner still needs authenticated historical initialization/order/data provenance, complete source and RNG exclusions, explicit compute and storage caps, terminal failure preservation, and an independent saved-output audit. The twelve/eleven reconstruction is deliberately more work than a one-step persistent update; equal update counts alone would not create a compute-matched comparison. The completed geometry-memory study's passed gate remains a separate scientific result and is not evidence that this new two-observation component is effective.
