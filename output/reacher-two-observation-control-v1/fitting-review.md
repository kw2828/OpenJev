# Independent fitting adapter review

Static review found no material blocker in the fit serialization adapter. The root reviewer read the complete source and the success, binding, partial-update, I/O failure and deadline tests. No tests, model calls or training were run by the reviewer.

- Source SHA256: `dede26dda3d1dee88a3e2df6a6d9834190b99baf8240166477ed3a2a45cd5857`.
- Test SHA256: `03fff080c14db7af094d2a707dc66a5d5381a8b3a9041d30b330a6e12c1def8b`.
- Author report: 33 fake-trainer tests passed in 0.57 seconds; Ruff clean.

The adapter requires the caller's original tensor, complete sealed-order and public-data hashes before constructing the trainer. It creates an exclusive output directory and generates no initial weights, orders or scientific streams. Full-order hashing correctly includes the seal rather than substituting the embedded integrity digest.

Every returned training log is checked against its update, batch indices and prior hash, then written without filtering fields and flushed with fsync. Final weights must agree with the checkpoint. The checkpoint binds the actual model configuration, settings, initial tensors, data, orders, provenance, sources, runtime, complete counters and final log chain. The success manifest contains the four payload files, with a fifth completion receipt. Schema and optimizer mathematics remain the separate independent training auditor's responsibility.

Failure handling distinguishes optimizer steps completed from log rows durably flushed. It preserves available partial checkpoints and returned-but-unflushed rows without inventing logs for a call that raised. Export or preservation errors are attached to the original exception. A failed outer attempt is terminal even when its partial trainer could technically resume. A completion write that fails or crosses the cap is invalidated and retained; any filesystem failure to invalidate is also reported and must not be accepted by the enclosing study.

Reported fit wall time includes construction, updates, validation, payload writes and hashes. Nested timers are identified as overlapping. The completion receipt's own write is excluded from its timestamp but followed by a deadline check. The outer study still owns full input/source authentication, exact pair/name membership, symlink rejection, aggregate costs, all-fit boundaries and final cap enforcement.

These tests establish serialization and failure mechanics with fake trainers. Actual trainer integration, full-study rehearsal and capacity measurement remain necessary before scientific fitting. No empirical effectiveness follows from this component review.
