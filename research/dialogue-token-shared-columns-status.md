# Shared state columns: implemented, timing not started

The implementation shares one fresh differentiable state-weight view within
each forward. It preserves every recurrent state calculation and monitor check.
The proposed benefit is less repeated gradient bookkeeping. **Its speed effect
is unmeasured.** This is an execution optimization, not a new learned architecture.

All **113 focused tests** passed: 66 model tests and 47 harness tests. The model
test file received an import-order-only lint fix afterward; the production source
did not change. The [full receipts](../output/dialogue-token-shared-columns-v1/preflight-01.json)
retain the original passing tests, failed lint and successful fix. Independent
source review found no material issue. Tests cover all four model arms, float32
and float64, outputs and gradients, actual monitoring inputs, two pending forward
graphs, changed weights, ordinary one-step behavior and absence of a stored view.

The [protocol](dialogue-token-shared-columns-protocol.md),
[plan](../output/dialogue-token-shared-columns-v1/protocol-01/plan.json), and all
71 source/protocol/fixture snapshots were published in commit `3e58d2f` before
timing. The public plan bytes were checked against the local frozen file.
The comparison retains all sixteen cases, four measured pairs per case, a 0.90
minimum-workload floor, 1.10 thresholds for larger cases, a 6 GiB RSS limit and
a 300-second command cap. Full-update timing includes the proposed change.

At the [recorded scheduling check](../output/dialogue-token-shared-columns-v1/execution-status-01.json),
another independent OpenJev training job was using the CPU. The timing attempt
has **not been created or launched**. No job was interrupted, and no benchmark
was run concurrently. Recheck host availability and the frozen source/runtime
identities before executing the single unchanged attempt. No retry or selective
retiming is authorized by this status note.

The [previous required-fill result](dialogue-token-required-fill-results.md)
passed 15/16 speed requirements; overall admission failed. There is no new speed result, accuracy
result, world-model advantage or full-training admission. A successful cost
screen would still require a separately declared [representative cost check](dialogue-representative-cost-design.md).
Earlier incomplete training remains closed and unscored.

- [Source design](dialogue-shared-state-columns-design.md)
- [Implementation](../src/openjev/research/dialogue_token_shared_columns.py)
- [Qualification harness](../scripts/qualify_dialogue_token_shared_columns.py)
