# Shared state columns: qualification completed and failed

The frozen single attempt completed with all sixteen numerical checks passing
and **15/16 speed requirements passing**. Medium candidate/scalar reached
**1.088323x**, below its fixed **1.10x** threshold. Overall admission failed.
The [full result and figure](dialogue-token-shared-columns-results.md) include
every measured pair and an independent saved-output audit. No training restart
or new task-accuracy result follows.

The run followed **113 passing focused tests** and the 71-source freeze
published in commit `3e58d2f`. The model test file received an import-order-only
lint fix after its tests; production source did not change. The
[preflight receipt](../output/dialogue-token-shared-columns-v1/preflight-01.json)
retains the original test and lint outcomes.

The earlier [scheduling receipt](../output/dialogue-token-shared-columns-v1/execution-status-01.json)
is preserved unchanged. It records the true pre-run waiting state, when another
heavy CPU job was active. That job finished normally before this qualification
started. The run was neither retried nor selectively retimed.

- [Results](dialogue-token-shared-columns-results.md)
- [Original source design](dialogue-shared-state-columns-design.md)
- [Frozen protocol](dialogue-token-shared-columns-protocol.md)
- [Implementation](../src/openjev/research/dialogue_token_shared_columns.py)
- [Conditional representative-cost design, eligibility not met](dialogue-representative-cost-design.md)
