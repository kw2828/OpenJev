# Durable logging qualification

**Passed the predeclared engineering rule: 5.103x median speedup, faster in all six paired blocks, with byte-identical output.** Both implementations flushed and synced every event before acknowledging it. This result concerns fabricated logging work, not neural inference, learning quality or novel architecture.

The [previous query-value pilot](otto-query-advantage-results.md) timed out with logging accounting for 65.36% of measured elapsed time inside its completed panels. This qualification tested an implementation change before allocating any new scientific work.

## All paired blocks

Each arm wrote the same 4,096 prebuilt fabricated records. Implementation order alternated by block. Total time includes logger initialization, all appends, final close and reconciliation; trace construction and directory creation are excluded. The native Python runtime and per-event fsync contract stayed unchanged.

| Block | Order | Original seconds | New seconds | Speedup |
|---|---|---:|---:|---:|
| 1 | Original, new | 1.201151 | 0.238877 | 5.028x |
| 2 | New, original | 1.106308 | 0.213668 | 5.178x |
| 3 | Original, new | 1.240268 | 0.238186 | 5.207x |
| 4 | New, original | 1.029290 | 0.229015 | 4.494x |
| 5 | Original, new | 1.282490 | 0.232426 | 5.518x |
| 6 | New, original | 1.088350 | 0.229245 | 4.748x |

All twelve journals are exactly 968,282 bytes with the same SHA256. Each arm recorded 4,096 append acknowledgments and 4,096 append fsyncs. The new implementation performs one additional fsync at final close, included in its total. The original supervisor completed successfully in **8.555973 seconds**; the worker recorded **8.434102 seconds**. These are whole-qualification times, distinct from the timed per-arm intervals.

The frozen rule required a median paired speedup of at least 2.0 and improvement in at least five of six blocks, alongside identical output and clean closure. All requirements passed. The [saved independent admission](../output/otto-query-advantage-v2/logging-admission-01.json) verified original process closure, all payloads, source hashes, every fabricated record, timing reductions and the qualification evidence. It contains 1,341 checks and makes no model-quality claim.

## Implementation and scope

The new journal encodes each event once, retains open streams, and tracks exact byte allocations instead of scanning the output directory on every event. It checks file identity and size at explicit boundaries. Failed or uncertain writes poison the writer, retain pending evidence and cannot be silently resumed. External artifacts require reservations, and failure receipts have reserved space. Final artifact hashes remain the runner's responsibility.

Nineteen journal tests cover exact bytes, resource limits and injected I/O failures. Thirty-three additional fabricated tests cover runner integration and audit launch contracts. The unchanged paired sampler and reducer retain their earlier 24-test qualification. The final benchmark lifecycle patch was separately source-reviewed and statically checked; the 52 new tests do not claim to cover that patch. The completed bounded benchmark supplies its actual execution evidence. A failed metadata-only qualification aggregation caused by an older log-field format is preserved; correcting that lookup did not rerun tests or scientific work.

This is one local, alternating-order engineering comparison. It does not establish performance on other filesystems, long runs or actual simulation workloads. A fresh scientific cohort may have more anchors or longer branches. The [V2 scientific protocol](otto-query-advantage-v2-protocol.md) therefore retains its separate 900-second cap and all eleven scientific conditions. The incomplete V1 pilot remains unchanged.

## Evidence

[Qualification protocol](otto-query-logging-qualification-protocol.md) · [Frozen workload and sources](../output/otto-query-advantage-v2/logging-plan-01.json) · [All timings and payload hashes](../output/otto-query-advantage-v2/logging-01/receipt.json) · [Original supervisor](../output/otto-query-advantage-v2/logging-supervision-01.terminal.json).

| Record | SHA256 |
|---|---|
| Plan | `896389bae7ff5ffdd5a538be523b22d07827f4bd31a8cfad9d0cbafcf206a7eb` |
| Worker receipt | `9486e1ee975e2037ff0637bdc476f5e74d634bda81d441e6e388f228d769ec2c` |
| Original terminal | `449bbc83949c11d1f46ebe2c58bc06e23ec84ce36dbffde6e2e1e6004115f408` |
| Independent admission | `95c37a7bb33885aed77eabf6586aa49a4dae8209e0628f5c1bf1e1bc23962c7f` |
