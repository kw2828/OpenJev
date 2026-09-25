# Native robot comparison: preserved failed result

**Benchmark FAILED. Independent saved-output evidence audit PASS. No timing was collected.**

All 30 selected fit/DEV cases failed the fixed numerical parity requirement. Every case is retained for both batch sizes 22 and 1, with standardized and physical forecasts and both complete native final-state comparisons. The package includes all 30 opaque forecast archives, 30 per-case records, the original aggregate results, and the original successful audit of the failed result. The audit independently checked 420 saved arrays and 240 comparisons; it did not rerun inference.

The numerical gate stopped execution before warmups or timed pairs. This provides no latency or speedup result and does not revise the parent study's quality rule. The earlier 110-test fabricated native qualification passed; those fixtures did not establish parity for these trained weights.

- `study/`: complete original benchmark folder, copied byte for byte.
- `audit/`: original independent audit and its manifest.
- `processes/`: original run launch, failed run closure/log and successful audit closure/log.
- `registration/`: frozen benchmark registration and protocol.
- `benchmark-fixture/` and `audit-fixture/`: original 22-test and 24-test qualification receipts/logs.
- `sources/`: the exact benchmark, auditor and corresponding tests.
- `provenance.json`: source-to-copy byte identities and scope.
- `manifest.json`: every published file except this manifest, with SHA-256 and byte count.

The [native qualification package](../robot-native-qualification-results/README.md) preserves both native qualification attempts and the source-derived repair. Its manifest and provenance hashes are recorded here as external references. Compiled libraries, source measurement arrays, targets and trained checkpoints are not included. Original dependency paths and hashes remain in the preserved receipts. Re-executing the benchmark requires the separately retained parent study and its measurement permissions; this package alone supports inspection of the saved parity evidence.

Publication performed only JSON/metadata checks and opaque byte copies, followed by independent byte verification. It did not decode forecast arrays, load models, refit, score targets or collect new timings.
