# Reacher memory whole-pipeline engineering rehearsal

**The reduced rehearsal completed all 12 fits and all 51 controller rows, and its saved-output audit replayed 2,850 native transitions with zero numerical discrepancy.** This verifies engineering integration on the explicit fixture. It is not evidence that any memory architecture improves control.

The attempt is bound to plan SHA `a3077c2e88595df0d4795bda0fa63fe149339fd2e995d5ef143d1dc0d4dd0e0a` and the separate `reacher-memory-engineering-whole-tree-v1` namespace. The [fixture description](fixture-description.json) explicitly disables production-lineage authentication and confirms no inherited checkpoint calls. The [source snapshot](source-snapshot/tests/reacher_memory_fixture.py) preserves the exact fixture boundary alongside all 58 plan-bound files and the exact [launcher](source-snapshot/rehearse.py).

| Coverage | Completed engineering fixture |
|---|---|
| Models | Four architectures, all three initialization pairs: 12 fresh fits |
| Training | Four fresh engineering episodes, one epoch, batch size two, two updates per fit |
| Model widths | GRU 4; packet MLP 7 |
| Evaluation | Two prediction episodes; one case for each of 51 controller rows |
| Panels | Full, ordinary blackout, shifted blackout |
| Native episodes | 50 steps; planning horizon 12 |
| Saved native replay | 200 training + 100 prediction + 2,550 control transitions |
| Audit calls | Zero learned-model calls, zero policy calls, zero new fits |
| Ordering | All 12 actual model classes restored before fresh evaluation |

Execution took 23.266610 seconds, the recorded audit validation took 2.732762 seconds, and the launcher recorded 26.644062 seconds overall. These are nested wall-time measurements from the reduced fixture, not a forecast for full-size training or an isolated performance comparison. Detailed accounting is preserved in [summary.json](summary.json). The fixture plan recorded its runtime/package and native-source identities during the rehearsal; no new hardware observation is represented as execution-time evidence.

The only audit substitutions were historical lineage and the inherited-training cohort identity, replaced by explicit fixture metadata and four fresh engineering records. Their 200 transitions were still replayed. Source/runtime bindings, complete artifact membership, restoration and phase ordering, public state/history buffers, paired streams, CEM reconstruction, native replay and recorded cost checks remained enabled. The saved audit does not independently repeat neural forward passes or optimizer transitions.

The complete raw engineering audit, including its gate output and toy-model utility numbers, is retained losslessly in the archive. **Those values are non-scientific fixture outputs and must not be cited as effectiveness, a passed research gate, or evidence for a winning architecture.** The full-size scored protocol is a separate artifact and is not authorized or executed by this package.

The root task reported **300 tests passed in 12.11 seconds and Ruff clean**. [That report](root-observed-checks.json) is attributed to the root's observation; packaging did not rerun tests and does not include an original test transcript.

The local verification archive is `output/reacher-memory-rehearsal-v1/publication-v1/reacher-memory-rehearsal-v1.tar.gz`. It includes every file in closed `attempt-01`, plus all matching source snapshots. [The manifest](bundle-manifest.json) binds every path, length and SHA-256, and every archive member was reopened and checked. Checkpoints and raw trace binaries remain outside this evidence directory. [The packaging receipt](receipt.json) links the original plan/audit/execution hashes, source snapshot and archive. No upload is claimed. External historical dependencies referenced by the fixture are not recursively bundled; the package does not claim a standalone production-lineage replay.
