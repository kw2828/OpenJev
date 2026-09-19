Completed development pilot: four memory-update gates, three paired initializations, 12 fits, 1,920 optimizer updates, and 48 initial/final evaluations. **The normalized-error gate failed its continuation rule: 21/29 checks passed.** Its improvement over age gating was only 0.004% with original sensing and 0.122% with additional censoring, below the required 3% in both panels.

The evidence bundle contains all twelve fitted models and optimizer checkpoints, every saved prediction, training logs, public input splits, actual initial tensors and epoch orders, frozen source snapshots, process receipts, the completed saved-output audit, and the comparison figure. Every one of its 445 decompressed members was verified against the included public manifest before upload. Additional independent arithmetic and training diagnostics are in the repository report.

- [Readable results and limits](https://github.com/kw2828/OpenJev/blob/087e8429eb0c6517a942e8ec02d23f26d688264c/research/reacher-innovation-pilot.md)
- [Protocol published before training](https://github.com/kw2828/OpenJev/tree/9c7151a1892b0ccf15d122965339cf258971c245/evidence/reacher-innovation-pilot-v1/protocol)
- Protocol SHA256: `561eb5a73f30ce81453941a6ade73cf15f0332af26cfb6ef49a7a17a35eff3de`
- Archive SHA256: `bfc9d8842c28e84d670874b4edb800b034164b40c7f83b6a9bfff5d4dc66dc4c`
- Archive size: 60,545,371 bytes; uncompressed members: 111,713,402 bytes.

This uses an exposed development corpus and a masking stress test on recorded trajectories. It is not an untouched-test, native-control, calibrated-uncertainty, biological-wiring, or new-architecture result. No model or simulator calls were made during the audit or independent results review. The failed synthetic capacity attempt is retained in the source commit; the real pilot was run once without extension or retry.
