Completed four-arm Reacher memory comparison: **all 25 frozen continuation checks passed**.

Persistent GRU lowered mean native cost by 9.27% on six-step gaps and 9.00% on ten-step gaps versus the trained current-packet GRU. Its longer-gap gain versus three-packet history was 6.51%. Every paired GRU comparison improved on both gap panels. All 12 fits and 51 control rows are retained.

The gain is uneven across fits. MLP family means are close, the persistent model loses one paired MLP comparison, and supplied-physics references remain better. This supports useful public-history memory under this training recipe. It does not establish biological wiring, velocity inference, a new architecture or an ICLR contribution.

- [Results and full limitations](https://github.com/kw2828/OpenJev/blob/main/research/reacher-memory-ablation.md)
- [Every paired comparison](https://github.com/kw2828/OpenJev/blob/main/evidence/reacher-memory-ablation-v1/paired-figure/paired-improvements.png)
- [Recorded first-case schematic replay](https://github.com/kw2828/OpenJev/blob/main/evidence/reacher-memory-ablation-v1/replay/first-ordinary-case-pair0.gif)

The saved-output audit replayed 206,400 native transitions with zero discrepancy, with no new learned-model calls. The execution took 1,083.97 seconds; audit validation took 30.97 seconds. Equal data and updates do not imply equal compute.

The scored archive preserves the complete execution, checkpoints, audit, frozen sources, protocol, supervision and reporting. If split into `.partNNN` assets, concatenate those parts in receipt order before opening the gzip tar archive. Verify the full SHA-256 in `receipt.json`; `manifest.json` binds every archived file. Upstream historical lineage artifacts remain separate release dependencies.

The capacity and rehearsal archives are separate engineering evidence, not scientific effectiveness results. No best fit or episode was selected. The schematic replay deliberately shows case 0 / pair 0, which favors the MLP.

Frozen plan SHA-256: `05f09ee5425190f5d652aedb10af1641037035d85e906d04d86baf33552a5786`

Final audit receipt SHA-256: `2589383326a2312d7f738821da7fd11f82b9bee1b7168aba2e7f44aaa4946d5d`
