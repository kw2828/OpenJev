Six continuation fits on the same 54 training paths were evaluated on 36 fresh development paths (14,338 states). The full recurrent-training candidate failed its registered rule: only 1 of 6 seed/setting comparisons passed. Measured training-cost comparability passed for all three fit seeds.

Training only the 116-parameter readout achieved 3.01% / 12.82% lower mean later teacher-cost gaps than full joint training across the two settings. It beat joint training in 4/6 comparisons and the pretrained parent in 4/6, with three simultaneous wins. Both arms still use the same recurrent network. No inference speedup, autonomous-control gain, connectome benefit or novel architecture is established.

The recipe fixed 74 readout epochs versus 40 joint epochs before fresh collection. Actual readout training took 0.59% to 6.63% more elapsed time, within the registered 10% comparison interval. The independent audit reconciled nine prediction views, six fits, 3,078 updates and 18,468 episode exposures without new inference or training.

- [Results, chart and limits](https://github.com/kw2828/OpenJev/blob/main/research/otto-readout-compute-results.md)
- [Every fit and registered check](https://github.com/kw2828/OpenJev/blob/main/research/otto-readout-compute-results/README.md)
- [Prospective protocol](https://github.com/kw2828/OpenJev/blob/main/research/otto-readout-compute-protocol.md)

The archive preserves current checkpoints, predictions, training journals, fresh collection, original process receipts, independent audit, registered sources and authenticated prior lineage. The manifest lists every member and discloses external native assets. Registered paths are absolute and pinned Python environments are external: this is evidence packaging, not a portable self-contained installation.

`SHA256SUMS` covers the archive, manifest and archive-verification receipt. Prior TEST and reserved-confirmation payloads are not included or admitted by this study.
