The fixed 80-versus-320 epoch comparison completed: six paired fits, identical R64 labels, exact first-80-epoch prefixes, 6,000 updates and 504 fresh full-horizon searches.

Longer training lowered final TRAIN MSE by 11.38% and the chosen-action cost gap against the saved labels by 14.78%. Weighted success rose from 10.19% / 20.42% / 4.52% to 15.34% / 21.57% / 9.52%. Analytic control found all 72 sources. The frozen continuation rule failed: 10/33 passed, including 0/18 competence conditions. No architecture advantage or promoted checkpoint.

The independent saved-record audit agrees on all outcomes and criteria. It checks recorded scores, training prefixes, diagnostic arithmetic, work and costs; it does not regenerate neural scores, gradients or intermediate posterior filtering.

The archive includes all current-phase checkpoints, raw work/evaluation records, fixed TRAIN diagnostics, frozen plans, qualifications, audit, figures and execution sources. SHA256SUMS and manifest.json provide checksums for the archive and its 72 members. Historical inputs and earlier releases remain dependencies specified by the frozen plan; this is not a standalone reconstruction of the entire lineage.

[Full results](https://github.com/kw2828/OpenJev/blob/main/research/otto-training-budget-results.md) · [Protocol](https://github.com/kw2828/OpenJev/blob/main/research/otto-training-budget-protocol.md) · [Next proposed diagnostic](https://github.com/kw2828/OpenJev/blob/main/research/otto-training-budget-next-decisions.md)
