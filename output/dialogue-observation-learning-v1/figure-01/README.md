# Saved-report figures

This directory is outside the 53 frozen scientific sources. `plot.py` reads only a complete, externally hash-pinned report. It authenticates the three report payloads before decoding `summary.json`. It does not read predictions, checkpoints, corpus records, active fits, or model assets. The independent result audit is a separate prerequisite handled by the study owner, not repeated by this presentation tool.

After the entire study, report, and independent audit finish:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
.venv/bin/python output/dialogue-observation-learning-v1/figure-01/plot.py \
  --report /absolute/path/to/completed-report \
  --summary-sha256 EXTERNALLY_VERIFIED_SUMMARY_SHA256 \
  --receipt-sha256 EXTERNALLY_VERIFIED_REPORT_RECEIPT_SHA256 \
  --out output/dialogue-observation-learning-v1/figure-01/render-01
```

The output directory must not exist. Limits are 60 seconds, 1 GiB process RSS, and 32 MiB output. Success writes two PNG/SVG pairs, exact plotted values, a start record, and an input/source/output-bound receipt. Failure is retained and never silently retried.

`observation` shows seen/unseen three-stratum macro accuracy and unseen micro NLL/Brier. `retention-and-conditions` shows assigned-retention error and the seven original logical conditions. Every panel contains all four arms and three training seeds, with descriptive mean ticks. Dots use zoomed axes, not bars. Seed repetitions use the same DEV examples; they do not justify confidence intervals. Support counts remain visible. This is exposed development evidence, with no new architecture or calibration claim.

The separately marked synthetic smoke exercises only presentation and rejection of bad external pins or extra report members before quality decoding. Its invented numbers are not study results.

Publication includes the generator, smoke source and receipts. Invented report
fixtures and rendered synthetic PNG/SVG files remain local; their hashes are
retained in the receipts. Actual benchmark figures will be published only after
the complete scientific result and independent audit exist.
