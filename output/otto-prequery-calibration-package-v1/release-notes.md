Directly supervising the forecast before each planner query improved both recurrent architectures, but the frozen architecture continuation rule failed: **51/55 conditions passed**.

- 90 complete trajectories, 12 final fits, 8,640 optimizer updates.
- Explicit correction: primary teacher-score gap fell 20.32% / 40.62% and prior MSE fell 44.87% / 40.59% versus its original-loss control.
- The ordinary error-fed GRU also improved. The proposed correction did not consistently beat it and lost full-path agreement versus its own baseline.
- Auxiliary fits cost about 1.9 times their respective baselines. Equal update counts did not mean equal computation.
- Independent saved-record audit agreed on 775,806 checks. All 55 criteria and all 12 fits are retained.

These are fixed-path teacher-score forecasts, not autonomous-control, world-model, biological-learning or architectural-novelty results. Prior MSE is not probability calibration. No scientific phase was retried or extended after validation.

[Results and costs](https://github.com/kw2828/OpenJev/blob/main/research/otto-prequery-calibration-results.md) · [Frozen protocol](https://github.com/kw2828/OpenJev/blob/main/research/otto-prequery-calibration-protocol.md)

The archive includes trajectories, all 12 checkpoints, predictions, qualification records, receipts, source and figures. Verify SHA256SUMS and the member inventory before reuse. Exact inherited runtimes, native weights and historical dependencies are listed separately; this is not a standalone runtime. Saved-output auditing does not independently rerun model inference, gradients, teacher outputs or timers.
