# Predicting observations better does not yet make decisions reliable

**Adding public-history prediction improves observation forecasts, but both
versions still fail all three learning criteria.** Six matched fits completed;
the independent audit agrees with the saved targets, scores and work counts.
This is a bounded supervision result in a synthetic world, not an architecture
breakthrough or a calibrated text model.

![All six fits, prediction errors and training costs](finite-prefix-learning-results/benchmark.png)

Both versions use the identical 1,088-parameter recurrent filter, paired starting
weights, attempt order and 480-epoch schedule. The intervention adds a fixed
weight-one observation likelihood loss. It predicts each event before seeing
its label, including the first terminal event. No learned model receives the
true hidden state or dynamics. Both retain a known, fixed cost readout and
uniform reset prior, which limit generality.

| Criterion | Existing forecast loss | Forecast plus history prediction |
| --- | --- | --- |
| Short-horizon learning | FAIL, 6/24 | FAIL, 6/24 |
| Blind extrapolation | FAIL, 9/21 | FAIL, 9/21 |
| Observed filtering extrapolation | FAIL, 2/8 | FAIL, 2/8 |

These are condition counts, not accuracy scores. Every threshold must hold for
every fit seed. Lower observation loss cannot rescue a failed decision criterion.

| Measurement | Existing loss | Added prediction loss | Change |
| --- | ---: | ---: | --- |
| Fresh prefix NLL, nats/event | 1.040459 | 0.836951 | 19.56% lower |
| Four-step blind regret | 0.526399 | 0.555055 | 5.44% higher |
| Eight-step blind regret | 0.553331 | 0.519877 | 6.05% lower |
| Eight-step blind cost MSE | 0.105958 | 0.106566 | 0.57% higher |
| Eight-step observed KL, nats | 0.219293 | 0.161247 | 26.47% lower |
| Eight-step observed cost MSE | 0.162727 | 0.195079 | 19.88% higher |
| Training seconds, all three fits | 43.94 | 62.15 | 41.45% higher |

Lower is better. Quality entries are arithmetic means of three independently
initialized fits on the same fresh cases, not ensemble predictions or confidence
intervals. Percentages compare those means. Eight-step regret improves in every
paired seed; four-step regret worsens in two of three. Observed KL improves in
all six paired H4/H8 comparisons, while four-step blind cost MSE worsens in all
three. Better event prediction therefore does not establish better control.

The history-ignorant, known-dynamics reference has eight-step regret **0.484664**,
which remains lower than both learned means. Its cost MSE is **0.104593**.
The exact-world prefix NLL on these realized DEV events is **0.763123**; this is
an informative reference, not a finite-sample lower bound on empirical NLL.

All **512 TRAIN attempts** contribute to the population: 4,535 valid events,
including 22 first-terminal events, and 490 eligible forecast cases. Fresh DEV
has **128 attempts**, 1,128 valid events, seven terminal histories and 121
eligible forecast cases. No replacements were sampled. All six final
checkpoints were saved before DEV generation.

Training completed **23,040 updates**, 1,474,560 attempt exposures and 1,411,200
eligible endpoint exposures. The added-loss arm used **6,530,400 additional
prefix-event exposures**. There were no all-ineligible batches in this run;
their zero-gradient Adam behavior was tested separately. Equal optimizer steps
are not equal supervision or compute. The model and inference procedure are
unchanged; no inference speedup is claimed.

Qualification passed **103 tests** on its first attempt, with one test-only
requires-grad tensor-to-scalar warning. The complete original fit/evaluation
phase took **107.74 seconds**, qualification **3.43 seconds**, and independent
audit **1.21 seconds**. The audit reconstructed public-history targets, terminal
masks, likelihoods and work without model or optimizer calls. All previous
study source pins remain unchanged.

This comparison does not establish probability calibration, convergence,
correct latent-state recovery, connectome benefits or transfer to chess, Doom
or robotics. The fixed cost basis distinguishes only four decision signatures
across eight latent states. A better observation model can still be poorly
aligned with the decisions that matter.

[Every seed, paired difference and measured cost](finite-prefix-learning-results/report.md) ·
[Machine-readable results](finite-prefix-learning-results/summary.json) ·
[Frozen protocol](finite-prefix-learning-protocol.md) ·
[All six checkpoints and original evidence](https://github.com/kw2828/OpenJev/releases/tag/finite-prefix-learning-v1).

[Next proposed comparison: train the cost readout](finite-prefix-learning-next.md).
