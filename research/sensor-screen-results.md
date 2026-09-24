# Sensor calibration is the wrong next memory benchmark

**Rejected under the frozen rule: 3/6 conditions passed.** The nine-method,
TRAIN-only screen found too little headroom for a learned recurrent-memory
experiment on this benzene target. The independent audit agrees. This is a
benchmark rejection, not evidence that recurrent memory is generally ineffective.

![All nine predictors on the two TRAIN months](sensor-screen-results/benchmark.png)

Static quadratic calibration already reaches **0.04724 May RMSE and 0.05825
June RMSE**, below the predeclared minimum-headroom threshold of **0.07340**.
That threshold is 1% of the FIT target standard deviation. The best static
model uses all seven inputs in May and only the S2 sensor in June.

| Predictor | May RMSE | June RMSE |
| --- | ---: | ---: |
| S2 quadratic | 0.06260 | 0.05825 |
| Seven-input quadratic | 0.04724 | 0.06719 |
| Quadratic RLS, lambda 1 | 0.04422 | 0.03917 |
| Quadratic RLS, lambda 0.995 | 0.04577 | 0.03737 |
| Persistence | 6.47470 | 5.60724 |

Adaptation helps in June. Neither adaptive configuration improves on the best
static model by the required 10% in May, so no fixed configuration meets the
both-month rule. The [complete table](sensor-screen-results/report.md) includes
all nine methods, MAE, retained-state accounting and every condition.

We fit on **1,179** complete rows whose reference labels were available strictly
before May 1, then scored **730 May rows and 682 June rows**. The seven public
inputs include all five sensors, temperature and relative humidity. Other
reference-analyzer columns are excluded. Labels are revealed after an imposed
24-hour delay. Normalization is fitted only on eligible early rows. Missing
values preserve their calendar positions. July and later measurements remain
numerically unparsed. These are development-period results, not held-out results.

The three data-support conditions pass. Both minimum-headroom conditions and
the repeated-adaptation condition fail. We stop this benzene-memory branch;
removing S2, changing the target or weakening controls after these results would
not rescue the registered experiment.

The screen also corrects a budget error in the earlier
[prospective sketch design](measurement-next.md). With seven float64 inputs and
a 24-hour delay, each pending-input queue alone costs **1,344 bytes**. Actual
logical state is 2,049 bytes for linear RLS and 12,129 for quadratic RLS. The
screen has no matched-byte cap, inference-latency comparison, learned sketch or
calibrated Bayesian-posterior claim.

**Validation:** 75 fabricated-data tests pass. The independent audit reconciles
all 52 arrays across three saved files with the original TRAIN CSV and separate
regression equations, including nine prediction streams, initial/final states,
label-release indices and all six conditions. Original qualification, run,
audit, plot and package processes closed successfully. No scientific retry or
held-out evaluation occurred. Plotting and packaging perform no scientific fits
or array replays.

The [registration](sensor-screen-registration.json) and seven source files were
committed at `a99c05ff91bf1b29c22812065fc3dba60a0067a5` before the first real-data
fit. Registration SHA256:
`e99a649965127a6bbb4ca3b6d6cfc999ab71c73c1ef0bca382db8773751afe60`.

[Protocol](sensor-screen-protocol.md) ·
[Independent audit](sensor-screen-results/audit.json) ·
[Public evidence manifest](sensor-screen-results/manifest.json) ·
[PDF chart](sensor-screen-results/benchmark.pdf) ·
[Next research constraint](sensor-screen-next.md).

Source: [UCI Air Quality, Saverio Vito (2008)](https://archive.ics.uci.edu/dataset/360/air+quality),
DOI 10.24432/C59K5F. Its page describes reference-analyzer targets; a close S2
calibration fit does not establish the target's provenance or prove leakage.
The source page has conflicting license language. The public bundle contains
derived results, source hashes and process evidence; the original CSV and
numeric TRAIN arrays remain local. The publicly downloadable CSV is pinned in
the protocol. This screen does not establish a new architecture, biological
learning benefit or an ICLR-level result.
