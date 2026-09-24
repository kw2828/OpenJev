# Learning the decision readout helps, but long forecasts remain unreliable

**The learned readout passes two of three criteria across all three fit seeds.**
Both fixed-readout controls fail all three. Nine matched fits and an independent
saved-record audit completed on their first attempts. This supports training
the decision readout under this recipe in a small synthetic world. It does not
establish a new architecture, calibrated probabilities or native task transfer.

![All nine fits, prediction quality and training time](finite-cost-readout-results/benchmark.png)

Each version uses the same recurrent filter, observation-prediction loss,
public histories, paired starting filter weights and 480-epoch schedule. The
controls hold either the known exact cost matrix C or its softened version
0.9C fixed. The candidate learns a bounded linear cost matrix from the same
initial function as 0.9C. It adds 32 trainable logits to the existing 1,088
parameters. All three initial readouts remain aligned with the known world;
none learns from unstructured text or pixels.

| Criterion | Fixed exact C | Fixed softened C | Learned readout |
| --- | --- | --- | --- |
| Short-horizon learning | FAIL, 8/24 | FAIL, 10/24 | **PASS, 24/24** |
| Blind extrapolation | FAIL, 9/21 | FAIL, 9/21 | **FAIL, 16/21** |
| Observed filtering extrapolation | FAIL, 5/8 | FAIL, 7/8 | **PASS, 8/8** |

These are condition counts, not accuracy percentages. Every condition must
hold for every seed. Short-horizon learning tests one- and two-step forecasts.
Blind extrapolation tests four- and eight-step forecasts without new evidence;
observed filtering receives intervening observations and tests event KL.

| Three-seed mean, lower is better | Fixed exact | Fixed softened | Learned |
| --- | ---: | ---: | ---: |
| Four-step blind regret | 0.522908 | 0.494941 | 0.137260 |
| Eight-step blind regret | 0.489451 | 0.455190 | 0.228868 |
| Four-step blind cost MSE | 0.128634 | 0.125317 | 0.041360 |
| Eight-step blind cost MSE | 0.101825 | 0.098586 | 0.046908 |
| Four-step observed cost MSE | 0.184953 | 0.180912 | 0.037481 |
| Eight-step observed cost MSE | 0.193688 | 0.183660 | 0.045668 |
| Four-step observed KL, nats | 0.112620 | 0.088435 | 0.047095 |
| Eight-step observed KL, nats | 0.096693 | 0.070397 | 0.040684 |
| Prefix NLL, nats/event | 0.936112 | 0.925175 | 0.913936 |
| Training seconds, all three fits | 62.94 | 62.12 | 62.39 |

Four- and eight-step regret fall **73.75% / 53.24%** against the exact control
and **72.27% / 49.72%** against the softened control. Every paired seed improves
at both horizons, as do blind and observed cost MSE. Percentages compare
arithmetic means of separately fitted models, not ensemble outputs, statistical
significance or a new relative-performance criterion. Observed KL improves in
11 of 12 paired comparisons against the two controls; the remaining cell is
worse despite meeting its absolute threshold.

The mean hides a large reliability gap. Eight-step regret is **0.00918,
0.38266 and 0.29477** for the three learned fits. The latter two fail both
eight-step cost thresholds; the second also fails four-step cost MSE. These
are all five failed blind-extrapolation conditions. No checkpoint is selected
for promotion. The history-ignorant, known-dynamics reference has eight-step
regret 0.536681 and cost MSE 0.095998; the criterion requires halving both in
every fit, not only on average.

Learning beats the initially identical softened control and the exact-C
anchor, so merely starting with softened costs does not explain the gain.
The result still does not identify the cause as latent alignment: additional
parameters, optimization geometry and global gradient clipping change together
when the head trains. The true world is representable with fixed exact C.
Readout movement alone does not establish hidden-state recovery.

All **512 TRAIN attempts** are retained: 481 eligible endpoint cases, 31
terminal histories and 4,502 valid events. Fresh DEV retains all **128 attempts**:
117 eligible endpoint cases, 11 terminal histories and 1,100 valid events.
All nine final checkpoints precede DEV generation. There were 34,560 updates,
2,211,840 attempt exposures, 2,077,920 endpoint exposures and 19,448,640
prefix-event exposures, with no replacement sampling or all-ineligible batches.

The learned version stores 1,120 trainable float64 parameters; each fixed
control stores 1,088 parameters plus 32 fixed costs. Each therefore stores
8,960 parameter-plus-buffer bytes, excluding optimizer state and activations.
Equal storage and optimizer steps do not imply equal compute. The report
includes every head softmax, separate diagnostic exports and measured timing;
no speedup is claimed from these single-process measurements.

Qualification passed **117 tests**, with one test-only tensor-to-scalar warning.
The original fit/evaluation phase took **188.88 seconds**, qualification
**3.97 seconds**, and independent audit **1.66 seconds**. The audit reconstructed
targets and metrics from saved arrays without model, optimizer or generator
calls. The complete evidence retains all nine checkpoints, attempts, failures
of scientific criteria and original execution records.

This is a favorable eight-state synthetic problem, with only three fit seeds,
one fresh DEV population and no scenario shift. It does not establish
connectome benefits, biological learning, a new RL method or transfer to chess,
Doom or robotics. Previous studies and their failed criteria remain unchanged.

[Every seed, paired difference and measured cost](finite-cost-readout-results/report.md) ·
[Machine-readable results](finite-cost-readout-results/summary.json) ·
[Frozen protocol](finite-cost-readout-protocol.md) ·
[All nine checkpoints and original evidence](https://github.com/kw2828/OpenJev/releases/tag/finite-cost-readout-v1).

[Next proposed comparison: replication, then longer blind supervision](finite-cost-readout-next.md).
