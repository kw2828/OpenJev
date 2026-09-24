# A smaller recurrent world model improves means but remains unstable

**The 352-parameter factorized model passes observed filtering, but fails short-horizon learning and blind extrapolation. It does not advance as a reliable candidate.** All nine fits completed and the saved-output audit passed; scientific success is separate from technical completion.

![Every fit seed, decision regret, observed event error, likelihood and fitting time](finite-factorized-dynamics-results/benchmark.png)

[Full tables and work accounting](finite-factorized-dynamics-results/report.md) ·
[Machine-readable results](finite-factorized-dynamics-results/summary.json) ·
[Frozen protocol](finite-factorized-dynamics-protocol.md) ·
[All nine checkpoints and original evidence](https://github.com/kw2828/OpenJev/releases/tag/finite-factorized-dynamics-v1)

## What changed

All three models retain the learned bounded cost head and the same H2 training objective. The smaller model factors dynamics into learned transition probabilities, observation probabilities and an absorbing-event hazard. Its observation law also supplies the reset distribution. These conditional independences come from the synthetic task; their numerical values are learned.

The first 1,120-parameter control starts with the same predictive functions, within 1e-12, but can learn unrestricted dynamics afterward. The second retains the previous dense initialization. Matching initial predictions does not match gradients or optimizer geometry. Structure, shared reset/emission, capacity and parameterization are bundled in this comparison.

## Results across all three seeds

| Measure | Factorized | Initially matched free | Dense free |
| --- | ---: | ---: | ---: |
| Trainable parameters | 352 | 1,120 | 1,120 |
| Short-horizon criterion | FAIL 23/24 | FAIL 19/24 | FAIL 21/24 |
| Blind extrapolation criterion | FAIL 17/21 | FAIL 12/21 | FAIL 9/21 |
| Observed filtering criterion | PASS 8/8 | PASS 8/8 | FAIL 7/8 |
| Mean four-step decision regret | 0.152220 | 0.234729 | 0.292058 |
| Mean eight-step decision regret | 0.214276 | 0.265822 | 0.351249 |
| Mean eight-step observed event KL | 0.057438 | 0.091386 | 0.098923 |
| Mean prefix event NLL | 0.886284 | 0.903219 | 0.919316 |
| Total fitting seconds, three fits | 67.713 | 64.571 | 64.451 |

Lower is better for the errors, regret and NLL. Factorized mean four/eight-step regret is **35.15% / 19.39% lower** than the initially matched control and **47.88% / 39.00% lower** than the dense control. The same-seed regret comparisons improve in only four of six and five of six cells, respectively.

Seed **426261002** is worse than the initially matched control at both long horizons. Factorized eight-step regrets are **0.010825, 0.388265 and 0.243739** for seeds 001, 002 and 003. Seed 002 fails four absolute conditions; seed 003 fails eight-step cost MSE. A favorable mean and the strongest seed cannot override those failures. The observed-filtering criterion measures event KL while consuming intervening observations; it does not establish good autonomous planning or a cost-accuracy pass.

The smaller model uses **68.57% fewer parameters**, but its measured total fit time is **4.87% / 5.06% longer** than the two controls. These are descriptive single-machine timings, not matched-compute or inference-speed claims. Constructor work/time, both initial/final snapshot types, prefix inference and endpoint inference are recorded separately in the full report. Dense operator products still materialize during inference.

## Evidence retained

The new namespace is **426260924**, with fit seeds **426261001-426261003**. Training retains all 512 attempts, including 41 terminal prefixes; 471 surviving prefixes have H2 endpoint targets. Development retains all 128 attempts, including ten terminal prefixes; 118 survivors have H8 targets. All nine final checkpoints preceded development generation.

Every arm uses 480 epochs, batches of 64, Adam 0.003, gradient clip 5, coefficient-one prefix likelihood and paired minibatch orders. Completed work is **34,560 updates**, **2,211,840 attempted-case exposures**, **2,034,720 endpoint exposures** and **19,357,920 prefix-event exposures**. No data replacement, model-selection rerun or scientific retry occurred.

Qualification passed **135 tests** with one scalar-conversion warning in a reference test. The original qualification, fit and audit phases closed successfully in **4.450, 199.538 and 1.905 seconds**. The audit decoded 18 saved numerical files and no checkpoints, made no model calls, independently reconstructed targets and checked work, snapshots, rows and criteria. The rendered figure was visually checked.

The first publication attempt stopped before reading metrics because its helper expected the wrong registration filename. A two-literal publication-only correction pointed to the existing original registration. The failed helper copies and failure record are preserved; no study source, result or original phase was changed.

Registration SHA256: `b36151240ba10217c3e4252f8a3657c1705511caeddf00564c73ff4bda1f68b0`.

## Interpretation

This supports the combined factorization as a compact baseline worth understanding, not a stable solution, a discovered biological architecture or an ICLR novelty result. Initial-function matching rules out different starting predictions as the complete explanation; it does not isolate which structural constraint caused the mean improvement. The task-derived factorization, known uniform prior and world-aligned readout initialization are advantages supplied to the learner.

The previous [replication failure](finite-cost-readout-replication-results.md) stays closed, including its proposed H4 training follow-up. Native chess, Doom and robotics transfer, scenario shifts and calibration of text decisions remain unproven. [Next diagnostic proposal](finite-factorized-dynamics-next.md).
