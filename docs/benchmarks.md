# OpenJev benchmark figures

[Paper draft (PDF)](../output/pdf/openjev-improvement-paper.pdf) · [LaTeX source and build instructions](../paper/README.md)

These figures visualize existing development evidence. No new model training or gameplay runs were performed to produce them. Each experiment has a separate scope; they do not form a single leaderboard.

## Prospective improvement studies

[All three studies and failed continuation gates](improvement-studies.md). These ran 6,408 new episodes; the figures below were generated from preserved results. Each confirmation uses 96 paired seeds in both scenarios. Bars are exploratory 99.375% bootstrap intervals, adjusted within the primary comparison family of each study.

![Event-memory confirmation](../evidence/event-cadence-doom-v1/confirmation-contrasts.png)

The combined event-memory rule reduces redundant commands with identical paired kills, but fails the broader comparisons against historical models.

![Portable model confirmation](../evidence/portable-head-doom-v1/confirmation-contrasts.png)

The selected fitted model improves utility over rules, but does not establish kill noninferiority in Line or clear the historical-bank comparisons. [All component comparisons](portable-head-study.md).

## Original gameplay

![Original gameplay benchmark](../evidence/benchmarks/gameplay.png)

Twenty matched seeds per policy, two-tic actions, Defend the Center. Every dot is an episode; red marks are means. The 5,253-parameter imitation model approximately matches its rule teacher. This uses full steering policies, unlike the Bayesian firing-only comparison. [Episode data](../evidence/defend-center.json) · [Vector figure](../evidence/benchmarks/gameplay.svg).

## Bayesian decision utility

![Bayesian utility comparison](../evidence/bayesian-doom-v2/utility-comparison.png)

The 440-episode valid development study includes training, audit and evaluation episodes. Learned methods use five fits and ten paired evaluation seeds per scenario. Existing baseline episodes are reused across fits, not counted as independent copies. Seven-tic actions and rule steering are shared. Intervals are exploratory, unadjusted paired crossed-bootstrap intervals. [Protocol and limits](bayesian-rlcd.md) · [Analysis](../evidence/bayesian-doom-v2/analysis-001.json).

## Probability prediction on a common audit

![Common-audit Brier scores](../evidence/benchmarks/audit-brier.png)

Lines pair MAP and Bayesian predictions from each of five fitted models. Diamonds are means, not confidence intervals. Each scenario uses the same ten audit episodes for all fits, with dependent events within each episode. A lower Brier score indicates better overall probability prediction, not calibration alone. The shifted audit's descriptive improvement did not establish better control utility. [Vector figure](../evidence/benchmarks/audit-brier.svg).

## Rust versus Python/NumPy scoring

![Score-kernel timings](../evidence/benchmarks/score-kernel.png)

Eleven timed batches per implementation, after three warmups, on identical float64 inputs: 32 rows, 151,936 vocabulary entries and six candidates. Points are batch milliseconds divided by 32; red marks show the median and interquartile range. Rust ran first, other Docker services were active, and only one environment was measured. NumPy uses native numerical kernels. These are implementation-specific timings, not a general language comparison or model-inference speedup. [Raw timings and environment](../evidence/rust-python-score-kernel.json) · [Vector figure](../evidence/benchmarks/score-kernel.svg).

## Regenerate

From the repository root:

```sh
uv run --no-sync --with matplotlib==3.11.2 python research/plot_benchmarks.py
```

This creates PNG, SVG and PDF figures, a LaTeX results table, and a [source-hash manifest](../evidence/benchmarks/manifest.json). It does not rerun experiments or overwrite their measurements. The earlier utility comparison has its own `research/plot_bayesian_doom.py` renderer.
