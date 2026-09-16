# OpenJev benchmark figures

[Paper draft (PDF)](../output/pdf/openjev-memory-paper.pdf) · [LaTeX source and build instructions](../paper/README.md)

These figures visualize existing development evidence. No new model training or gameplay runs were performed to produce them. Each experiment has a separate scope; they do not form a single leaderboard.

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
