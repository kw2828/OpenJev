# OpenJev research draft

[Read the compiled paper](../output/pdf/openjev-rl-paper.pdf) · [LaTeX source](main.tex) · [References](references.bib)

This is a working development report, not a submitted or accepted ICLR paper. It reports the original pilots, the frozen 1,980-episode causal-memory follow-up, and three prospective engineering studies totaling 6,408 further episodes. Event memory improves command efficiency, but all three earlier broader continuation gates failed. The memory continuation gate failed against the capacity control, so the conditional downstream efficacy experiments were not launched. A subsequent PPO/DQN study trained nine fits for 294,912 interactions. History-PPO passed its separately frozen combat gate on fresh seeds; Line behavior matches always-fire. The report includes this distinction and the delayed-hit limitation of the earlier command metric. It does not claim proprietary RLCD reproduction or a new RL algorithm. Project-level authorship is used until the authors and affiliations are confirmed.

## Build

From the repository root, regenerate figures and the evidence-derived table:

```sh
uv run --no-sync --with matplotlib==3.11.2 python research/plot_benchmarks.py
uv run --no-sync --with matplotlib==3.11.2 python research/plot_memory_doom.py
```

Compile from `paper/` using a TeX Live installation with `latexmk`:

```sh
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
mkdir -p ../output/pdf
cp build/main.pdf ../output/pdf/openjev-rl-paper.pdf
```

Alternatively, `tectonic --outdir build main.tex` can compile with its own downloaded bundle after creating `build/`. TeX dependencies may require network access on first use. The checked-in figures and table allow paper compilation without rerunning Python or Doom.

The new confirmation figures can be regenerated without rerunning Doom:

```sh
uv run --no-project --with matplotlib python research/plot_cadence_doom.py evidence/event-cadence-doom-v1/analysis-001.json --prefix evidence/event-cadence-doom-v1/confirmation-contrasts
uv run --no-project --with matplotlib python research/plot_cadence_doom.py evidence/portable-head-doom-v1/analysis-001.json --prefix evidence/portable-head-doom-v1/confirmation-contrasts
```

To regenerate the RL confirmation figure:

```sh
uv run --no-project --with matplotlib python research/plot_rl_doom.py evidence/rl-doom-v1/confirmation-analysis-001.json --prefix evidence/rl-doom-v1/confirmation-results --controls evidence/rl-doom-v1/confirmation-controls-means.json
```

## Evidence and figures

- [RL study and simple-control diagnostics](../docs/rl-study.md).
- [All improvement studies and their preserved evidence](../docs/improvement-studies.md).
- `evidence/defend-center.json`: original 60-episode, two-tic imitation/rule/random benchmark.
- `evidence/bayesian-doom-v2/analysis-001.json`: valid 440-episode study, seven-tic windows and shared rule steering.
- `evidence/rust-python-score-kernel.json`: one local microbenchmark, with raw batch measurements and environment details.
- `evidence/benchmarks/manifest.json`: hashes of the figure inputs and plotting code, plus rendering versions.
- `evidence/bayesian-doom-v1/run-001/quality-review.json`: preserved earlier measurement exception.

Figures are available as PNG for README, SVG for editing and PDF for LaTeX. The audit figure shows Brier score, which includes more than calibration error. The speed figure shows IQR, not a confidence interval. The existing utility plot uses exploratory paired bootstrap intervals. These error representations are intentionally labeled separately.

Regenerating figures does not run new experiments. See `docs/bayesian-rlcd.md` for experiment commands and limits. The original gameplay and Bayesian pilot use different protocols and cannot be combined into one policy leaderboard.

The original `output/pdf/openjev-paper.pdf` and `paper/build-receipt.json` are preserved as the earlier baseline draft and its receipt. The earlier memory draft and its `paper/openjev-memory-build-receipt.json` are also preserved. The earlier improvement draft and its `paper/openjev-improvement-build-receipt.json` remain preserved. The current draft is `output/pdf/openjev-rl-paper.pdf`, with `paper/openjev-rl-build-receipt.json`. The interim DecisionTics draft and its build receipt remain historical artifacts.

The subsequent [SRPO adaptation pilot](../docs/srpo-study.md) is reported separately. This PDF covers studies through PPO/DQN and does not include the later SRPO pilot.

The subsequent [pretrained JEPA and nine-method RL pilot](../docs/jepa-rl-study.md) is also reported separately, with its frozen protocol and complete evidence. The PDF has not been expanded to claim these later results.
