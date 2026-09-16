# OpenJev research draft

[Read the compiled paper](../output/pdf/openjev-paper.pdf) · [LaTeX source](main.tex) · [References](references.bib)

This is a working development report, not a submitted or accepted ICLR paper. It reports existing results, including negative findings. It does not claim proprietary RLCD reproduction or a new RL algorithm. Project-level authorship is used until the authors and affiliations are confirmed.

## Build

From the repository root, regenerate figures and the evidence-derived table:

```sh
uv run --no-sync --with matplotlib==3.11.2 python research/plot_benchmarks.py
```

Compile from `paper/` using a TeX Live installation with `latexmk`:

```sh
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
mkdir -p ../output/pdf
cp build/main.pdf ../output/pdf/openjev-paper.pdf
```

Alternatively, `tectonic --outdir build main.tex` can compile with its own downloaded bundle after creating `build/`. TeX dependencies may require network access on first use. The checked-in figures and table allow paper compilation without rerunning Python or Doom.

## Evidence and figures

- `evidence/defend-center.json`: original 60-episode, two-tic imitation/rule/random benchmark.
- `evidence/bayesian-doom-v2/analysis-001.json`: valid 440-episode study, seven-tic windows and shared rule steering.
- `evidence/rust-python-score-kernel.json`: one local microbenchmark, with raw batch measurements and environment details.
- `evidence/benchmarks/manifest.json`: hashes of the figure inputs and plotting code, plus rendering versions.
- `evidence/bayesian-doom-v1/run-001/quality-review.json`: preserved earlier measurement exception.

Figures are available as PNG for README, SVG for editing and PDF for LaTeX. The audit figure shows Brier score, which includes more than calibration error. The speed figure shows IQR, not a confidence interval. The existing utility plot uses exploratory paired bootstrap intervals. These error representations are intentionally labeled separately.

Regenerating figures does not run new experiments. See `docs/bayesian-rlcd.md` for experiment commands and limits. The original gameplay and Bayesian pilot use different protocols and cannot be combined into one policy leaderboard.
