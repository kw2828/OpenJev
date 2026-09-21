# OpenJev research draft

[Read the compiled paper](../output/pdf/openjev-rl-paper.pdf) · [LaTeX source](main.tex) · [References](references.bib)

These PDFs are scoped development snapshots. The [experiment archive](../research/experiment-index.md) records later results and active studies; the PDFs have not been expanded to include every follow-up.

## Bellman versus Monte Carlo continuation

[Standalone PDF](../output/pdf/openjev-otto-bellman-control.pdf) · [LaTeX](otto-bellman-control.tex) · [Complete results](../research/otto-bellman-control-results.md) · [Raw evidence](https://github.com/kw2828/OpenJev/releases/tag/otto-bellman-control-v1)

Six continuations and 1,440 odor searches compare training targets in the same small MLP. Bellman training improves over further Monte Carlo training, but the full rule fails: **19/42 checks passed, including 0/18 competence checks**. It also loses to the unchanged model after the sensing-kernel shift. All fitting seeds, training costs, deployment accounting and 42 conditions are included. The independent audit agrees on 64,603,745 comparisons. This is a development report, not an architecture advance or ICLR submission.

From `paper/`, compile with `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build/bellman otto-bellman-control.tex`. The checked-in figure and LaTeX require no model or simulator calls. All five pages were visually checked after an offline build; [build and publication record](../output/otto-bellman-control-v1/paper-support/publication-01/receipt.json). Earlier Doom, chess and reaching reports remain separate snapshots.

## Doom development report

The original report is a working development draft, not a submitted or accepted ICLR paper. It reports the original pilots, the frozen 1,980-episode causal-memory follow-up, and three prospective engineering studies totaling 6,408 further episodes. Event memory improves command efficiency, but all three earlier broader continuation gates failed. The memory continuation gate failed against the capacity control, so the conditional downstream efficacy experiments were not launched. A subsequent PPO/DQN study trained nine fits for 294,912 interactions. History-PPO passed its separately frozen combat gate on fresh seeds; Line behavior matches always-fire. The report includes this distinction and the delayed-hit limitation of the earlier command metric. It does not claim proprietary RLCD reproduction or a new RL algorithm. Project-level authorship is used until the authors and affiliations are confirmed.

## Chess development report

[Chess paper (PDF)](../output/pdf/openjev-chess-study.pdf) · [Standalone LaTeX](chess-study.tex) · [Models and results](../docs/chess-student.md) · [Arena](../docs/chess.md)

Nine original chess students, a 24-position public puzzle panel and 18 recorded games are complete. The circuit topology gate failed. This separate six-page development report includes a proposed memory/state-transition experiment; that proposal has not been run and is not an established ICLR contribution. Build with `latexmk -pdf -outdir=build/chess chess-study.tex` from this directory.

## Geometry-scored robot memory

[Four-page PDF](../output/pdf/openjev-reacher-geometry-memory-study.pdf) · [LaTeX source](reacher-geometry-memory-study.tex) · [Results](../research/reacher-geometry-memory.md) · [Build receipt and exact input snapshots](../output/reacher-geometry-memory-v1/paper-build/receipt.json)

Twelve inherited models and all 51 comparisons are retained. Persistent GRU reduced control cost by 31.86%/25.99% versus cached-observation GRU on six-step/ten-step gaps, passing all 25 frozen checks. Supplied-physics references still perform better. This is a development report, not an ICLR submission or a new-architecture claim. It is separate from the earlier Doom and chess PDFs.

The [subsequent stronger history control](../research/reacher-two-observation-control.md) is now complete. Against two observations and intervening actions, the memory advantage shrank to 4.44%/2.94%. Its continuation rule failed with 24/25 checks passed because the ten-step mean missed the required 3% margin. The earlier PDF and its positive weaker-control result remain unchanged; they do not establish superiority over this later control.

From the repository root, build with `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=paper/build/reacher-memory paper/reacher-geometry-memory-study.tex`. The recorded build used the existing pinned TeX container without network access. Both plot pages use landscape orientation for legibility; all four rendered pages were checked. No new model or native-environment calls were made to write the paper.

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

## Spatial chess follow-up

[Paper PDF](../output/pdf/openjev-chess-spatial-study.pdf) and [LaTeX source](chess-spatial-study.tex) report all twelve board-aware chess fits. Future prediction did not pass its continuation criterion. The tables are generated from the audited completed release by `scripts/write_chess_spatial_paper.py`.
