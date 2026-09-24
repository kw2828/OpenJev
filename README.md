# OpenJev

**Context → questions and candidate answers → probabilities → your application's next step.**

OpenJev is a local decision interface and a collection of controlled learning experiments. Score choices with stable IDs, try text decisions in your browser, or explore recorded chess, Doom and robot policies.

[**Text demo**](https://kw2828.github.io/OpenJev/) · [**Chess replay**](https://kw2828.github.io/OpenJev/chess-candidate-replay.html) · [API](docs/decision-api.md) · [Setup](docs/project-reference.md) · [Experiment archive](research/experiment-index.md)

## Decision interface

Supply English context, your questions, and candidate IDs with descriptions. OpenJev returns each chosen ID and the relative probability of every supplied answer. Your application decides what to do next; the API itself executes no external action.

The local scorer uses a pinned Qwen model, maps candidates to single-token labels, and scores those labels without generating an explanation. Questions are independent. Stable IDs constrain the response format, not whether the choice is correct.

Use `POST /api/decide` with the `X-OpenJev: 1` header, or `openjev decide request.json`. The [complete example](examples/support.json) and [API reference](docs/decision-api.md) cover request limits, errors, model metadata and execution boundaries.

## Run locally

Apple Silicon macOS, Python 3.11-3.13:

```sh
git clone https://github.com/kw2828/OpenJev.git
cd OpenJev
uv sync --frozen --extra language
uv run --extra language openjev setup
uv run --extra language openjev serve
```

Open **http://127.0.0.1:8000**. Setup downloads the pinned Qwen3-4B weights; inference then uses the local cache. No paid API key is needed.

The free [browser demo](https://kw2828.github.io/OpenJev/) runs Qwen3-0.6B through WebGPU. Its scores are not interchangeable with the local 4B model. [Browser requirements](docs/browser-model.md) · [Linux, Docker and hosting](docs/hosting.md).

## Chess

[![First scheduled candidate-policy game, drawn by repetition](https://raw.githubusercontent.com/kw2828/OpenJev/main/docs/assets/chess-candidate-game-001.gif)](https://kw2828.github.io/OpenJev/chess-candidate-replay.html)

Four small policies score legal moves across **twelve fits and 288 games**. Exact next-board differences raise ordinary move agreement from **31.75% to 33.15%**, but cost more computation and fail the engine-loss and game-score criteria. The GIF is the first scheduled game, not a selected win. No learned world model or Elo rating is established.

[Open chess GIF](https://raw.githubusercontent.com/kw2828/OpenJev/main/docs/assets/chess-candidate-game-001.gif) · [Results, all checkpoints and controls](docs/chess-candidate.md) · [ChessBench transfer](docs/chessbench-transfer.md) · [Connectome comparison](docs/chess-connectome.md) · [Graph-quality comparison](docs/chess-pin-quality.md).

## Doom

[![First JEPA-trained policy and first evaluation seed](https://raw.githubusercontent.com/kw2828/OpenJev/main/docs/assets/jepa-policy-181000-preview.gif)](docs/assets/jepa-policy-181000.gif)

**13 kills in 19.26 game seconds**, from the first fit and first evaluation seed. JEPA supplies training rewards; a small policy plays this recording. The full nine-method comparison did **not** establish a reliable JEPA advantage. This is one illustrative episode, not an inference-speed benchmark.

[Open Doom GIF](https://raw.githubusercontent.com/kw2828/OpenJev/main/docs/assets/jepa-policy-181000-preview.gif) · [Results and limits](docs/jepa-rl-study.md) · [Recording receipt](docs/assets/jepa-policy-181000.json) · [Other Doom controllers](docs/project-reference.md).

## Robot reaching

[![Preselected first robot case with persistent memory and trained reset controls](https://raw.githubusercontent.com/kw2828/OpenJev/main/evidence/reacher-geometry-memory-v1/report/fixed-case-replay.gif)](research/reacher-geometry-memory.md)

[Open robot GIF](https://raw.githubusercontent.com/kw2828/OpenJev/main/evidence/reacher-geometry-memory-v1/report/fixed-case-replay.gif).

The stronger memory comparison **failed its rule: 24/25 checks passed**. Persistent GRU lowers control cost by **4.44% / 2.94%** versus a separately trained two-observation controller; the rule requires at least 3% on both sensing-gap panels. Supplied-physics controllers still perform better.

The GIF is from the earlier, weaker-control study. The [stronger comparison](research/reacher-two-observation-control.md) retains all nine models, five references and complete costs. Neither establishes a new architecture or biological-wiring advantage.

## Research status

The latest [recurrent-model diagnostic](research/finite-head-learning-results.md)
**fails its continuation rule: 14/15 conditions pass**. Fifteen fresh fits test
whether performance depends on a task-specific starting decision head. The
random-head candidate passes all three absolute criteria and wins 9/10 paired
four/eight-step comparisons, but loses one eight-step comparison.

| Random-head comparison | Four steps | Eight steps |
|---|---:|---:|
| Rounded model mean regret | 0.002477 | 0.002656 |
| Initially matched free model mean regret | 0.121260 | 0.133666 |
| Relative reduction | 97.96% | 98.01% |

[![All five seeds: task-derived and random heads, decision regret and full training time](https://raw.githubusercontent.com/kw2828/OpenJev/main/research/finite-head-learning-results/benchmark.png)](research/finite-head-learning-results.md)

[Open recurrent chart](https://raw.githubusercontent.com/kw2828/OpenJev/main/research/finite-head-learning-results/benchmark.png) ·
[Complete evidence](research/finite-head-learning-results/evidence.tar.gz) ·
[Protocol](research/finite-head-learning-protocol-v2.md).

Two weak control fits drive much of the average improvement. Candidate fitting
averages **26.48 seconds**, versus **24.95 seconds** for the control. All 375
qualification tests pass, and the independent audit agrees. The original
qualification's test-only failure is preserved alongside the corrected attempt.

This supports learning without the task-specific head in this synthetic world.
It does not establish a consistent comparative advantage or fix the
[previous observation-noise shift failure](research/finite-reuse-replication-results.md).
Known transition structure remains favorable. A second environment and evidence
of novelty are still needed; no further architecture experiment is admitted by
this failed continuation rule.

The [saved-prediction audit](research/finite-decision-error-results.md) traces
the losing comparison to nine differing decisions, including one costly error
with a large true action gap. All 29,160 records reconcile independently.
The [next hypothesis](research/finite-decision-error-next.md) tests action-error
contrast losses against stronger loss-weight controls; no new training has run.

The separate [computation-sharing benchmark](research/finite-joint-reuse-throughput-results.md)
measures **1.20x / 1.20x / 1.35x median training throughput** across three models
on one machine. It establishes no new task-performance result.

[![All nine measured training-throughput ratios](https://raw.githubusercontent.com/kw2828/OpenJev/main/research/finite-joint-reuse-throughput-results/benchmark.png)](research/finite-joint-reuse-throughput-results.md)

Every completed or failed study stays in the [experiment archive](research/experiment-index.md).

The separate [shared-prefix scoring experiment](research/shared-prefix-results.md)
measured **2.08x** speedup on four-question synthetic workloads; single-question
caching was slower. This does not reproduce a proprietary RLCD system.
[Public Qwen RLCD code review](research/rlcd-public-implementation-review.md) ·
[Jev architecture claims, SemIf and Jevlike](research/jev-architecture-source-review.md).

## Limits and provenance

Candidate probabilities are **uncalibrated and relative to the supplied choices**. Rewording or changing the choices can change the scores.

These demos use different models and protocols. Replays illustrate preserved episodes; they do not replace complete comparisons. The research has not established a novel architecture, a biological-wiring advantage or a new RL algorithm.

OpenJev is independent of [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [zhihz/openjev](https://github.com/zhihz/openjev) and [openjev.com](https://openjev.com/). It does not reproduce proprietary Jev architecture or weights.

The root MIT license applies except where component notices specify other terms, including GPL-3.0-only chess research files. Third-party data and model licenses apply separately.
