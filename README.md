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

The [latest recurrent comparison](research/finite-reuse-learning-results.md)
passes **all 19 predeclared continuation conditions**: nine fits, three seeds,
and exactly 1,024 history-learning plus 3,072 joint updates per fit. The constrained
transition lowers mean four/eight-step decision regret by **99.13% / 99.53%**
against the original free model and **99.11% / 99.51%** against the initially
matched free model. Every paired seed improves; two weak fits in each control
drive much of the large average gap.

[![Every recurrent fit: four- and eight-step decision regret and full training time](https://raw.githubusercontent.com/kw2828/OpenJev/main/research/finite-reuse-learning-results/benchmark.png)](research/finite-reuse-learning-results.md)

[Open recurrent chart](https://raw.githubusercontent.com/kw2828/OpenJev/main/research/finite-reuse-learning-results/benchmark.png) ·
[Complete evidence](research/finite-reuse-learning-results/evidence.tar.gz) ·
[Protocol](research/finite-reuse-learning-protocol.md).

Mean fitting time is **23.86 seconds**, versus **22.35 / 22.14 seconds** for the
controls. All 229 qualification tests pass and the independent saved-output
audit agrees. This is a local result in a small synthetic world with task-derived
constraints and privileged starting readouts. Untouched replication, a scenario
shift and a second environment are still needed. It establishes neither
biological-wiring benefits nor calibrated text probabilities.

The [earlier equal-time comparison failed](research/finite-rounded-learning-results.md),
and the [first equal-update attempt stopped before scientific training](research/finite-update-learning-stop-results.md).
The new study uses fresh data and seeds; it does not isolate why those outcomes differ.
The [integrated computation-sharing benchmark](research/finite-joint-reuse-throughput-results.md)
now measures **1.20x / 1.20x / 1.35x median training throughput** across the
three models. All nine measured pairs favor reuse; 78 tests and the independent
saved-state audit pass. This is a training-speed result on one machine, with
no new task-performance claim.

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
