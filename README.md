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

[![First scheduled candidate-policy game, drawn by repetition](docs/assets/chess-candidate-game-001.gif)](https://kw2828.github.io/OpenJev/chess-candidate-replay.html)

Four small policies score legal moves across **twelve fits and 288 games**. Exact next-board differences raise ordinary move agreement from **31.75% to 33.15%**, but cost more computation and fail the engine-loss and game-score criteria. The GIF is the first scheduled game, not a selected win. No learned world model or Elo rating is established.

[Results, all checkpoints and controls](docs/chess-candidate.md) · [ChessBench transfer](docs/chessbench-transfer.md) · [Connectome comparison](docs/chess-connectome.md) · [Graph-quality comparison](docs/chess-pin-quality.md).

## Doom

[![First JEPA-trained policy and first evaluation seed](docs/assets/jepa-policy-181000-preview.gif)](docs/assets/jepa-policy-181000.gif)

**13 kills in 19.26 game seconds**, from the first fit and first evaluation seed. JEPA supplies training rewards; a small policy plays this recording. The full nine-method comparison did **not** establish a reliable JEPA advantage. This is one illustrative episode, not an inference-speed benchmark.

[Results and limits](docs/jepa-rl-study.md) · [Recording receipt](docs/assets/jepa-policy-181000.json) · [Other Doom controllers](docs/project-reference.md).

## Robot reaching

[![Preselected first robot case with persistent memory and trained reset controls](evidence/reacher-geometry-memory-v1/report/fixed-case-replay.gif)](research/reacher-geometry-memory.md)

The stronger memory comparison **failed its rule: 24/25 checks passed**. Persistent GRU lowers control cost by **4.44% / 2.94%** versus a separately trained two-observation controller; the rule requires at least 3% on both sensing-gap panels. Supplied-physics controllers still perform better.

The GIF is from the earlier, weaker-control study. The [stronger comparison](research/reacher-two-observation-control.md) retains all nine models, five references and complete costs. Neither establishes a new architecture or biological-wiring advantage.

## Research status

Memory updated by observed prediction errors **failed its development rule:
6/13 conditions passed** across 18 fits. Its later decision-cost gap was
**54.1% higher** than the best control in one setting and **14.3% lower** in the
other. The independent audit completed; the held-out test remains unused.

[![All eight memory methods and three fit seeds, including the failed candidate](research/otto-query-memory-dev-results/query-memory-dev.png)](research/otto-query-memory-results.md)

[Results and costs](research/otto-query-memory-results.md) ·
[Every method and seed](research/otto-query-memory-dev-results/query-memory-dev.md) ·
[All models and evidence](https://github.com/kw2828/OpenJev/releases/tag/otto-query-memory-dev-v1).
This fixed-path comparison establishes no autonomous or architecture advantage.

The next [Bayesian memory comparison](research/otto-residual-estimator-status.md)
passes 531 implementation tests and its capacity check. Fresh-data evaluation
is registered; no new performance gain is established yet.

The [earlier protected-readout failure](research/otto-protected-readout-results.md),
other comparisons and incomplete runs remain in the
[experiment archive](research/experiment-index.md).

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
