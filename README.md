# OpenJev

**Context → questions and candidate answers → probabilities → your application's next step.**

OpenJev is a local decision interface and a collection of controlled learning experiments. Score choices with stable IDs, try text decisions in your browser, or explore recorded chess, Doom and robot policies.

[**Text demo**](https://kw2828.github.io/OpenJev/) · [**Chess replay**](https://kw2828.github.io/OpenJev/chess-candidate-replay.html) · [API](docs/decision-api.md) · [Setup](docs/project-reference.md) · [Experiment archive](research/experiment-index.md)

## Decision interface

Supply English context, your questions, and candidate IDs with descriptions. OpenJev returns each chosen ID and the relative probability of every supplied answer. Your application decides what to do next; the API itself executes no external action.

The local scorer uses a pinned Qwen model, maps candidates to single-token labels, and scores those labels without generating an explanation. Questions are independent. Stable IDs constrain the response format, not whether the choice is correct.

[Architecture and calibration notes](docs/decision-model-architecture.md) explain what public Jev-inspired implementations establish and which claims still need evidence.

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

The latest [public-history diagnostic](research/schedule-headroom-results.md)
finds larger average gains, but **does not admit a new model experiment**.
Ten fixed references face five fresh datasets. No model is retrained.

| Exact schedule tracking with the learned model | Lower mean regret |
|---|---:|
| Compared with the GRU filter | 40.95% |
| Compared with the global adapter | 21.86% |
| Compared with a two-mode static filter | 12.18% |

Every dataset favors exact tracking in those comparisons. However, normal-noise
regret rises to **2.55x the unchanged model's**. Even the reference supplied with
true world laws worsens normal-noise regret to **1.89x**. Each passes 14/15
conditions and fails the preservation requirement, so this adaptation direction
is closed for the tested setting.

[![All ten references across five fresh datasets, with all noise families and inference costs](https://raw.githubusercontent.com/kw2828/OpenJev/main/research/schedule-headroom-results/benchmark.png)](research/schedule-headroom-results.md)

[Open chart](https://raw.githubusercontent.com/kw2828/OpenJev/main/research/schedule-headroom-results/benchmark.png) ·
[All results and evidence](research/schedule-headroom-results.md) ·
[Protocol](research/schedule-headroom-protocol.md) ·
[Research directions and papers](research/decision-recurrence-calibration-reading.md).

Exact tracking receives the correct schedule prior and stores 288 probability
entries, versus the GRU's 28 state entries. Its measured inference takes **4.28x
as long**. These are diagnostic advantages and costs, not a new architecture.
All **123 tests** pass; the independent audit agrees with all 50 result rows.
The earlier [GRU comparison](research/reliability-memory-results.md) remains FAIL
at 8/13. Every outcome stays in the [experiment archive](research/experiment-index.md).

Separate engineering studies measured [1.20-1.35x training throughput](research/finite-joint-reuse-throughput-results.md)
from shared computation and [2.08x scoring speed](research/shared-prefix-results.md)
on four-question synthetic workloads. Neither reproduces proprietary RLCD.
[Public Qwen RLCD review](research/rlcd-public-implementation-review.md) ·
[Jev architecture claims](docs/decision-model-architecture.md).

## Limits and provenance

Candidate probabilities are **uncalibrated and relative to the supplied choices**. Rewording or changing the choices can change the scores.

These demos use different models and protocols. Replays illustrate preserved episodes; they do not replace complete comparisons. The research has not established a novel architecture, a biological-wiring advantage or a new RL algorithm.

OpenJev is independent of [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [zhihz/openjev](https://github.com/zhihz/openjev) and [openjev.com](https://openjev.com/). It does not reproduce proprietary Jev architecture or weights.

The root MIT license applies except where component notices specify other terms, including GPL-3.0-only chess research files. Third-party data and model licenses apply separately.
