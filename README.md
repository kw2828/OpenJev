# OpenJev

**Context → questions and candidate answers → probabilities → your application's next step.**

OpenJev scores choices you define at request time. Use it for text routing, bounded decisions and a local Doom demo. Candidate IDs stay stable, so your application can use the result directly.

[**Try the browser demo**](https://kw2828.github.io/OpenJev/) · [API](docs/decision-api.md) · [Benchmarks](docs/benchmarks.md) · [Detailed reference](docs/project-reference.md)

## See it work

![Local Doom policy trained with V-JEPA 2 rewards](docs/assets/jepa-policy-181000.gif)

First training fit, first evaluation seed: **13 kills in 19.26 game seconds**. JEPA scored training clips; a small policy controls this recorded game. [Recording receipt](docs/assets/jepa-policy-181000.json).

[![OpenJev browser decision playground](docs/assets/browser-playground.png)](https://kw2828.github.io/OpenJev/)

The free browser playground runs Qwen3-0.6B locally through WebGPU. The screenshot shows its initial state before loading. Browser inference and the recorded Doom policy are separate models.

## Run locally

Apple Silicon macOS, Python 3.11-3.13:

```sh
git clone https://github.com/kw2828/OpenJev.git
cd OpenJev
uv sync --frozen --extra language
uv run --extra language openjev setup
uv run --extra language openjev serve
```

Open **http://127.0.0.1:8000**. The Mac text scorer uses a pinned Qwen3-4B model. For Linux/Docker and free hosting options, see [deployment instructions](docs/hosting.md).

Score a request without starting the server:

```sh
uv run --extra language openjev decide examples/support.json
```

Supply a context, a question and 2-12 candidates. Receive the selected ID and probabilities for every candidate. [Request and response examples](docs/decision-api.md).

## What we have measured

- **Text:** the original eight development checks were smoke tests. The Astra teacher/student benchmark is documented in [text distillation](docs/text-distillation.md); its status and results are kept separate from Doom.
- **Doom:** history-PPO improved Center combat in a frozen comparison. On Line, learned policies matched always-fire. [PPO/DQN study](docs/rl-study.md).
- **JEPA + RL:** 27 fits and 992 evaluation episodes. Pretrained JEPA had the highest Center mean, **12.40 kills**, but did not establish a reliable advantage over GRPO or pixel similarity. [Full comparison](docs/jepa-rl-study.md).

[All charts and earlier results](docs/benchmarks.md) · [Paper draft](output/pdf/openjev-rl-paper.pdf) · [LaTeX](paper/README.md)

## Limits

Scores are **uncalibrated probabilities relative to your supplied choices**, not guarantees of correctness. The text scorer is a pretrained transformer; the Doom policies use structured game observations and restricted controls. See the [text model card](docs/language-model-card.md) and [research notes](docs/research.md).

OpenJev is independent of [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [zhihz/openjev](https://github.com/zhihz/openjev) and [openjev.com](https://openjev.com/). It does not reproduce proprietary RLCD or claim a new RL algorithm.

MIT license for this project's code and original trained weights. Third-party models, datasets and game assets retain their own licenses. [Detailed setup, tests and project map](docs/project-reference.md).
