# OpenJev

**Context → questions and candidate answers → probabilities → your application's next step.**

Score choices you define at request time, with stable candidate IDs. Explore text decisions in your browser and locally trained policies playing Doom.

[**Try the demo**](https://kw2828.github.io/OpenJev/) · [API examples](docs/decision-api.md) · [Benchmarks](docs/benchmarks.md) · [Setup and project reference](docs/project-reference.md)

## See it work

[![Local Doom policy trained with V-JEPA 2 rewards](docs/assets/jepa-policy-181000-preview.gif)](docs/assets/jepa-policy-181000.gif)

[Open animation](docs/assets/jepa-policy-181000-preview.gif) · [Original recording](docs/assets/jepa-policy-181000.gif)

**13 kills in 19.26 game seconds**, first fit and first evaluation seed. JEPA scored training clips; a small policy plays this recorded game. [Recording receipt](docs/assets/jepa-policy-181000.json).

[![Browser decision playground before loading its model](docs/assets/browser-playground.jpg)](https://kw2828.github.io/OpenJev/)

The free browser demo runs Qwen3-0.6B through WebGPU. It is separate from the Doom policy and the research students below.

## Run locally

Apple Silicon macOS, Python 3.11-3.13:

```sh
git clone https://github.com/kw2828/OpenJev.git
cd OpenJev
uv sync --frozen --extra language
uv run --extra language openjev setup
uv run --extra language openjev serve
```

Open **http://127.0.0.1:8000**. Local text scoring uses Qwen3-4B. [Linux, Docker and hosting](docs/hosting.md).

## Train a text model with Astra

Astra labeled 128 training examples through Codex; three MiniLM student fits are complete. Mean accuracy reached **62.1% on domain routing** versus 9.5% untrained, but **51.2% on BoolQ** remained near chance. The combined continuation rule failed.

[Results, benchmark scope and local checkpoints](docs/codex-astra-distillation.md). This uses previously scored development examples. The separate [Responses API plan](docs/text-distillation.md) remains unrun.

## Research results

![Recurrent world-model experiment, controls and training costs](evidence/recurrent-world-v1/world-model-results.png)

- **Recurrent world models:** the [RL studies](docs/recurrent-world-model-study.md) found no advantage. In a [supervised memory diagnostic](docs/associative-learnability.md), associative stores retain 100% versus 50% for GRU on the longest route. This uses only four examples and forced navigation; selective writes tie simpler global writes.
- **Text routing:** 95.35% development accuracy on 150 intents. The recurrent gain fell below our continuation threshold. [Study](docs/corrective-associative-study.md).
- **Rule reasoning:** a 225-parameter operator scores 100% on 949 new development questions using a restricted English parser. A fixed logical solver matches it; novelty is unproven. [Study and runnable model](docs/bound-rule-study.md).
- **Doom:** history-PPO improved Center combat. JEPA + RL reached 12.40 mean Center kills without establishing an advantage over GRPO or pixel similarity; Line policies matched always-fire. [PPO/DQN](docs/rl-study.md) · [JEPA + RL](docs/jepa-rl-study.md).

[All charts and limitations](docs/benchmarks.md) · [Paper draft](output/pdf/openjev-rl-paper.pdf) · [LaTeX](paper/README.md)

Scores are uncalibrated and relative to the supplied choices. Doom uses structured observations and restricted controls. [Model card](docs/language-model-card.md) · [Research notes](docs/research.md).

Independent of [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [zhihz/openjev](https://github.com/zhihz/openjev) and [openjev.com](https://openjev.com/). No proprietary RLCD reproduction or new RL algorithm is claimed. MIT for project code and original weights; third-party licenses still apply.
