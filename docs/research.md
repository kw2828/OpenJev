# Jev research notes

Checked September 16, 2026. The user's [DEV article](https://dev.to/jamilxt/this-new-ai-model-refuses-to-write-text-that-is-exactly-why-it-runs-100x-faster-4he) and [Hacker News discussion](https://news.ycombinator.com/item?id=49717558) were discovery leads. Implementation decisions below use the vendor's documentation and source repositories.

## What the primary sources support

1. **The output interface is the central idea.** State and independent typed questions produce values and probability distributions. The primitives are Choice, Score, and Noul. Questions share input state and are evaluated independently in one request. This suggests keeping control flow in ordinary code and asking the model narrowly defined questions. [TypeSafe introduction](https://docs.typesafe.ai/introduction)

2. **The launch speed comparison has a specific workload.** The vendor reports large gains for its workflow evaluation, while describing input-length and comparison-wrapper caveats. Its Doom demonstration consumes structured textual game state rather than images, at about ten queries per second. Those results do not establish a universal 100x advantage or a pixel-based Doom agent. This project makes neither claim. [Launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

3. **There is an implementable API.** The documented route is `POST https://api.typesafe.ai/v1/systemone`, authenticated by a bearer key. A request includes state, model, and questions. Responses return answers under matching question IDs. Choice returns a selected option, probabilities, and confidence; Noul returns a number in [0,1]. OpenJev implements this contract with validation, timeouts, and a call limit. [API reference](https://docs.typesafe.ai/api), [official Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python)

4. **Confidence requires careful interpretation.** The vendor describes Choice/Score confidence as a statistic of the returned probability distribution. It is distinct from the probability attached to one answer. OpenJev preserves the vendor confidence field when using Jev; it does not relabel its own largest softmax score as calibrated confidence. A valid enum can still be the wrong decision. [Confidence documentation](https://docs.typesafe.ai/confidence)

5. **Public clients are not public model weights.** Inspection of the vendor's public repositories and documentation did not locate a downloadable Jev checkpoint or enough architectural and RLCD detail to reproduce training. The organization includes SDKs, an LLM-backed compatibility adapter, and unrelated/forked projects. A fork of another model is not evidence that Jev uses that model. This is a scoped search result, not proof that no additional materials exist. [TypeSafe GitHub](https://github.com/typesafe-ai), [System One LLM adapter](https://github.com/typesafe-ai/system-one-adapter-python)

6. **Independent open experiments already exist.** Jevlike describes scoring variable text options through option-to-context attention and explicitly distinguishes its model from Jev. It is useful related work, not an authoritative account of Jev's internals. OpenJev was implemented independently and uses a much smaller fixed-action numeric policy; no Jevlike source or weights were copied. [Jevlike](https://github.com/vinnylarouge/jevlike)

7. **Real Doom offers a reproducible environment.** ViZDoom exposes game state, labels, action buttons, seeds, and rewards. Its defense scenarios are bounded combat tasks with limited ammunition. OpenJev uses those environments without changing their action spaces, rewards, or difficulty. [Python quick start](https://vizdoom.farama.org/introduction/python_quickstart/), [scenario definitions](https://vizdoom.farama.org/environments/default/), [game state API](https://vizdoom.farama.org/api/python/game_state/)

## The engineering choice

OpenJev demonstrates the part we can implement and inspect: a shared small network outputs multiple finite decisions without autoregressive text generation, and normal code applies them to a real simulator. It ships its own trained weights and a full local path so access to a closed service is optional.

The tradeoff is substantial: the local model has no general language ability. It takes engineered numerical features and imitates a hand-written controller on synthetic data. The three mission presets are explicit categorical inputs. It has not learned new strategies through reinforcement learning. Its probabilities were not trained using TypeSafe's undisclosed RLCD recipe.

## What the evidence does and does not show

The saved 20-seed evaluation supports a narrow claim: the local policy can repeatedly kill enemies in Defend the Center and performs approximately like its teacher. The random baseline provides a basic sanity check. It does not establish generalization to unseen game mechanics, campaign navigation, visual understanding, optimal gameplay, calibrated uncertainty, or superiority to Jev or an LLM.

The API adapter has only offline contract tests because no TypeSafe key was available. To evaluate Jev fairly, use an explicitly authorized API budget, the same states and action rules, record latency and failures, and report service cost separately from local compute. A tiny numeric MLP and a general text-conditioned model are not comparable capabilities simply because both emit controls.
