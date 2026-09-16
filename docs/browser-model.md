# DecisionTics browser playground

[Open the free playground](https://kw2828.github.io/DecisionTics/)

The static page runs language inference on the visitor's GPU, through a dedicated Web Worker. It accepts one English question, text context and two to eight candidate IDs/descriptions, then returns all candidate scores and an exportable receipt. The recorded Doom episodes use separate local models. This page does not run a live Doom engine or the memory-ablation policy.

## Exact model and readout

- Runtime: [WebLLM 0.2.85](https://github.com/mlc-ai/web-llm/tree/v0.2.85), loaded from a versioned jsDelivr module.
- Model: `mlc-ai/Qwen3-0.6B-q4f16_1-MLC`, pinned revision `8c14ce481d4c692769976ad52afea453a102df19`.
- Token labels A-H were verified as IDs 32-39 in that revision's tokenizer.
- The official runtime catalog supplies the WebGPU model library; the model revision pins weights/tokenizer assets, while the runtime version pins the catalog reference. Runtime/CDN availability is still an external dependency.
- `extra_body.enable_thinking=false` asks this pinned runtime to prefill Qwen's empty thinking block. `LogitProcessor.processLogits` captures raw first-answer-position logits before sampling. The runtime samples and discards one token; the UI does not parse a generated explanation or generated probability numbers.
- Candidate scores use a numerically stable softmax over the supplied label logits. Full-vocabulary candidate-label mass is also reported. No logit bias, top-logprob truncation, or grouping approximation is used.
- Scores remain conditional, uncalibrated label preferences, not probabilities of successful actions. Candidate order and prompt wording can affect them.

## Requirements and limits

Use a current WebGPU-capable desktop browser whose adapter supports `shader-f16`. Allow approximately 350 MB of model downloads and about 1.5 GB of GPU memory. Downloads start only after clicking Load; browser caching can reduce subsequent downloads. The load button performs an explicit capability check. Cancellation terminates the worker; scoring then requires reloading the model.

The model uses a 4,096-token context configuration. Input fields have character limits, but these are not tokenizer guarantees. Requests that exceed model limits return a visible runtime error, rather than silently truncating context. First scoring may include compilation; displayed time covers work inside the model worker, not page load or model download. There is no controlled speed comparison in this UI.

Inputs stay in the page and worker; this app does not upload them to an inference API, save them to browser storage, or use analytics. Model weights and runtime assets come from Hugging Face, GitHub-hosted model libraries and jsDelivr. Those hosts receive ordinary asset requests and may log network metadata. Downloaded weights may remain in the browser cache. JSON export includes the user's input and only occurs on request.

The existing Python CLI remains `openjev` for compatibility. The browser implementation is independent of its MLX/Transformers backends, and those backends' results do not establish the quality of this smaller model.

## Reproduce and inspect

```sh
python3 -m http.server 9876 --directory docs
node tests/browser-core.mjs
```

Open the local server in a supported browser. The pure scoring checks exercise invalid requests, duplicate IDs, large logits, candidate normalization and candidate mass. Browser smoke evidence is recorded separately after actual model loading and scoring; code checks alone do not prove GPU compatibility.

Source files: `docs/decision-core.mjs`, `docs/model-worker.mjs`, `docs/playground.mjs`. The scorer uses the documented [WebLLM logit processor interface](https://github.com/mlc-ai/web-llm/blob/v0.2.85/src/types.ts). The existing [openjev.com demo](https://openjev.com/) informed the browser-hosting direction; its implementation was not copied into this project.

## Observed browser smoke

The actual desktop Chrome run loaded the model and exported two scored requests. The support example selected billing; the empty-ammunition example incorrectly selected fire. Both retained candidate IDs and normalized scores, but successful integration does not imply semantic reliability. Duplicate IDs were rejected before inference and releasing the model terminated its worker. [Smoke receipt](../evidence/browser-smoke.json) · [Support export](../evidence/browser-support-smoke.json) · [Incorrect game export](../evidence/browser-game-smoke.json). A request-format issue found before release is preserved in the [correction receipt](../evidence/browser-preflight-correction.json).
