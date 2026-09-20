# What the Qwen RLCD reference contributes to OpenJev

Source review, September 18, 2026. This is a static implementation review, not
a reproduced benchmark or new performance result. Inspected Hugging Face model
revision `2af86848be75847ccb3553b0941cc51d6ef7e4e9` and the linked Space metadata
at revision `2cb1107ab79b2120a1c35984cf57993af5356b63`. No external code was
executed, weights downloaded or model inference performed.

## Artifact identity

[harshatheg/Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD)
currently contains application and inference source, not model weights, tokenizer
files or a model configuration. No RL training recipe is included. The MLX
engine loads `mlx-community/Qwen2.5-1.5B-Instruct-4bit`; the PyTorch default is
`Qwen/Qwen2.5-1.5B-Instruct`. The name does not establish a separately trained
1B RLCD checkpoint. See the [pinned file tree](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/tree/2af86848be75847ccb3553b0941cc51d6ef7e4e9)
and [MLX source](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/blob/2af86848be75847ccb3553b0941cc51d6ef7e4e9/core/engine_mlx.py).

The inference idea is useful: prefill shared context once, replicate the KV
cache across fields, and batch their suffixes. It avoids generating complete
JSON strings. The card's reported 5.6-7.0x speedups compare this with autoregressive
JSON generation on its hardware. They are not measurements against OpenJev's
existing zero-generation scorer.

The shared-prefix diagram in the supplied social post describes this inference
pattern, not a disclosed TypeSafe architecture. The implementation still computes
ordinary full-vocabulary logits before slicing candidate entries. Each field
sees the common context and schema, but not other fields' selected answers;
parallel field evaluation is not joint constraint solving.

## Probability and accounting issues

[The schema compiler](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/blob/2af86848be75847ccb3553b0941cc51d6ef7e4e9/core/schema.py)
removes a common character prefix, then takes the first token of each candidate
remainder. This need not represent full multi-token answer probabilities.

In the MLX overlapping-token branch, the engine greedily generates up to four
tokens, heuristically matches a candidate and falls back to the first choice
if matching fails. It then clamps the selected confidence to at least .75 and
spreads the remaining mass uniformly over other choices. Those values are not
shown to be empirically calibrated probabilities. The same return object reports zero
generated tokens and one sequential forward pass even when that branch makes
additional model calls.

[The PyTorch path](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/blob/2af86848be75847ccb3553b0941cc51d6ef7e4e9/core/engine_torch.py)
always scores the compiled first-token IDs. It does not implement the MLX
collision branch, so candidates with the same ID receive identical logits.
Neither ordinary restricted softmax nor a manually clipped confidence establishes
calibration. Valid assembled JSON also does not establish answer accuracy or
cross-field consistency.

These are source-level findings. We did not run either backend or validate the
published latency numbers.

## The useful comparison for OpenJev

Our [decision scorer](../src/openjev/decisions.py) already maps candidates to
verified single-token letters, scores their logits without text generation,
returns stable IDs and explicitly labels the probabilities uncalibrated. It
also reports total vocabulary mass assigned to allowed letters. Its current
MLX and CPU implementations recompute prompts sequentially across questions.

A fair improvement is shared-prefix inference on the same pinned Qwen model:

- Preserve complete prompt token sequences, model revision, quantization,
  candidate labels and output semantics.
- Find the longest common token prefix, prefill once, and use independent
  suffix caches with correct positions and attention masks.
- Compare one, two and four questions; short and long contexts; and two, four
  and twelve candidates. Keep the existing scorer as the direct baseline.
- Report median/p95 end-to-end latency, peak memory, top-choice agreement and
  maximum probability difference. Charge tokenization, cache replication,
  synchronization and probability readback.
- Preserve candidate-token mass and reject ambiguous label tokenization.
  Any accuracy or calibration claim needs a separate labeled evaluation.

One bounded parity/speed pilot would use 54 fixed English requests: three question
counts, two context lengths, three candidate counts and three requests per cell,
with five paired warm repetitions. Freeze those requests before timing. This
does not replace a labeled accuracy or calibration evaluation.

Follow-up: the [shared-prefix experiment is now complete](shared-prefix-results.md).
It measured a **2.08x median paired speedup** for four-question synthetic requests
against OpenJev's serial scorer. Single-question caching was slower. All selected
IDs were preserved, but the questions had unusually high margins, so this does
not establish accuracy or calibration on difficult inputs. The production
scorer remains unchanged. This result does not reproduce the upstream comparison
against autoregressive JSON generation.

Decoder KV reuse does not transfer directly to our separate bidirectional text
student. It supplies an inference baseline for OpenJev; it provides no evidence
about connectome wiring, JEPA objectives, recurrent world models or a new RL method.
