# What the public Qwen parallel-decoding implementation adds

Source inspection, September 18, 2026. The user's diagram describes a useful
inference optimization, but is not a verified diagram of TypeSafe's internals.
This review did not run the upstream engine or reproduce its speed claims.

The [upstream repository](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/tree/2af86848be75847ccb3553b0941cc51d6ef7e4e9)
was inspected at `2af86848be75847ccb3553b0941cc51d6ef7e4e9`.
[Source receipt and snapshots](../evidence/qwen-parallel-source-review-v1/receipt.json)
preserve the files inspected. Its MLX engine loads
`mlx-community/Qwen2.5-1.5B-Instruct-4bit`. The inspected file listing contains
application code, not new model weights or a training pipeline. Its RLCD alias
calls parallel inference. The repository name does not establish training with
TypeSafe's Reinforcement Learning for Calibrated Decisions.

## Source findings

| Finding | Evidence at the inspected revision | Consequence |
| --- | --- | --- |
| Shared context, parallel field suffixes | `core/engine_mlx.py:329-352` prefills once, repeats KV arrays across fields, then evaluates a padded batch | Useful candidate optimization; cache storage still scales with the field batch |
| Vocabulary projection precedes selection | `engine_mlx.py:351-366` gets model vocabulary logits, then indexes candidates | This implementation does not demonstrate a candidate-only output projection |
| Unfitted confidence | `engine_mlx.py:364-370` applies candidate softmax with default temperature 1 | Normalization alone does not establish calibration |
| Collision fallback changes probability semantics | `engine_mlx.py:377-419` greedily continues up to four tokens, heuristically matches a choice, can select the first choice by default, clamps winner confidence to at least 0.75 and spreads the remainder uniformly | Do not use these values as measured candidate probabilities or outcome forecasts |
| Collision behavior differs by backend | `engine_torch.py:89,139-157` reads the collision flag but only scores first-token IDs | Candidates sharing that ID receive identical scores; the MLX continuation behavior is absent |
| Work telemetry omits calls | `engine_mlx.py:448` reports one sequential forward pass even when the collision branch makes extra model calls | Count actual prefill, suffix and continuation calls in a new benchmark |
| Baseline includes formatting work | `core/prompt_builder.py:17-25` requests indented multiline JSON | The reported speed ratio is specific to that baseline, not an optimized decision scorer |

The MLX collision branch also slices a cache after right-padded suffixes without
cropping each field back to its true length. Continuation therefore receives
padding history for shorter fields. This is a source-level concern; no measured
impact is claimed here. Selecting a field's logits at its true suffix position
does not itself have that continuation problem under causal attention.

The upstream revision was checked again after the supplied screenshot and is
unchanged. Its [benchmark runner](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/blob/2af86848be75847ccb3553b0941cc51d6ef7e4e9/core/benchmark.py)
performs one timed comparison per preset after warmup. The README's example
also reports 148 tokens in 421.3 ms alongside 122.4 tokens/s, although those
first two values imply approximately 351 tokens/s. That inconsistency does
not prove the speedup false, but the example cannot serve as an internally
consistent measurement receipt. A new benchmark needs actual repeated runs
and complete timing records.

## Fit with OpenJev

Our [decision scorer](../src/openjev/decisions.py) already reads a final-position
candidate distribution without generating an answer. It maps descriptions to
verified unique single-token labels A-L, then returns stable caller IDs. It
also exposes candidate vocabulary mass and explicitly marks probabilities as
uncalibrated. Preserve those properties.

The missing optimization is shared-prefix caching and batched independent
question suffixes. The current API permits one to four questions and runs a
fresh prefill for each. A new implementation should retain exactly the same
tokenized prompts where possible, sharing only their actual common token
prefix. If a new prompt layout is needed, version it and test quality separately
from the cache optimization. Independent field marginals do not enforce
cross-field consistency; dependent decisions need staged conditioning or joint
candidates.

Before publishing a speed claim, compare serial and batched restricted scoring
on the same model, precision, complete prompts and candidate sets. Verify
logit/selection agreement, unequal suffix lengths, cache isolation and actual
work counts. Report cold and warmed end-to-end latency, memory, and all
supported question counts. Add optimized constrained JSON generation as a
separate baseline. Do not infer a 5.6-7x OpenJev gain from the upstream card.

The [implementation follow-up](../output/qwen-parallel-followup-review.md)
specifies four matched methods to separate batching from prefix reuse while
preserving the pinned model, candidate vocabulary mass and request timing
semantics. This benchmark remains proposed, not measured.

For research, keep choice preference separate from probability of an action's
outcome. A recurrent world model could forecast the latter, trained and tested
against observed outcomes. Compare accuracy, log loss/Brier score, calibration,
control utility and total computation on held-out episodes. Conformal sets
would need a stated target and appropriate calibration units; token softmax
does not provide coverage. Caching is an engineering baseline for that study,
not evidence of a biological or recurrent learning contribution.
