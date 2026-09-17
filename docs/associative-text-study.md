# Adaptive associative text routing

This follow-up tests a sparse recurrent decision head over a frozen MiniLM sentence encoder. The encoder is still a transformer. This is an architecture experiment using established retrieval ideas, not a claim to have invented associative memory or a replacement language model.

The earlier 128-example text pilot was weak on BoolQ and unstable on routing. This study tests a different question: can a fast prototype decision plus selective associative retrieval preserve strong retrieval accuracy while using fewer memory searches? It uses the full known-intent CLINC training bank, not the tiny earlier training set. Any gain over that pilot would mix more data, a different task and a different architecture; the direct comparisons are only between arms in this study.

## Frozen comparison

- **Prototype:** cosine similarity to each of 150 intent centroids.
- **Nearest:** maximum cosine similarity per intent across all known training examples.
- **Recurrent:** retrieve 32 neighbors, refine the query twice using softmax associative retrieval anchored halfway to the original query, then search the whole bank again. Rejection confidence cannot exceed the original query's nearest-memory support.
- **Gated nearest / gated recurrent:** use prototype margin to decide whether to retrieve. Otherwise retain the prototype answer. Compare gates at the 25th, 50th and 75th development-margin percentiles.

All arms share the same frozen, pinned `all-MiniLM-L6-v2` encoder and training bank. No Astra labels, paid requests or encoder fine-tuning are used in this study. These heads recognize a learned intent inventory from labeled examples; they do not yet support arbitrary unseen candidate descriptions in the public OpenJev API.

Official CLINC validation data is divided equally within each class into a development-selection split and a calibration split. Development selects rejection thresholds, a score temperature, and one recurrent gate using balanced utility minus a cost penalty. Balanced utility is half correct-and-accepted in-scope accuracy plus half out-of-scope recall, avoiding a majority-class accuracy shortcut. Out-of-scope training examples are unused by every arm.

Confirmation uses official CLINC test data after removing all 88 previously evaluated routing contexts and normalized text duplicates with earlier splits. This makes it a filtered confirmation subset, not an official full-test benchmark result. All variants are specified before confirmation; only the development-selected recurrent gate drives the continuation rule. Public pretrained-model exposure remains possible.

Split-conformal sets use only in-scope calibration examples, targeting 90% marginal in-scope coverage. Report coverage and mean set size on in-scope confirmation examples, separately from rejection performance. This does not guarantee detection of unknown intents or coverage after arbitrary distribution shifts.

Continue only if the selected recurrent gate improves balanced utility over nearest retrieval at the same gate, is within one point of always-nearest utility with a paired lower bound of at least -0.01, and uses at most 0.5 full-bank searches per query. These are research screening criteria, not deployment guarantees. Report all failures. A positive result would still need random-gate/matched-compute controls and another environment before a novelty claim.

Memory-search counts include both whole-bank passes in recurrent retrieval. CPU head timing includes gate and recurrence, uses batches of 128 and four CPU threads, and excludes the shared encoder. Embedding time is recorded separately. A batched throughput measurement must not be called single-request latency or an end-to-end speedup.

## Run

The frozen text-study dependencies and pinned public source data from [text distillation](text-distillation.md) are prerequisites.

```sh
.venv/bin/python scripts/associative_text_study.py prepare \
  --source runs/text-distill-inputs/clinc.json \
  --prior runs/text-distill-v1/packet/packet.json \
  --out runs/associative-text-v1/packet --device mps
.venv/bin/python scripts/associative_text_study.py develop \
  --packet runs/associative-text-v1/packet --out runs/associative-text-v1/selection.json
.venv/bin/python scripts/associative_text_study.py confirm \
  --packet runs/associative-text-v1/packet --selection runs/associative-text-v1/selection.json \
  --out runs/associative-text-v1/confirmation
```

Output directories are immutable. Code, input and selection hashes are checked before execution. Raw texts, embeddings and per-item predictions stay local; public reports contain aggregates and hashes. The preceding Astra pilot and its frozen sources remain unchanged.

## Sources

[CLINC150 authors and data](https://github.com/clinc/oos-eval) · [Modern Hopfield networks](https://arxiv.org/abs/2008.02217) · [Conformal prediction beyond exchangeability](https://arxiv.org/abs/2202.13415). The recurrent update is inspired by existing associative retrieval; the anchoring, gating and support-preservation choices require ablation rather than novelty assertions.
