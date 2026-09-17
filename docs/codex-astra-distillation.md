# Astra teacher through Codex

**Status: preparing a separate teacher run; no result yet.** This route uses the explicitly selected `gpt-6-astra` Codex model to label the existing 128 training examples, then trains the same MiniLM student used in the [original text pilot](text-distillation.md). It does not replace that pilot's unexecuted Responses API plan or fabricate API receipts.

## What stays matched

Reuse the frozen BoolQ and CLINC domain-routing selections, student architecture, three training seeds (17, 29, 43), three epochs, AdamW settings and final evaluation. The student receives context, question and candidate descriptions and learns a shared candidate score. The benchmark gold labels and evaluation examples are excluded from every teacher packet.

The original gold-label and untrained-head controls are retained as explicitly reused controls. The same 172 public evaluation examples have already been scored, so this is development evidence, not a new independent confirmation set. The teacher may have encountered these public benchmarks during pretraining.

## What changes

The teacher runs through a fresh Codex subagent with model `gpt-6-astra`, low reasoning and no inherited conversation, using batches of 32 training examples. Teacher-visible fields are opaque item IDs, context, question and candidates. The parent retains the mapping to source IDs. The worker may read its supplied input packet and write its output; it may not browse, inspect other workspace files or read benchmark labels.

Batching means examples share a teacher context. This differs from the original API plan's independent requests. Teacher supervision remains hard candidate IDs, not soft probabilities or rationales. Every required ID must appear exactly once and select a supplied candidate before student training can begin.

Preserve the frozen plan, sanitized request hashes, worker dispatch metadata, returned choices and output hashes. Model identity is recorded from the requested Codex model setting; this route does not supply an API response model field, API token usage or a billed dollar amount. Codex account usage is not claimed to be free. No Responses API call is made by this route.

## Interpretation

Compare each trained student fit with the reused controls on accuracy, class-balanced accuracy, negative log-likelihood and Brier score. Report teacher agreement with training gold labels only after all choices have been collected and fixed. Teacher training-label agreement is not held-out teacher accuracy.

Use the original descriptive screening rule: at least ten percentage points above the untrained-head control and within five points of gold supervision on each task. Passing would only warrant a larger study. This route trains a small text scorer from Astra's decisions; it does not fine-tune Astra or establish an RL, world-model or architectural novelty result.

The existing [license and benchmark limitations](text-distillation.md#licenses-and-scope) still apply. Keep raw passages, teacher packets and model weights local; publish aggregate results and provenance.
