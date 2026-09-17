# Astra teacher through Codex

**Completed: 128 teacher labels and three trained students. The continuation rule failed on BoolQ.** This route used the explicitly selected `gpt-6-astra` Codex model to label the existing training examples, then trained the same MiniLM student used in the [original text pilot](text-distillation.md). It is separate from that pilot's unexecuted Responses API plan.

![Astra-supervised text students and reused controls](../evidence/codex-astra-text-v1/student-results.png)

## Results

Accuracy is averaged across all three final fits on 84 BoolQ questions and 88 CLINC domain-routing examples. These are balanced, filtered benchmark subsets and previously scored development examples, not full official benchmark results.

| Supervision | BoolQ mean | CLINC routing mean | CLINC by seed 17 / 29 / 43 |
| --- | ---: | ---: | --- |
| Untrained head, reused | 51.98% | 9.47% | 12.50% / 7.95% / 7.95% |
| Gold labels, reused | 50.40% | 60.23% | 76.14% / 36.36% / 68.18% |
| Astra through Codex | 51.19% | 62.12% | 76.14% / 44.32% / 65.91% |

Routing improved by 52.65 percentage points over the untrained head, but the 1.89-point difference from gold supervision has an exploratory paired interval of -3.41 to 9.09 points. It does not establish an advantage over gold labels. BoolQ changed by -0.79 points from the untrained control. All three Astra-supervised BoolQ fits scored 43/84.

Out-of-scope routing remains weak: mean accuracy is only 8.33% on the eight out-of-scope development examples per fit. The overall routing score does not support reliable abstention.

The fixed rule required at least a ten-point gain over the untrained head on each task and performance within five points of gold supervision. Three of four checks passed; the BoolQ improvement check failed. The aggregate result therefore does not justify advancing this student as a general text decision model.

Astra's fixed training choices agreed with **57/64 BoolQ labels and 64/64 CLINC labels**. This is agreement on training data, not held-out teacher accuracy. Seven changed BoolQ training labels also affect the shared encoder used for routing; the slight routing difference cannot be attributed to better CLINC labels.

The student fits used 1,152 optimizer updates and 63.55 seconds of local training in total, excluding pretrained model creation, teacher work, loading and evaluation. Historical control timings and concurrent PPO work prevent interpreting this as a matched speed comparison. Student NLL/Brier were 0.696/0.503 on BoolQ and 1.437/0.569 on routing; calibration remains unestablished.

[All metrics and exploratory intervals](../evidence/codex-astra-text-v1/summary.json) · [Provenance](../evidence/codex-astra-text-v1/provenance.json) · [Frozen plan metadata](../evidence/codex-astra-text-v1/plan-metadata.json)

## What stays matched

Reuse the frozen BoolQ and CLINC domain-routing selections, student architecture, three training seeds (17, 29, 43), three epochs, AdamW settings and final evaluation. The student receives context, question and candidate descriptions and learns a shared candidate score. The benchmark gold labels and evaluation examples are excluded from every teacher packet.

The original gold-label and untrained-head controls are retained as explicitly reused controls. The same 172 public evaluation examples have already been scored, so this is development evidence, not a new independent confirmation set. The teacher may have encountered these public benchmarks during pretraining.

## What changes

The teacher ran through four fresh Codex subagents with model `gpt-6-astra`, low reasoning and no inherited conversation, using batches of 32 training examples. Teacher-visible fields were opaque item IDs, context, question and candidates. The parent retained the mapping to source IDs. Workers were instructed to read their supplied input packet and write their output, without browsing, inspecting other workspace files or reading benchmark labels.

Batching means examples share a teacher context. This differs from the original API plan's independent requests. Teacher supervision remains hard candidate IDs, not soft probabilities or rationales. Every required ID must appear exactly once and select a supplied candidate before student training can begin.

Preserve the frozen plan, sanitized request hashes, worker dispatch metadata, returned choices and output hashes. Model identity is recorded from the requested Codex model setting; this route does not supply an API response model field, API token usage or a billed dollar amount. Codex account usage is not claimed to be free. No Responses API call is made by this route.

## Interpretation

Compare each trained student fit with the reused controls on accuracy, class-balanced accuracy, negative log-likelihood and Brier score. Report teacher agreement with training gold labels only after all choices have been collected and fixed. Teacher training-label agreement is not held-out teacher accuracy.

Use the original descriptive screening rule: at least ten percentage points above the untrained-head control and within five points of gold supervision on each task. Passing would only warrant a larger study. This route trains a small text scorer from Astra's decisions; it does not fine-tune Astra or establish an RL, world-model or architectural novelty result.

The existing [license and benchmark limitations](text-distillation.md#licenses-and-scope) still apply. Keep raw passages, teacher packets and model weights local; publish aggregate results and provenance.

## Run the trained student locally

The three saved checkpoints are under `runs/codex-astra-text-v1/student/seed-{17,29,43}` in the execution checkout. They are not bundled in GitHub or used by the browser demo. This example loads seed 17, selected by its fixed order rather than its score:

```python
from openjev.research.text_student import CandidateStudent

model = CandidateStudent.load("runs/codex-astra-text-v1/student/seed-17", device="cpu")
answer = model.decide_record({
    "context": "The museum is closed on Mondays.",
    "question": "Is the museum open on Monday?",
    "candidates": [
        {"id": "yes", "description": "Yes"},
        {"id": "no", "description": "No"},
    ],
})
print(answer["choice"], answer["probabilities"])
```

This illustrates the interface, not an additional benchmark. Scores are uncalibrated softmax probabilities over the supplied candidates.

## Reproduce the route

Use the [original data preparation and controls](text-distillation.md#reproduce), then prepare a fresh teacher directory with `scripts/codex_astra_distillation.py prepare`. The parent must dispatch each saved request through Codex with its recorded model, reasoning setting and fresh conversation, then record the actual worker identity with `accept`. The script never dispatches a model by itself and rejects retries or edited responses. All 128 choices must be sealed before training.

For the completed local teacher directory, the executed training and reporting commands were:

```sh
.venv/bin/python scripts/codex_astra_distillation.py train \
  --study runs/codex-astra-text-v1/teacher \
  --out runs/codex-astra-text-v1/student --device mps
.venv/bin/python scripts/codex_astra_distillation.py report \
  --study runs/codex-astra-text-v1/teacher \
  --run runs/codex-astra-text-v1/student \
  --out runs/codex-astra-text-v1/report
```

Existing result directories are immutable; use fresh paths for a separately frozen study. Code and source hashes were frozen in `a382554` before teacher dispatch. All three final student weights were saved before evaluation began.
