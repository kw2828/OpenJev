# Train a text student with Astra

This experiment trains a small English candidate scorer using **gpt-6-astra as a frozen teacher**. It is supervised hard-label distillation. It does not fine-tune Astra, use reinforcement learning, or transfer the Doom JEPA policy.

**Status:** pipeline implemented; Astra labeling awaits accessible API credentials. No Astra-trained model or efficacy result exists yet. The existing browser demo continues to use Qwen3-0.6B.

## Benchmarks

| Task | Training | Evaluation | Decision |
| --- | ---: | ---: | --- |
| [BoolQ](https://github.com/google-research-datasets/boolean-questions) | 64 | 84 | Answer yes/no from a question and passage |
| [CLINC150 domain adaptation](https://github.com/clinc/oos-eval) | 64 | 88 | Route a request to 10 domains or out of scope |

BoolQ draws training items from its official training split and evaluation items from its public validation split, balanced across yes/no. CLINC draws from official training and test splits. It maps the original 150 intents to their 10 documented domains, enumerates each domain's supported intents in its description, and adds out of scope. This is **an 11-way adaptation, not a full CLINC150 benchmark score**. Its training set contains six examples per domain and four out-of-scope examples; evaluation has eight per class.

Source revisions are pinned. Selection excludes exact normalized question/context duplicates and pairs longer than 512 student tokens before a deterministic sample. Full contexts are identical for teacher and student. Filtering, class balancing and public data limit generalization: neither sample represents production prevalence, and either pretrained model may already have encountered these datasets.

## Student and controls

The student starts from the 22.7M-parameter [all-MiniLM-L6-v2 encoder](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), adapted into a cross-encoder. For each question it encodes **context + question + one candidate description**, pools the nonpadding token states, and applies a shared scalar scoring head. A softmax across candidates gives the returned probabilities. They are uncalibrated candidate scores, not token probabilities or Astra's confidence.

Candidate IDs route results back to the caller; the student sees descriptions, not IDs. The same encoder/head supports request-time candidate sets without a fixed class-specific output layer. Costs grow with the number of candidates. This pilot does not establish transfer to unseen candidate vocabularies.

Three arms use the same architecture and three fixed initialization/training seeds:

1. **Untrained head:** pretrained encoder with a newly initialized scoring head, no updates. This is a weak starting-point control, not a competitive zero-shot LLM baseline.
2. **Gold labels:** all encoder parameters and the head trained on benchmark training labels.
3. **Astra labels:** identical training using Astra's selected candidate IDs. Gold labels and evaluation examples never enter teacher requests.

Each fitted arm uses three epochs, AdamW at 2e-5, weight decay 0.01, gradient clipping at 1.0 and one question per update. There is no early stopping, calibration fitting, best-seed selection or evaluation-based tuning. The frozen code and protocol must match the packet manifest before execution. Seeded MPS runs are not promised to be bitwise reproducible across hardware or library releases.

Report accuracy, macro class accuracy, per-class accuracy, categorical negative log-likelihood, multiclass Brier score, and per-question median/p95 latency for every fit. Paired accuracy intervals resample training fits and evaluation items within gold classes. The same 172 evaluation examples are reused across fits; three fits do not triple the test-set size. Latency includes tokenization and reading scores back to the CPU, after a training-item warmup; it excludes model loading. Training time includes tokenization and optimization, excludes pretraining, and is reported separately from teacher cost.

The descriptive continuation rule requires Astra supervision to improve mean accuracy over the untrained head by at least 10 percentage points on **each** task, and to finish within 5 points of gold supervision on each task. Passing this small pilot only warrants a larger confirmation with stronger baselines, unseen tasks/candidate sets and an independently collected evaluation. It is not evidence of research novelty or ICLR readiness.

## Reproduce

Use the repository's frozen environment and hosted-model dependencies. Input downloading uses a separate pinned script environment, leaving the project lockfile unchanged:

```sh
uv sync --frozen --extra hosted --extra dev
uv run --script scripts/fetch_text_sources.py --out runs/text-distill-inputs
.venv/bin/python -m openjev.research.text_distillation prepare \
  --sources runs/text-distill-inputs --out runs/text-distill-v1/packet
```

Preparation freezes the protocol, selected examples, source hashes, code hashes and runtime versions. Keep that packet unchanged. Raw passages stay under ignored `runs/`; the public report exports only aggregate metrics, selection IDs/hashes and provenance.

Review the bounded request plan before labeling:

```sh
.venv/bin/python -m openjev.research.text_teacher \
  --packet runs/text-distill-v1/packet --out runs/text-distill-v1/teacher --plan-only
```

Provide `OPENAI_API_KEY` through your local environment, or pass `--key-file` pointing to an accessible owner-only file. Never commit or paste credentials. The teacher uses the [official Astra Responses API](https://developers.openai.com/api/docs/models/gpt-6-astra), low reasoning, strict candidate-ID JSON, no tools, `store:false`, and at most 1,024 output tokens per request. The ceiling is $20 for 128 requests, checked against conservative input/output reservations before any request is sent. Reservations are estimates; actual usage and estimated cost are retained in local receipts.

```sh
.venv/bin/python -m openjev.research.text_teacher \
  --packet runs/text-distill-v1/packet --out runs/text-distill-v1/teacher

for arm in untrained gold astra; do
  .venv/bin/python -m openjev.research.text_distillation train \
    --packet runs/text-distill-v1/packet --out "runs/text-distill-v1/$arm" \
    --arm "$arm" --teacher runs/text-distill-v1/teacher --device mps
done
```

Use `--device cpu` or `--device cuda` on other machines. Every paid request gets a durable started record and terminal receipt; failures stop the run without retrying. Existing output directories are rejected. Interrupted runs must be audited rather than restarted into a new directory. A partial or wrong-model teacher run cannot become training supervision. Do not tune the protocol after opening evaluation results and call the next run a confirmation.

Checkpoints remain local under each arm's `seed-*` directories. They can be loaded with `CandidateStudent.load(path, device='cpu')` from `openjev.research.text_student` and used with `decide_record` or the existing `DecisionRequest` via `score`. This research adapter is not wired into the public API; it deliberately does not fabricate vocabulary token-mass fields.

Generate an aggregate report and figure in a new directory (requires Matplotlib in the execution environment):

```sh
.venv/bin/python -m openjev.research.text_report \
  --packet runs/text-distill-v1/packet --runs runs/text-distill-v1 \
  --teacher runs/text-distill-v1/teacher --out evidence/text-distillation-v1
OPENJEV_TEST_TEXT_ENCODER=1 .venv/bin/python -m pytest tests/test_text_distillation.py -q
```

Omit `--teacher` for a controls-only report. Such a report explicitly marks the Astra comparison as unassessed. Never substitute gold or synthetic labels while calling an arm Astra-trained.

## Licenses and scope

BoolQ is CC-BY-SA-3.0, CLINC150 is CC-BY-3.0, and the MiniLM checkpoint is Apache-2.0; retain their attribution and applicable conditions. The student's weights derive from MiniLM and should not be described as wholly original MIT weights. Review the teacher provider's current terms before redistributing teacher outputs or a derived checkpoint. Public benchmark exposure, tiny training sets and the weak untrained-head baseline prevent broad model-quality claims.
