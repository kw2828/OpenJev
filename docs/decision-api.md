# Decision API

`POST /api/decide` accepts English-focused text and caller-defined questions. `GET /api/model` reports lazy model loading/scoring status. `openjev decide request.json` exposes the same contract without a server. Interactive OpenAPI documentation is available at `/docs`; the POST endpoint additionally needs the `X-OpenJev: 1` header.

## Request

- `context`: nonblank string, at most 12,000 characters. Serialize application state to text yourself.
- `questions`: one to four questions evaluated independently and sequentially.
- Each question has a unique `id`, a nonblank `question` of at most 2,000 characters, and two to twelve `candidates`.
- Each candidate has an `id` unique within its question and a nonblank `description` of at most 1,000 characters. Descriptions differing only by whitespace or capitalization are rejected as duplicates.
- IDs are 1-64 characters: an ASCII letter or digit first, followed by ASCII letters, digits, underscore, dot, colon or hyphen. IDs are preserved in outputs and never enter the model prompt.
- Unknown fields are rejected. There is no caller-selected model path, remote URL, shell command or execution endpoint.
- The rendered input for each question is limited to 4,096 tokens. All questions are tokenized and checked before any question is scored. Over-limit inputs fail explicitly; context is never silently truncated.

All questions are mutually exclusive **choice** questions. A two-candidate question supplies a binary distribution. Independent properties should be separate questions, not mutually exclusive candidates in one question. This release has no continuous score, multilabel head or automatic abstention. Add a caller-defined uncertainty candidate when needed.

See [support.json](../examples/support.json) for a complete two-question request.

## Response

Top-level metadata: `model`, `revision`, `backend`, `protocol`, `protocol_sha256`, `calibration`, `questions_sequential`, `generated_tokens`, `latency_ms`, and `answers`.

Each answer contains:

| Field | Meaning |
|---|---|
| `id` | The caller's question ID |
| `choice` | Highest-scoring caller candidate ID |
| `probabilities` | Mapping of every candidate ID to its relative probability; sums to one |
| `candidate_token_mass` | Probability mass on all candidate-letter tokens within the full vocabulary |
| `entropy_nats` | Shannon entropy of the relative candidate distribution |
| `input_tokens` | Number of tokens in this question's rendered prompt |
| `latency_ms` | Forward scoring and materialization time for this question |

Top-level latency includes lazy loading and request preparation, but excludes waiting for the service's worker. HTTP end-to-end time can be longer. There is one model worker, with capacity for one running and one waiting request. Doom shares this service with the playground. At capacity, requests fail rather than grow an unbounded queue.

An answer with 0.99 relative probability can still have low candidate-token mass or be wrong. Entropy is not semantic entropy, and token mass is not a calibrated out-of-distribution detector. Probabilities and entropy can change when candidates are added, removed or reworded. Exact candidate reordering leaves the rendered prompt unchanged because descriptions are sorted before assigning letters. Stable ordering does not remove lexical/position bias.

## Scoring

The pinned Qwen instruction model receives context, the question, and descriptions associated with letters A-L. Each letter must be one unique tokenizer token. The model performs a prefill, projects the final hidden position to vocabulary logits, and a softmax restricted to the candidate letters yields relative probabilities. The runtime maps letters back to caller IDs. No output tokens are decoded and no rationale is generated. All quantities come from the same frozen model; no teacher or rule fallback is used.

Every question gets a fresh prefill. Shared-context caching, batching and persistent KV reuse are future optimizations. The checkpoint is a third-party MLX quantization of Qwen, not a newly trained OpenJev foundation model or a numerically identical full-precision copy.

## Errors and execution boundary

- 403: missing request header or cross-origin request.
- 422: malformed/ambiguous request or token limit exceeded.
- 429: inference capacity reached; caller decides whether to try later.
- 503: language runtime or local weights unavailable; run setup.
- 500: unexpected model failure; no fabricated answer is returned.

There are no automatic inference retries. Context and candidates can still prompt-inject or confuse a language model despite their structured rendering. Stable output IDs constrain syntax, not semantic correctness. This API performs no external action. Application adapters own execution and constraints.

The server is loopback-only with Host/Origin checks, not a multi-user authenticated service. Setup contacts Hugging Face to download the pinned public artifact. Model inference is local and uses the cached revision. The application does not persist arbitrary playground requests; exported JSON is saved only when the user requests it in the browser. Published test receipts contain authored synthetic fixtures only.
