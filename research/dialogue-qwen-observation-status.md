# Stronger Qwen observation comparison: cost pilot admitted

The [prospective protocol](dialogue-qwen-observation-protocol.md) fixes two
arms on all 7,819 existing development rows: current exchange and four public
exchanges, with identical correct previous values and public lexical flags.
The comparison has not produced a model-quality result.

The first model-free preparation attempt stopped because an unrestricted
local cache lookup demanded absent `README.md` and `.gitattributes` files.
It made no model calls. The required weights and tokenizer assets were already
present. [Failure receipt](../output/dialogue-qwen-observation-v1/preparation-01/failed.json).

The preparation lookup now uses the same file filter as the production scorer,
while still checking all eleven model-file hashes. No model, task wording,
cohort, candidate, or numerical-run rule changed. A separate metadata preparation
preserves the failed attempt. Full inference has not started.

The second preparation completed in **12.96 seconds**, with **882,016,256 bytes**
peak RSS and zero model calls. It froze **7,444 batches and 15,638 decisions**.
Current-exchange prompts contain 581-1,139 tokens; four-exchange prompts contain
612-1,292. All 7,819 rows and every candidate are retained. The cost-only pilot
has 28 batches selected before inference, including both arms of the longest
prompt groups. An independent model-free audit now agrees on all 15,638
decisions, all 7,819 evaluator rows, the exact public contexts and candidate
mappings, and all 28 pilot requests. It completed in 1.56 seconds without
model, tokenizer or encoder calls. Tokenization correctness and model-weight
contents remain inherited preparation witnesses; the runner verifies both
before inference.

- [Independent preparation audit](../output/dialogue-qwen-observation-v1/preparation-audit-01/receipt.json).
- [Actual runner metadata integration check](../output/dialogue-qwen-observation-v1/integration-01/receipt.json): all requests accepted in 2.46 seconds, with no model or label access.

- [Plan](../output/dialogue-qwen-observation-v1/preparation-02/plan.json):
  `2d5f7e6b512ae7260cc01685ae03220891ad4ca5236092645197d4074be90111`.
- [Preparation completion](../output/dialogue-qwen-observation-v1/preparation-02/completed.json):
  `6a7a8283efa612866a0a9f0c2bcce92bb54e9ce8982bde26baaa8d8aeb24eb9f`.
- [Lossless request download and restore instructions](../output/dialogue-qwen-observation-v1/publication-01/README.md).

The uncompressed requests exceed GitHub's single-file limit. The delivery gzip
round-trips to the exact frozen SHA256; compression changes no prompt or token.

[Forty-six focused synthetic checks](../output/dialogue-qwen-observation-v1/preflight-02/receipt.json)
passed for public-input isolation, causal history, mappings, stable probabilities,
batching, resource handling and fake-backend completion. The additional
[partial-cache regression](../output/dialogue-qwen-observation-v1/preflight-03/receipt.json)
passes. Initial lint findings remain preserved. These checks establish
implementation behavior, not task quality.

## Completed cost pilot

The sole pilot completed all **28 batches / 56 decisions** in **17.564 seconds**,
with **2,999,812,096 bytes** process-lifetime peak RSS. Its fixed token-rate
projection is **6,316 seconds**, below the **7,200-second** admission limit.
The separate request-count projection is **8,135 seconds** and remains
descriptive under the prospective rule. Admission is a heuristic, not a
full-run time guarantee; the full run retains its hard two-hour cap.

No labels were accessed, no quality-bearing outputs were retained, and no
accuracy claim follows. [Pilot receipt](../output/dialogue-qwen-observation-v1/pilot-01/completed.json),
SHA256 `3b8bcf6a4e1fb27322e6260e4cdc10b0514b9db72c98760e5a7d04284137f54d`.
The [independent pilot audit](../output/dialogue-qwen-observation-v1/pilot-audit-01/receipt.json)
agrees on all selected requests, work totals, payload hashes and the exact
6,316-second projection. Finite logits, timing and memory are authenticated
execution witnesses, with no replayed model calls.

The saved-output reporter is independently source-reviewed and passes
[30 focused synthetic checks](../output/dialogue-qwen-observation-v1/report-preflight-02/receipt.json).
It reconstructs decisions from candidate logits, applies the two separate
behavioral rules and proper-score checks, and retains every service and
historical seed. The first synthetic fixture/lint failure is preserved.
No real accuracy outputs have been read. The reporting allocation is fixed
before execution at 60 seconds, 2 GiB RSS and 64 MiB output.
