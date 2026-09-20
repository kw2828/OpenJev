# Stronger Qwen observation comparison: prompts frozen

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
preserves the failed attempt. No pilot or full inference has started.

The second preparation completed in **12.96 seconds**, with **882,016,256 bytes**
peak RSS and zero model calls. It froze **7,444 batches and 15,638 decisions**.
Current-exchange prompts contain 581-1,139 tokens; four-exchange prompts contain
612-1,292. All 7,819 rows and every candidate are retained. The cost-only pilot
has 28 batches selected before inference, including both arms of the longest
prompt groups. Independent packet checking precedes pilot execution.

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
