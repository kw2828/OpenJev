# Stronger Qwen observation comparison: preparation

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
will preserve the failed attempt. No pilot or full inference has started.

[Forty-six focused synthetic checks](../output/dialogue-qwen-observation-v1/preflight-02/receipt.json)
passed for public-input isolation, causal history, mappings, stable probabilities,
batching, resource handling and fake-backend completion. The additional
[partial-cache regression](../output/dialogue-qwen-observation-v1/preflight-03/receipt.json)
passes. Initial lint findings remain preserved. These checks establish
implementation behavior, not task quality.
