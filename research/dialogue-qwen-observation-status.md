# Stronger Qwen observation comparison: completed, both comparisons fail

The [prospective protocol](dialogue-qwen-observation-protocol.md) fixes two
arms on all 7,819 existing development rows: current exchange and four public
exchanges, with identical correct previous values and public lexical flags.
All **7,444 batches / 15,638 decisions** completed in **3,229.40 seconds**.
Current-exchange Qwen gains changed-value accuracy over the historical small
model but introduces severe retention errors; four-exchange history worsens
the aggregate result. Both behavioral comparisons and both proper-score
nonregression checks fail. [Completed results and chart](dialogue-qwen-observation-results.md).

The first model-free preparation attempt stopped because an unrestricted
local cache lookup demanded absent `README.md` and `.gitattributes` files.
It made no model calls. The required weights and tokenizer assets were already
present. [Failure receipt](../output/dialogue-qwen-observation-v1/preparation-01/failed.json).

The preparation lookup now uses the same file filter as the production scorer,
while still checking all eleven model-file hashes. No model, task wording,
cohort, candidate, or numerical-run rule changed. A separate metadata preparation
preserves the failed attempt. Full inference subsequently completed once.

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
No real accuracy outputs were read while preparing the reporter. Its allocation was fixed
before execution at 60 seconds, 2 GiB RSS and 64 MiB output.

## Full-run launch

The sole full run started on **2026-09-20 at 12:45:45 UTC**, session **14410**,
process group **83366**, after the other task released the numerical allocation.
It scores all **7,444 batches / 15,638 decisions**, retaining the prospective
**7,200-second** cap (approximately **14:45:45 UTC** deadline), 12 GiB RSS
and 512 MiB output limits. No fitting, generated text or paid calls are involved.
No partial quality was inspected. The original rule prohibited resuming a
failure or using it for complete-cohort conclusions.

The [independent result checker](../scripts/audit_dialogue_qwen_result.py) and
[figure renderer](../scripts/plot_dialogue_qwen_observation.py) are source-reviewed
before results are read. The checker reconstructs candidate choices and
distributions, 42 primary metric cells and the 16 decision components. It
inherits inference/provenance witnesses and does not independently recheck
rare-category, equal-dialogue or paired-repair metrics. Its saved-artifact
allocation is 60 seconds, 2 GiB RSS and 64 MiB output. The figure allocation is
60 seconds, 1 GiB RSS and 32 MiB output. Both subsequently executed against the
completed run, as recorded below.

The unchanged independent checker now passes
[12 synthetic regression checks](../output/dialogue-qwen-observation-v1/result-audit-preflight-01/receipt.json)
in **0.46 seconds**, with one CPU thread and no model, tokenizer, checkpoint or
live-score access. These cover a complete artificial report, corrupted metrics,
decisions and probabilities, missing/reordered requests, authentication before
score decoding, prompt-order ties, underflow-safe log loss, service weighting
and preservation of failed attempts. The test fixture replaces only local
paths, historical pins and cohort counts; this is an engineering check, not an
audit of the live experiment or a quality result. All frozen study, reporter
and auditor sources remained unchanged during those checks.

## Completed execution, reporting and publication

Session **14410 exited 0**. PID and process group **83366** were absent on the
terminal cleanup check; the numerical allocation was released to the waiting
task. The run completed all 7,444 calls with **zero generated tokens and zero
optimizer updates**, using **3,712,122,880 bytes** process-lifetime peak RSS.
Its 3,229.40 seconds include authentication, model load, inference, serialization
and closing hashes. Preparation and the repeated 56-decision cost pilot remain
separate charged work, totaling **3,259.92 seconds** across the three phases.

- [Execution receipt](../output/dialogue-qwen-observation-v1/run-01/completed.json), SHA256
  `872ee6819af4cbc4f7e0bd6397d6c90907dba9aaecc995abb6045fd20520f70c`.
- [Main report](../output/dialogue-qwen-observation-v1/report-01/summary.json): completed in
  **2.893 seconds**; SHA256 `931f60349dc7dace7508d0f3307a6c480c4e14022de7e2c047b6f024ec349c6c`.
- [Independent audit](../output/dialogue-qwen-observation-v1/result-audit-01/receipt.json):
  completed in **1.416 seconds**, agreeing on **530 scalar checks**, 42 primary
  metric cells and all 16 decision components. Its scope remains the primary
  cells and decisions, not an independent replay of model inference or every
  descriptive subgroup. SHA256 `8dc7996b9d70a2c330c040a62537e8db8ff4a79d6125c311f28978fd9a8c5848`.
- [Corrected chart](../output/dialogue-qwen-observation-v1/figure-02/comparison.png):
  both Qwen arms and every historical seed. The first render's overlapping
  header is [preserved](../output/dialogue-qwen-observation-v1/figure-01-visual-qa.json).
  The second changes only text placement, uses identical plotted values and
  passed visual inspection. No model or quality calculation was repeated.

The model's better changed-value recognition does not compensate for its
retention failures under the frozen rule. These results do not promote the
recurrent adapter to a model-quality experiment. First diagnose how the fixed
decision interface interprets retained and special values using saved outputs.
