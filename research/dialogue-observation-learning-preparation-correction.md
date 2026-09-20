# Correct split-qualified dialogue identity before preparation

The frozen [first preparation](dialogue-observation-learning-preparation.md)
stopped before completion with `ValueError('Dialogue shared across splits')`.
It ran **7.816 phase seconds** with **584,794,112 bytes** peak RSS and zero neural
calls. The supervisor recorded **8.263 seconds**, exit 1 and process group 9322
absent. All first-version source, partial data and failure receipts remain
unchanged. No model was trained, no quality was evaluated and the failed
attempt is not resumed.

The new workload aggregator had incorrectly required a dialogue's bare ID to
be globally unique across TRAIN and DEV. The original dataset and existing
packet separate these namespaces. The correct record identity is
`(split, dialogue_id)`. Reusing a bare ID in different splits must not merge
records, relabel a split or exclude either dialogue. The existing input cohorts
remain unchanged. Source-data checks are recorded separately before execution.

The independent [namespace audit](../output/dialogue-observation-learning-v1/id-namespace-audit-01.json)
found **298 overlapping bare IDs**, all with different public dialogue content
across the two splits, including under the original normalization rule. IDs are
unique within each split. The exact failed record was DEV `10_00001`, the second
development record. The existing original parser already scopes IDs by split
and separately excludes normalized development-content matches to training.
The audit did not read targets, tokenize text, execute models or score quality.

## Exact correction

Add `dialogue_finetune_dataset_v2.py` and a separately versioned preparation
runner. Reexport the unchanged actor, target, lexical-parity and epoch-order
helpers. In workload aggregation, retain unique `(split, dialogue_id)` keys and
consistent global feature-ID/token mappings; remove only the erroneous
cross-split bare-ID rejection. Same-split duplicate identities still fail.
Every per-dialogue work formula, aggregate sum/maximum and training batch
selection is unchanged. V1 source and tests stay frozen as failure evidence.

The V2 runner binds all passed qualification sources and all first-preparation
sources to their prior hashes. It also pins the failed receipt and its source
manifest. It uses the same 300-second, 8-GiB RSS and 512-MiB output limits, full
cohorts, tokenizer, numeric transformation, target validation, planned twelve
fits, 20 epochs, effective batch 32 and seven practical scientific conditions.
Publish this correction, new source and tests before a new, exclusive
`preparation-02` run. Do not read or reuse partial actor/lexical outputs.

This is an explicit replacement preparation version after a metadata defect,
not a retry or continuation of the failed frozen program. It admits no model
execution. Complete workload profiling, worst-case cost qualification, a full
training allocation and a frozen reporter remain necessary before fitting.
