# Dialogue-memory preparation and provenance

Preparation completed before training. The 21-fit study has now launched under its frozen plan; final results are pending. This is an official-development-set comparison for supplied categorical service/slot queries, not full dialogue-state tracking, a test-set result, or an architecture claim. No official test dialogue contents were fetched or parsed. Raw dialogues, embeddings, and individual label packets remain local under ignored `runs/sgd-state-v1/`.

## Attribution and scope

The Schema-Guided Dialogue dataset is from Google Research, pinned to [commit `e852981ae34990f4358979625854259302feaa78`](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/tree/e852981ae34990f4358979625854259302feaa78). Credit Rastogi et al., [Towards Scalable Multi-Domain Conversational Agents: The Schema-Guided Dialogue Dataset](https://arxiv.org/abs/1909.05855). Its [license is CC BY-SA 4.0](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/blob/e852981ae34990f4358979625854259302feaa78/LICENSE.txt). Preparation reorganizes public text and categorical annotations; it does not change upstream source files.

Text features use Sentence Transformers' [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41), revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, whose [pinned model card declares Apache 2.0](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/README.md). This pretrained Transformer is frozen, with attention-mask mean pooling and unit-normalized 384-dimensional outputs. The experiment compares memory components downstream of that encoder; it is not a Transformer-free system. Upstream pretraining-corpus overlap was not audited.

## Data and selected cohort

All 127 train shards, 20 dev shards, and two schemas were fetched: 149 files totaling 504,286,598 bytes. Each raw file was verified against the pinned Git blob SHA-1 and size and recorded with SHA-256. The source schemas have 53 train and 36 dev categorical service/slot definitions, with 2-10 ontology values each. Two additional reserved states are NOT_MENTIONED and DONTCARE; the legitimate ontology string `None` remains distinct.

| Quantity | Train | Dev |
|---|---:|---:|
| Full dialogues | 16,142 | 2,482 |
| Full utterances | 329,964 | 48,726 |
| Full categorical labels | 416,248 | 62,329 |
| Dialogues selected before categorical filtering | 2,048 | 2,482 |
| Selected dialogues with categorical labels | 2,017 | 2,363 |
| Selected USER-step feature blocks | 20,600 | 23,810 |
| Selected scored queries | 51,741 | 62,329 |

Train selection uses the 2,048 smallest SHA-256 values of `openjev-sgd-v1:` plus dialogue ID, before inspecting label eligibility. Thirty-one selected train dialogues and 119 dev dialogues have no categorical labels and contribute no model loss. No service or revision-based selection is used. Whole-dialogue duplicate checks preserve speaker order and normalize text with NFKC, case-folding, and whitespace collapse. No within-split or cross-split duplicates were found, so no dev dialogues were excluded. This does not rule out shared templates or paraphrases.

Dev contains 29,236 seen-service and 33,093 unseen-service queries. Selected train has 396 assigned-value revisions and 25 clears; dev has 444 revisions and no clears. The median stream is only 10 USER steps, with maxima of 23 in train and 29 in dev. Maximum scored queries per dialogue are 80 and 87; candidate sets contain at most 12 entries. These streams do not establish long-context capability. Full transition counts and definitions are in the [source review](sgd-state-source-review.md).

## Actor inputs and encoding

Each step embeds the current USER utterance with its immediately preceding SYSTEM utterance. Independent structural reconstruction found that these blocks cover every past utterance in the selected, alternating dialogue streams. Query and candidate text comes only from service/slot descriptions and declared schema values. Supplied query identity defines the conditional task; it is not inferred service routing.

Labels, gold frame availability, transition bins, unseen-service flags, dialogue-level service lists, actions, spans, intents, and service-call results do not enter actor text features or memory updates. Labels and bins are used for supervised loss/evaluation. Memory is read after the indexed current step; the carry control replays only that query's prefix. Complete stored dialogues contain future turns, so this causal consumer boundary matters.

The successful encoder produced 44,763 unique feature vectors, processing 1,337,813 tokens including special tokens. No input exceeded the 254-content-token chunk limit and no tokens were truncated. Encoding used float32 on MPS, batch size 128. Its plan and completion bind the six actual model/tokenizer files, prepared input hashes, and an exact executed-source snapshot. A separate read-only review reconstructed all query catalogs and selected packets exactly, without encoder or model calls. This checks assembly and identities, not a numerical replay of pretrained inference.

## Preserved failures and recoveries

The initial full fetch retained 142 verified files and seven TLS failures: train shards 107-111, 113, and 114. A separate recovery copied those 142 files after verification and downloaded only the missing seven, using four concurrent requests. Recovery took 7.459 seconds; this is not the total original download cost. The failed fetch receipt has no complete wall-time measurement. Data preparation then took 4.487 seconds.

`features-01` failed after encoding because final cache enumeration demanded 24 unrelated, absent repository files. Its failed receipt and partial outputs remain preserved and were not promoted to a successful packet. The fix verifies the six used model/tokenizer files before inference. A fresh `features-02` completed successfully. Recorded encoder-attempt times are 19.574942 seconds failed plus 15.256808 seconds successful, totaling 34.831750 seconds; these are preprocessing costs, not memory-training timings.

## Receipt identities

The following links point to exact public receipt copies at the verified SHA-256 identities. The collection excludes raw dialogue text and feature arrays. The source/data receipts bind their local payloads, including three preparation implementation snapshots. These are artifact-integrity records, not external process attestations.

| Receipt | SHA-256 |
|---|---|
| [Initial three-file source fetch](../output/dialogue-memory-v1/provenance/source/fetch-receipt.json) | `7f0812bb450ebd337dfd77b18081570d962e2099f4a6277b3a96acda290f8503` |
| [Failed full fetch](../output/dialogue-memory-v1/provenance/source/all/failed.json) | `e05391a8135346113cab3ca4986bed30ceb2c6dd5a30b88f78531dec22c5f432` |
| [Recovered full fetch](../output/dialogue-memory-v1/provenance/source/all-recovery-01/fetch-receipt.json) | `b60346bb98e3b46419b9170fc1f2bdc776de1ecdd76e50a34977c3abb13c0cd3` |
| [Prepared data](../output/dialogue-memory-v1/provenance/data/completed.json) | `677d37ea581b3165b0adc96a0a0daab74271e0434df2ad23ace9568537e2cd12` |
| [Aggregate data counts](../output/dialogue-memory-v1/provenance/data/stats.json) | `6c80810d444390d5447696e6a10a9ff5a5036d86e7b9a36ae92181f7ed320544` |
| [Duplicate audit](../output/dialogue-memory-v1/provenance/data/duplicates.json) | `dc8e971e8907d7721febe408425bc216bdec23685ad348b4a4d008f3cac0934f` |
| [First encoder plan](../output/dialogue-memory-v1/provenance/features-01/encoder-plan.json) | `78a013a6d84e7220b0f97e156f73713478cab6703f8d75e97df38221ff7d31b1` |
| [First encoder failure](../output/dialogue-memory-v1/provenance/features-01/failed.json) | `36bdb5eb6e9c11a6de13f9749c426666c3b5135d95a9a05956225ebfb006ee87` |
| [Successful encoder plan](../output/dialogue-memory-v1/provenance/features-02/encoder-plan.json) | `6d9bd36d1233db0ec24b92b2ec14079d565842b45b385d4d739f23864d3b8307` |
| [Successful encoder completion](../output/dialogue-memory-v1/provenance/features-02/completed.json) | `e4503c3dd63d28b74e91877dfc9c69b81f4b880e4f07d240cf8c08331fa0b308` |

The [preparation manifest](../output/dialogue-memory-v1/preparation-manifest.json), SHA-256 `3a5b1fece20fff0ae65ad30f3db1fe1bfbe6d8ec1e0ce20b73660dd00af09200`, binds all 15 public metadata/source-snapshot files. The [frozen training plan](../output/dialogue-memory-v1/provenance/study-01/plan.json) is `8c62bd5853299f29fc232b044b69f24d1d0bf4fe9e1a860954a1515d1a617a38`. Its preparation manifest reports 142 passing synthetic preflight tests and Ruff; these are engineering checks, not study outcomes.
