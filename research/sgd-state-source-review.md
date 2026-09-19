# SGD categorical-state source review

Status: full pinned train/dev data preparation, not a frozen model protocol or experiment. No training, inference, or test dialogue contents were accessed.

## Provenance and scope

Official repository: [google-research-datasets/dstc8-schema-guided-dialogue](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/tree/e852981ae34990f4358979625854259302feaa78), pinned to `e852981ae34990f4358979625854259302feaa78` (commit date 2023-08-07; inspected 2026-09-19). The dataset is [CC BY-SA 4.0](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/blob/e852981ae34990f4358979625854259302feaa78/LICENSE.txt). Cite Rastogi et al., [Towards Scalable Multi-Domain Conversational Agents](https://arxiv.org/html/1909.05855v2), AAAI 2020.

The first inspection downloaded two schemas and one training shard. The separately authorized full preparation then fetched all 127 train and 20 dev shards plus both schemas: 149 files, 504,286,598 bytes. Every file was checked against the pinned Git tree's blob SHA-1 and size, and recorded with SHA-256. No test files were downloaded or parsed. The independently recounted train totals match the paper's 16,142 dialogues and 329,964 turns.

An initial full-fetch attempt retained 142 successful files and seven TLS connection failures under ignored `runs/sgd-state-v1/source/all/`. An exclusive recovery at `source/all-recovery-01/` copied those 142 files only after Git-blob verification and downloaded the seven missing shards with four concurrent requests. The failed attempt remains intact. Fetching has no automatic retry loop.

Prepared files are local-only under ignored `runs/sgd-state-v1/data/`. The completion receipt binds all source files, 11 prepared payloads, and snapshots of the parser, preparer, and tests. It records 4.487 seconds of preparation, including source verification, not the earlier download time.

| Receipt | SHA-256 |
|---|---|
| Full source `source/all-recovery-01/fetch-receipt.json` | `b60346bb98e3b46419b9170fc1f2bdc776de1ecdd76e50a34977c3abb13c0cd3` |
| Prepared `data/completed.json` | `677d37ea581b3165b0adc96a0a0daab74271e0434df2ad23ace9568537e2cd12` |
| `data/stats.json` | `6c80810d444390d5447696e6a10a9ff5a5036d86e7b9a36ae92181f7ed320544` |
| `data/duplicates.json` | `dc8e971e8907d7721febe408425bc216bdec23685ad348b4a4d008f3cac0934f` |

## Schema and first-shard findings

Counts below are service-qualified slots, not deduplicated slot names.

| Schema | Services | All slots | Categorical slots | Declared candidate entries |
|---|---:|---:|---:|---:|
| Train | 26 | 215 | 53 | 201 |
| Dev | 17 | 136 | 36 | 134 |

Candidate counts range from 2 to 10. Train slot counts by candidate-set size are `2:22, 3:9, 4:9, 5:5, 6:1, 8:2, 9:3, 10:2`; dev counts are `2:15, 3:6, 4:7, 5:3, 6:1, 8:1, 9:1, 10:2`. No within-slot duplicate candidates occurred. Eight dev services are absent from the train schema. The pinned train schema has 215 slots, versus the original paper's historical count of 214.

Use `is_categorical`, not a nonempty `possible_values` list: non-categorical cuisine slots can contain illustrative values. Preserve literal strings. Dev `Media_2.subtitle_language` has the legitimate value `"None"`, distinct from a missing constraint. Boolean-looking values such as `"True"` are strings.

## Full train/dev preparation findings

Every categorical slot in every present USER service frame is scored, including absent slots. Query sequences count distinct dialogue × service × categorical slot combinations. All dialogues remain in the utterance packets, including those without categorical queries.

| Quantity | Train | Dev |
|---|---:|---:|
| Dialogues | 16,142 | 2,482 |
| Utterances | 329,964 | 48,726 |
| USER turns | 164,982 | 24,363 |
| Query sequences | 66,719 | 10,765 |
| Categorical frame-slot labels | 416,248 | 62,329 |
| NOT_MENTIONED labels | 263,632 | 38,662 |
| Ontology-value labels | 150,095 | 23,379 |
| DONTCARE labels | 2,521 | 288 |
| Labels for service names unseen in train schema | 0 | 33,093 |

| Evaluator-only transition bin | Train | Dev |
|---|---:|---:|
| Assignment from NOT_MENTIONED | 34,223 | 5,264 |
| Assigned value changed to another assigned value | 2,987 | 444 |
| Assigned value retained | 115,406 | 17,959 |
| NOT_MENTIONED retained | 263,440 | 38,662 |
| Assigned value cleared to NOT_MENTIONED | 192 | 0 |

`first_assignment` is the implementation label for assignment from NOT_MENTIONED; it includes reassignment after a clear. DONTCARE counts as assigned: its train/dev bin counts are 600/78 assignments, 26/1 revisions, and 1,895/209 retentions. These are gold transition counts, not evaluated predictor accuracy. Most labels retain state, so aggregate micro accuracy alone would obscure revision failures.

All categorical present-label lists were singleton and schema-valid; any empty, multiple, or unknown categorical label would have failed preparation with its source file/dialogue retained. Non-categorical equivalent surface forms are not categorical targets. Whole-dialogue duplicate identity uses ordered speaker roles and NFKC, case-folded, whitespace-normalized utterances. There were zero duplicates within train, within dev, or across train/dev; therefore zero dev exclusions. The rule removes cross-split matches only from dev and retains/reports within-split duplicates. It does not establish absence of paraphrases, shared templates, or semantically related dialogues.

The initial shard remains separately preserved and contains only Restaurants_1. Its unusually high 95.30% unchanged-label rate should not be substituted for the full-corpus audit.

## State semantics and leakage boundary

The [official format](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/blob/e852981ae34990f4358979625854259302feaa78/README.md#dialogue-representation) stores service-specific state snapshots on USER frames. Categorical assignments are singleton lists; non-categorical multiple strings can be equivalent surface forms. `dontcare` is a state value outside the ordinary categorical candidate list. The paper's baseline predicts updates and accumulates them, but that does not make the stored snapshots update-only annotations.

For a proposed dialogue × queried service/slot sequence:

- Inputs: fixed query schema and declared candidates, plus speaker-tagged utterances from dialogue start through the current USER utterance. Candidate API returns one of `NOT_MENTIONED`, `DONTCARE`, or an exact ontology value, optionally with probabilities over that fixed list.
- Targets: when the queried service has a current USER frame, absent slot means `NOT_MENTIONED`; `['dontcare']` means `DONTCARE`. An absent service frame is **unscored**, never a memory reset. Keep all intervening utterances in the model stream. Synthetic parser tests cover absent-service frames and clears; the full training corpus also contains 192 clear labels.
- Exclude frame service tags, gold actions/canonical values, slot spans, active intent, requested slots, state, service calls/results, dialogue-level `services`, and future utterances from model inputs. The initially inspected shard exposes actions on every turn and 321 service-call/result frames. Passing those annotations can collapse the intended inference problem. The next SYSTEM reply is future information.
- Supplying a query service/slot defines conditional state tracking, not end-to-end service routing. Never use evaluator frame availability as a model update mask. Do not feed previous gold state at evaluation. Keep all queries and turns of each dialogue in one split; do not select services or examples because they revise.

## Prepared packet contract and verification

`train-dialogues.jsonl` and `dev-dialogues.jsonl` contain only dialogue ID, ordered speaker/utterance pairs, and USER turn indices with the immediately preceding SYSTEM index if present. They are built once independently of questions. Full files necessarily contain later turns; the model consumer must use `public_prefix(dialogue, turn_index)` or an equivalently causal stream, never the whole future-containing packet as the current input.

Separate `*-labels.jsonl` files carry query ID, current turn, label ID/index, and evaluator-only transition/DONTCARE/unseen-service annotations. `catalog.json` contains schema-only service/slot descriptions and candidate texts. Candidate IDs reserve `reserved:NOT_MENTIONED` and `reserved:DONTCARE`; exact ontology values use `value:<literal>`, preserving the genuine `value:None`. The first two candidate indices are reserved labels, followed by ontology order. No service-frame availability mask is a permitted model input.

The three implementation files are `scripts/prepare_sgd_state.py`, `src/openjev/research/dialogue_state_data.py`, and `tests/test_dialogue_state_data.py`. Sixteen focused synthetic tests passed in 0.24 seconds; Ruff passed. Tests cover feature poisoning, future-prefix isolation, absent-service persistence, absent-slot clearing, ontology identity, malformed singleton rejection, and duplicate exclusion/failure preservation. An independent source-only peer review found no material feature/label or duplicate-boundary issue. These engineering checks do not validate a learned model or authorize a training protocol.

## Useful controls before an architecture claim

Require an unassigned/majority baseline; last explicit value with negation and dontcare handling; a current-user plus preceding-system classifier with an explicit persistent update ledger; a full-prefix linear lexical classifier; and a matched bounded-history control. A learned recurrent memory must beat the update ledger, not just a classifier denied past state.

Report macro slot/value performance, active-value accuracy, revision accuracy, persistence by age, and dialogue-level joint categorical accuracy alongside micro accuracy. Separate seen and unseen dev services. Joint categorical accuracy is not the official full-state metric, which also includes non-categorical slots.

SGD offers genuine language-mediated corrections and persistence, but these observations do not establish that a new recurrent architecture is needed. Ontology inference, short-context semantics, and a cheap explicit state table may explain most performance. The full audit supports a categorical development pilot with explicit revision and retention reporting; it does not itself qualify a memory mechanism. Keep test dialogues untouched. No model protocol, seed allocation, model selection, or training/inference experiment was performed by this preparation task.
