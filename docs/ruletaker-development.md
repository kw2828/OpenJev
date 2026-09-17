# Multi-step text reasoning: prepared data

**Status: data prepared and audited; no model has trained or been evaluated on this packet.** This is the next diagnostic after the small [corrective-recurrence gain](corrective-associative-study.md) failed the routing continuation rule.

[RuleTaker](https://github.com/allenai/ruletaker) supplies English facts, rules and true/false questions. Its [original paper](https://arxiv.org/abs/2002.05867) already reported strong transformer results and deeper-chain generalization. This is an established reasoning benchmark, not a new dataset or an automatic novelty claim.

## Prepared split

| Part | Upstream source | Selected worlds | Questions | Allowed annotated question depth |
| --- | --- | ---: | ---: | --- |
| Training | depth-2/train | 2,000 | 19,809 | 0-2 |
| Same-depth development | depth-2/dev | 300 | 2,947 | 0-2 |
| Later-depth development | depth-5/dev | 300 | 1,941 | 3-5 |

Worlds, not individual questions, are the grouping unit. A world fingerprint normalizes case, whitespace and sentence order; duplicates across parts are excluded before deterministic world sampling. At most 12 verified questions are retained per world. The later-depth part is nearly balanced: 974 true and 967 false labels. Labels and proof-depth annotations come from the release, rather than a newly run theorem prover.

The model-input helper exposes only the English context, English question and stable `true`/`false` candidate IDs/descriptions. Proofs, logical representations, question-generation strategy, gold label and annotated depth stay outside model inputs. Original English questions are intersected with the release's Problog verification labels. Unmatched examples are counted, not silently relabeled.

Two audit findings matter for reproduction:

1. Problog question suffixes can change after filtering. Join by world ID and normalized English assertion, not question ordinal. An ordinal join incorrectly associated different questions.
2. A depth-n directory is not a sufficient query-depth filter. The unfiltered depth-2 sources contain some questions annotated above depth two, so the preparer explicitly enforces the table's depth sets.

No test archive members are opened. The original CLINC confirmation and calibration sets also remain unscored. Changing query depth also changes the distribution of source worlds, so later-depth results must be described as a depth-and-context shift, not an isolated causal effect of one extra reasoning step.

[Source and selection audit](../evidence/ruletaker-preparation-v1/manifest.json) · [Executable preparer](../scripts/prepare_ruletaker.py)

## Reproduce

```sh
.venv/bin/python scripts/prepare_ruletaker.py \
  --archive runs/ruletaker-source/rule-reasoning-dataset-V2020.2.5.zip \
  --out runs/ruletaker-depth-v1/packet
```

The preparer downloads the exact authors' release if absent and requires SHA-256 `080c8bca836603d9fea040e5a242bbd281f631255ca83428b7141ebf7f3deb0c`. A mismatched or partial archive is rejected. It streams selected JSONL members without extracting archive paths. Keep raw data under ignored `runs/`; the public audit exports counts and hashes. The upstream repository is Apache-2.0; the archive has no separate license file, and this repository does not redistribute it.

## Next experiment requirements

Freeze the model and training comparison before fitting. Include question-only and single-pass controls, a tied multi-step memory head, and a corrective variant. Keep the encoder, labeled examples and parameter budgets matched where possible, and report additional compute rather than treating equal parameter counts as equal cost. Evaluate both development parts separately, including accuracy by annotated question depth and paired uncertainty grouped by world. Neither proof metadata nor a symbolic oracle's hidden representation may be used as a model input.

Only a candidate that improves over the strongest appropriate control at a stated cost should proceed to a separately frozen test. Do not report the prepared packet, passing loader tests or literature results as model efficacy.
