# Why the conditional scorer misses updates

**All nine models chose the previous NOT_MENTIONED value on every one of the 31
unseen TRUE updates.** This saved-output diagnosis narrows the failure pattern;
it does not identify its cause or change the [failed conditional-attention
study](dialogue-conditional-results.md).

The analysis was specified after seeing the original results. Its source and
[protocol](dialogue-conditional-error-protocol.md) were published before this
single descriptive execution. There was no training, model inference, relabeling,
probability repair or official-test access.

## Which wrong answer was chosen?

The TRUE group contains 31 unique rows from 30 dialogues and three schema
queries. Across nine fits there are 279 prediction events, not 279 independent
examples. Every event selects the previous value, NOT_MENTIONED. Mean probability
assigned to that wrong previous value ranges from **95.10% to 99.68%** across
fits. The models receive the correct previous value explicitly; they have no
recurrent carry mechanism. Their outputs agree with the supplied prior, but
this does not distinguish use of that input from a default NOT_MENTIONED bias.
It is not evidence of recurrent-memory inertia.

All 23 unseen DONTCARE changes are also changes from NOT_MENTIONED.
DONTCARE means no preference, which is distinct from not mentioning a property
and from a Boolean false value. Counts below partition all 23 rows for each fit.

| Model | Seed | Correct | Wrong previous value | Other wrong value |
|---|---:|---:|---:|---:|
| Mean pooling | 5301 | 0 | 18 | 5 |
| Mean pooling | 5302 | 2 | 14 | 7 |
| Mean pooling | 5303 | 0 | 17 | 6 |
| Slot attention | 5301 | 2 | 21 | 0 |
| Slot attention | 5302 | 2 | 16 | 5 |
| Slot attention | 5303 | 2 | 15 | 6 |
| Candidate attention | 5301 | 0 | 17 | 6 |
| Candidate attention | 5302 | 0 | 20 | 3 |
| Candidate attention | 5303 | 0 | 22 | 1 |

There are **no FALSE changes or clears in the admitted development data**.
These results cannot establish transfer across both Boolean polarities.

## What explains the small favorable average loss?

The original candidate-minus-slot mean NLL difference is -0.016574 nats over
1,704 unseen changed rows. The following additive decomposition uses the full
1,704-row denominator for every contribution, then averages the three seeds.
Negative contributions favor candidate attention.

| Target value | Unique rows | Contribution to overall NLL difference |
|---|---:|---:|
| TRUE | 31 | -0.040856 |
| DONTCARE | 23 | -0.012586 |
| Other concrete values | 1,650 | +0.036868 |
| FALSE | 0 | 0.000000 |
| NOT_MENTIONED | 0 | 0.000000 |
| **Total** | **1,704** | **-0.016574** |

Candidate attention has lower loss on the TRUE and DONTCARE groups, yet selects
none of their correct answers at any seed. The larger ordinary-value group
contributes an average regression. Probability changes can therefore improve
the aggregate proper score without fixing these decisions. This is not a
calibration result. Empty groups have zero additive contribution and undefined
conditional means; they are not successful predictions.

Each reconstructed seed difference agrees with the original analysis within
1e-12. The original requirement of improvement at all three seeds still fails.

## Training support and next hypothesis

Among 3,347 admitted training changes, there are **213 TRUE**, **54 FALSE**,
**50 DONTCARE**, **25 clears**, and **3,005 other concrete-value changes**.
The 54 FALSE changes cover only one schema query. The existing loss balances
unmentioned retention, assigned retention and changes; it does not balance
these individual value classes.

Two explanations deserve a controlled test: insufficient transfer of property
meaning, and insufficient training support for rare outcomes. The current
[candidate construction](../src/openjev/research/dialogue_state_data.py) appends
bare values such as `True` or generic `DONTCARE (no preference)` to the schema
description. Low target probability alone does not establish that the frozen
features lack the needed information.

A separate development experiment should compare explicit typed interpretation
against a matched flat scorer, crossed with original versus training-only
rare-category weighting. Both heads need the same context, schema information,
candidate inputs and computation accounting. Preserve ordinary-value accuracy
and retained-state behavior, and expose the lack of FALSE schema diversity.
Improving these already inspected development cases is not fresh confirmation.
A new service-held-out split inside official training can support development,
but those rows participated in earlier fits and are not historically untouched.
Official test remains untouched. This proposed comparison tests output
factorization and training support within the existing representation, not new
language understanding or a new recurrent architecture.

The machine-readable diagnostic describes previous-value errors as "output
reliance." Interpret that as an output match only. A causal claim about the
previous-value input would require a separate intervention.

Typed handling is established in [TripPy](https://aclanthology.org/2020.sigdial-1.4/),
while [D3ST](https://arxiv.org/abs/2201.08904) motivates description-based schema
grounding. Neither is a new contribution by itself. In particular, TripPy's
`none` operation retains state and must not be confused with this experiment's
resulting NOT_MENTIONED value. Any recurrent follow-up must show a remaining
temporal problem after these interpretation controls.

## Evidence

- [Complete diagnosis](../output/dialogue-conditional-error-v1/diagnostic-01/summary.json):
  all 96 groups for all nine fits, training support and every seed's decomposition.
- [Execution receipt](../output/dialogue-conditional-error-v1/diagnostic-01/receipt.json):
  11.32 seconds, zero model and training calls; original result unchanged.
- [Pre-analysis publication verification](../output/dialogue-conditional-error-v1/publication-preanalysis-01/receipt.json):
  exact source, tests and protocol at commit `f4942dbe2af8dc19949d5455304f72361645ba02`.
- [Synthetic preflight](../output/dialogue-conditional-error-v1/preflight-01/summary.json):
  seven checks passed before execution.
- [Independent saved-output audit](../output/dialogue-conditional-error-v1/audit-01/summary.json)
  and [receipt](../output/dialogue-conditional-error-v1/audit-01/receipt.json):
  all 864 fit/group cells, 50 training-support cells and every primary
  contribution reconstructed without discrepancies in 6.10 seconds.
  The audit used only saved artifacts, with no model or reporter imports.

The original run completion hash is
`df3c172bae7163292b54cdd9a3b1d6e3bb5a07acb49b62be4f0c0dfc68efbe75`.
The new summary hash is
`c696c3f60d394f026d3cedea262df782b382198151fd40e254c12626398a1689`.
All original labels and decisions remain in the analysis. Manual wording
judgments are not used to exclude examples or alter denominators.
