# The apparent delayed references mostly expose number-word mismatch

The complete audit covers **51,741 labeled questions in 2,017 training
dialogues**. It finds **38 delayed SYSTEM-only literal proxies**, but inspecting
every public prefix does not establish a delayed proposal-acceptance problem.
All 38 targets are numerical quantities. Exact digit matching misses numbers
written as words, while finding unrelated older digits in search-result counts,
addresses, times and other slots.

This closes the proposed source-separated proposal-memory experiment **on
this cohort**. It does not close research on recurrence or demonstrate that
all dialogue decisions are history-free. The next practical control should
improve observation learning while maintaining an autonomous scalar state.

## Complete support counts

The [prospective protocol](dialogue-history-support-protocol.md) uses the
original complete TRAIN cohort, including first annotations and unscored
turns. No previous-gold input, model prediction, feature array or fitted
model is needed for this evaluator-side diagnostic.

| Literal-history subset | Rows | Dialogues | Services |
|---|---:|---:|---:|
| Changed OTHER target, distant SYSTEM match and no USER literal match | 38 | 37 | 12 |
| Changed OTHER target, distant latest literal evidence from either speaker | 47 | 45 | 12 |
| Retained OTHER target, distant latest literal evidence | 1,115 | 392 | 16 |
| Changed OTHER target, no literal evidence anywhere in its prefix | 933 | 759 | 18 |

These subsets overlap. OTHER excludes reserved NOT_MENTIONED/DONTCARE and
Boolean targets. Distant means at least four USER exchanges old; the current
and previous three exchanges form the recent window. Absence of a literal
match is not absence of semantic evidence. Remote evidence for unchanged
values primarily concerns ordinary carry, which the scalar control supplies.

There are **4,271 first assignments, 396 revisions, 25 clears, 32,718
unmentioned retentions and 14,331 assigned retentions**. Of all endpoints,
**42,810** follow an immediately preceding annotated USER endpoint,
**8,363** are first annotations, and **568** have gaps. Gaps remain unscored
between endpoints; they are not filled with gold carry or resets.

The complete public streams provide 20,600 USER exchanges and 91,856 possible
query-exchange positions for 8,363 supplied dialogue-query combinations.
There are no consecutive same-role pairs in this selected cohort and no
normalized candidate-literal collisions. Query inventory is supplied as task
input, not inferred service routing.

## What the 38 cases mean

The [follow-up review plan](dialogue-history-support-review-plan.md) includes
all 38 cases, without sampling or selecting examples. It is explicitly
label-aware source inspection, not blinded evaluation or model accuracy.
The [case-by-case review](dialogue-history-support-case-review.md) retains
the IDs and decisive turn references without publishing raw dialogue dumps.

Both source readers find the target number written as words within the recent
window in **34/38 cases**. Of these, **32** are direct current requests, one
has conflicting car/bus wording, and one requires plausible transfer of a
recent four-person group into a flight request. This is not a claim that a
model correctly answers all 34.

The remaining **four** contain older USER-specified event ticket quantities
and SYSTEM confirmations, followed by a restaurant booking without an explicit
party size. Carrying that group size across services is plausible but not
explicitly requested. Neither reader identifies a clean unresolved SYSTEM-only
proposal awaiting delayed acceptance. Cross-service transfer is a different
question from the proposed source-separated write rule, and these ambiguous
examples cannot establish that rule's advantage. Official targets and all
earlier failed comparisons remain unchanged.

## Verification and cost

The frozen implementation and plan were published in commit
[`ef7aeb5`](https://github.com/kw2828/OpenJev/commit/ef7aeb5) before the single
pass. **45 synthetic tests** and lint passed; separate source review found no
material issue. The actual process exited **0** after **2.566 seconds**, with
**238,108,672 bytes** peak process RSS and **zero model calls**. It completed
under the fixed 180-second, 2-GiB and 128-MiB caps without retry.

The input packet also contains already exposed DEV records, which were not
audited or scored. No DEV dialogue file or official TEST content was opened.
This is development evidence, not an untouched benchmark result. The raw
review prefixes remain local; the compressed evaluator metadata and all
aggregate counts are published.

The [separately implemented saved-metadata checker](../output/dialogue-history-support-v1/independent-check-01/receipt.json)
agrees on **all 7,955 summary count fields**, including every bin, service,
type, age, adjacency and proxy-support table. It also reconstructs endpoint
metadata relationships, for 1,094,528 total equality checks. It completed once
in **1.405 seconds** with zero model calls. Literal detection and the four
raw-turn structural totals inherit the authenticated producer; this checker
does not independently read dialogue text or establish semantic correctness.

The 38 primary rows comprise **32 first assignments and six revisions**:
31 have adjacent annotated predecessors, five are first annotations and two
have gaps. Thus gaps alone do not explain the proxy issue. Their latest SYSTEM
matches are four to nine exchanges old, but source inspection shows why literal
distance is misleading here.

Derived dataset metadata retain the source's **CC BY-SA 4.0** license and
attribution to [Google's Schema-Guided Dialogue dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/tree/e852981ae34990f4358979625854259302feaa78).
The repository's code license does not replace that data license.

| Artifact | SHA-256 |
|---|---|
| [Frozen plan](../output/dialogue-history-support-v1/plan-01/plan.json) | `e5be6b1cde77b5362a4235a706a1029bc073f1421c7369e721350bbf7adf8d06` |
| [Completion](../output/dialogue-history-support-v1/audit-01/completed.json) | `86d66aa5ccb422eb0cb866e248ba45ea6b4ce4064094e694174bba3403dea67e` |
| [Complete aggregate summary](../output/dialogue-history-support-v1/audit-01/summary.json) | `eeb9cbc5c5ffd81e3501ed036c58c3acb69c15d65782e052fd8db38c5f507c60` |
| [Compressed endpoint metadata](../output/dialogue-history-support-v1/audit-01/endpoint-metadata.jsonl.gz) | `6f6858f81cd5a14e173c84d4ef9cd19072065c55073ae0f48a81127c9a7e8108` |
| [Independent aggregate check](../output/dialogue-history-support-v1/independent-check-01/receipt.json) | `54b6e81d8bbe9989ee45b8f5dd3bf1582b3018422b1babfdcfbc362fee62383e` |

## Next experiment

Prioritize a fine-tuned observation encoder with the normalized autonomous
scalar memory, compared with the identical frozen-encoder scalar. Train from
each model's own state across complete streams; labels supervise losses only.
Include a transparent number-normalization control so gains from recognizing
number words cannot be credited to a recurrent mechanism. Recompute trainable
encoder features as weights change and count that computation.

This would test representation learning, not novel architecture. Only after
that control leaves a reproducible history-dependent gap should a new recurrent
operator compete against scalar state, an explicit operation ledger and
unrestricted memory with comparable capacity. No new fit is admitted or
claimed here, and all earlier failed results remain unchanged.

The [implementation design](dialogue-observation-finetune-design.md) identifies
the existing encoder/memory components, gradient-path changes and bounded
qualification needed before that comparison.
