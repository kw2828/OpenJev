# Dialogue copying: posthoc complementarity diagnostic

This analysis uses already exposed official development data and the same 21 completed fits. It changes no prediction, model, training recipe, or continuation decision. No neural, encoder, RNG, or official-test calls were made. Public text and individual predictions remain local; this folder contains code, aggregate results, and hashes.

The literal rule was reconstructed independently from every public USER prefix, including turns without a scored service frame. All **62,329** reconstructed choices exactly match the saved reference. All seven families and three seeds remain in [diagnostic.json](diagnostic.json); the detailed comparisons below use all three innovation-Kalman fits equally.

## Complementarity, not achieved hybrid performance

| Three-stratum macro accuracy | Seen services | Unseen services |
|---|---:|---:|
| Literal rule | 54.75% | 65.76% |
| Innovation Kalman | 71.65% | 53.98% |
| Target-aware union oracle | 85.48% | 81.50% |

The oracle counts a query correct when either the literal rule or one fixed neural seed is correct, then averages the three seeds. It uses the true target to choose, never selects across seeds, and is **not an implemented controller or attainable-score claim**. Unseen per-seed oracle macros are 81.15%, 82.67%, and 80.67%. The mean leaves 15.73 percentage points of diagnostic room above literal's 65.76%, without showing that public inputs can identify the better prediction.

Across 33,093 unseen queries, the mean counts per seed are **8,152.33 literal-only correct**, **3,236.33 neural-only correct**, and **4,051.67 both wrong**. Fractional counts arise solely from averaging three integer-count fits on the same queries. A selector cannot repair the both-wrong cases. On the 203 unseen revisions, literal reaches 79.80%, neural 39.57%, and the target-aware union 88.67%; only 18 revisions per seed on average are neural-only correct.

## Where the errors differ

“Current mention” means any Unicode-bounded, case-folded ontology string in the current USER utterance. It does not assert that the mention applies to the queried slot. Boolean slots are exactly those whose declared values are True/False; numeric 0/1 slots are not included. Label categories in this table use gold annotations **only for retrospective analysis**.

| Unseen subgroup | Queries per fit | Literal accuracy | Neural accuracy |
|---|---:|---:|---:|
| NOT_MENTIONED, all slots | 22,908 | 87.83% | 71.68% |
| DONTCARE | 179 | 0.00% | 3.72% |
| Assigned ontology value, all slots | 10,006 | 56.81% | 44.58% |
| Boolean slot, assigned ontology value | 524 | 0.00% | 1.46% |
| Nonboolean, ontology value, current mention | 1,541 | 86.05% | 44.32% |
| Nonboolean, NOT_MENTIONED, current mention | 740 | 1.22% | 49.77% |
| Nonboolean, ontology value, no current mention | 7,941 | 54.88% | 47.48% |

The 740 current-mention/NOT_MENTIONED cases expose a weakness of blind copying: a matching string is often not a constraint for this query. Neural predictions recover some of these cases, but still miss about half. Conversely, copying is far stronger when a current mention accompanies an assigned ontology target. The table does not establish which linguistic phenomenon caused each error, such as negation, wrong-slot reference, or incidental text.

The apparent unseen Boolean success is mostly absence prediction: **8,409 of 8,933** Boolean queries are NOT_MENTIONED. Neural assigned-Boolean accuracy falls from **74.03% on 1,938 seen queries to 1.46% on 524 unseen queries**. Routing Boolean slots to the current neural model is therefore unsupported. Neither arm supplies a useful DONTCARE solution. Most unseen neural-only wins also occur without a current exact mention: 2,783 per seed on average, versus 453.33 with one.

## Concrete implication

A future learned update detector could test whether a supplied query's literal mention should write, be rejected, or leave an explicit value ledger unchanged. Preserve literal copying as a control and separately test Boolean paraphrases, negation, and DONTCARE. The present evidence supports testing that mechanism; it does not validate a posthoc routing rule. Training must use training labels, with this development set declared exposed and a new prospective confirmation boundary. Do not select examples, seeds, or decision thresholds from these slices or reopen the failed 21-fit recipe.

The retained JSON reports every primary seed, all transition categories, seen/unseen panels, current-mention and Boolean partitions, and their predefined joint slices. Subgroup percentages are micro accuracies within each slice; the first table retains the original three-stratum macro. No significance or causal explanation is claimed.

Input and source hashes are bound in [diagnostic.json](diagnostic.json), SHA-256 `ee5152ec7edc04a187c33e2ffdb79cec9596743ebda6ec2dbf9e89122dfd6487`. The calculation source is [analyze.py](analyze.py). No additional peer-review clearance is claimed for this posthoc script; all literal choices and frozen input digests were verified during its successful saved-only execution.
