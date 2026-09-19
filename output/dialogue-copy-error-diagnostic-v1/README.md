# Dialogue copy error diagnosis

Posthoc saved-output analysis of exposed development data. All 15 fits and three seeds are retained; no model or encoder calls. The frozen continuation rule remains failed (7/13).

Counts below are mean per fit across the same three seeds, except the single literal control. Exact per-seed counts and all subgroups are in diagnostic.json.

## Revisions

A stale error selects the previous scored gold label; other wrong predictions select a different wrong candidate.

| Panel | Method | Revisions | Correct | Stale previous gold | Other wrong |
|---|---|---:|---:|---:|---:|
| seen | readout | 241 | 138.00 | 90.67 | 12.33 |
| seen | scalar | 241 | 132.67 | 97.00 | 11.33 |
| seen | selective | 241 | 135.33 | 93.33 | 12.33 |
| seen | selective_no_lexical | 241 | 97.00 | 102.33 | 41.67 |
| seen | candidate_gru | 241 | 136.00 | 94.00 | 11.00 |
| seen | literal | 241 | 120.00 | 57.00 | 64.00 |
| unseen | readout | 203 | 127.00 | 59.67 | 16.33 |
| unseen | scalar | 203 | 147.33 | 44.00 | 11.67 |
| unseen | selective | 203 | 149.33 | 43.00 | 10.67 |
| unseen | selective_no_lexical | 203 | 80.00 | 71.67 | 51.33 |
| unseen | candidate_gru | 203 | 143.33 | 28.67 | 31.00 |
| unseen | literal | 203 | 162.00 | 15.00 | 26.00 |

## Current evidence

These are lexical matches, not verified intent. SYSTEM mentions may propose or question a value.

| Panel | Gold-matching unique mention | Queries | Scalar correct | Selective correct |
|---|---|---:|---:|---:|
| seen | both_gold | 2 | 2.00 | 2.00 |
| seen | user_gold_only | 116 | 112.00 | 113.67 |
| seen | system_gold_only | 9 | 8.67 | 8.33 |
| seen | neither_gold | 114 | 10.00 | 11.33 |
| unseen | both_gold | 1 | 1.00 | 1.00 |
| unseen | user_gold_only | 161 | 139.00 | 140.33 |
| unseen | system_gold_only | 3 | 3.00 | 3.00 |
| unseen | neither_gold | 38 | 4.33 | 5.00 |

## Prior correctness

| Panel | Method | All revision errors | Stale errors | Stale after correct prior | Stale on adjacent USER step |
|---|---|---:|---:|---:|---:|
| seen | scalar | 108.33 | 97.00 | 74.33 | 97.00 |
| seen | selective | 105.67 | 93.33 | 69.00 | 93.33 |
| unseen | scalar | 55.67 | 44.00 | 35.67 | 44.00 |
| unseen | selective | 53.67 | 43.00 | 35.33 | 43.00 |

## Retention errors and new observed errors

A new error means the previous scored prediction was correct on the same retained gold label. This observes scored endpoints, not the internal update that produced the error.

| Panel | Method | Retention queries | Wrong | New after correct | Already wrong | No previous prediction |
|---|---|---:|---:|---:|---:|---:|
| seen | readout | 26071 | 6789.67 | 2441.33 | 3603.33 | 745.00 |
| seen | scalar | 26071 | 4332.33 | 807.67 | 3058.33 | 466.33 |
| seen | selective | 26071 | 4793.67 | 955.00 | 3342.00 | 496.67 |
| seen | selective_no_lexical | 26071 | 6419.67 | 952.67 | 4815.67 | 651.33 |
| seen | candidate_gru | 26071 | 4114.67 | 893.33 | 2755.00 | 466.33 |
| seen | literal | 26071 | 7688.00 | 219.00 | 7316.00 | 153.00 |
| unseen | readout | 30550 | 8622.33 | 2280.33 | 5624.00 | 718.00 |
| unseen | scalar | 30550 | 7674.00 | 1332.00 | 5732.00 | 610.00 |
| unseen | selective | 30550 | 7778.67 | 1306.67 | 5724.67 | 747.33 |
| unseen | selective_no_lexical | 30550 | 9653.67 | 928.00 | 8092.00 | 633.67 |
| unseen | candidate_gru | 30550 | 6979.00 | 1163.67 | 5260.67 | 554.67 |
| unseen | literal | 30550 | 6082.00 | 440.00 | 5173.00 | 469.00 |

## Assigned boolean and DONTCARE targets

| Panel | Subgroup | Queries | Method | Correct | Literal only | Neural only | Both wrong |
|---|---|---:|---|---:|---:|---:|---:|
| seen | assigned_boolean | 1938 | scalar | 1691.67 | 0.00 | 1691.67 | 246.33 |
| seen | assigned_boolean | 1938 | selective | 1688.33 | 0.00 | 1688.33 | 249.67 |
| seen | label/dontcare | 109 | scalar | 0.00 | 0.00 | 0.00 | 109.00 |
| seen | label/dontcare | 109 | selective | 0.00 | 0.00 | 0.00 | 109.00 |
| unseen | assigned_boolean | 524 | scalar | 27.33 | 0.00 | 27.33 | 496.67 |
| unseen | assigned_boolean | 524 | selective | 34.67 | 0.00 | 34.67 | 489.33 |
| unseen | label/dontcare | 179 | scalar | 0.00 | 0.00 | 0.00 | 179.00 |
| unseen | label/dontcare | 179 | selective | 0.00 | 0.00 | 0.00 | 179.00 |

## What the counts identify

On seen revisions, 93.33 of 105.67 mean selective errors select the old gold value. 69.00 of those followed a correct previous scored prediction. A unique current USER or preceding SYSTEM mention matches the new gold on 127/241 revisions. These counts distinguish error patterns, not their internal cause.
On unseen revisions, 43.00 of 53.67 mean selective errors select the old gold value. 35.33 of those followed a correct previous scored prediction. A unique current USER or preceding SYSTEM mention matches the new gold on 165/203 revisions. These counts distinguish error patterns, not their internal cause.

## Interpretation limits

- Already exposed official development only; no test content, fitting, inference, encoding or threshold search.
- A lexical match does not establish intent/relevance. Missing literal evidence does not establish missing semantic evidence.
- No saved gate/departure probabilities or intermediate belief states: internal inertia and observation error cannot be causally separated.
- The frozen 7/13 continuation failure is unchanged; subgroup diagnostics are not new success criteria.
- Lexical extraction is pinned-source bound, not regenerated from raw utterances in this diagnostic.

The aggregate files contain no raw dialogue examples or individual saved predictions. Subgroups describe where current methods fail; they do not identify a causal mechanism or validate a proposed replacement.
