# Normalized dialogue-copy error diagnosis

Descriptive saved-output analysis on already exposed development data. No model or encoder calls. The frozen thirteen-check rule is unchanged; this diagnostic selects no new winner.

Accuracy and count summaries below average all three fits per method on the same examples. Literal carry is one deterministic reference. Exact per-seed counts and literal-only/neural-only partitions are in diagnostic.json. A dash means zero denominator.

| Panel | Method | Boolean n | Boolean % | True % | False % | DONTCARE n | DONTCARE % | Boolean vs literal pp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| seen | readout | 1938 | 79.74 | 79.74 | - | 109 | 11.01 | 79.74 |
| seen | scalar | 1938 | 87.29 | 87.29 | - | 109 | 0.00 | 87.29 |
| seen | selective | 1938 | 87.12 | 87.12 | - | 109 | 0.00 | 87.12 |
| seen | selective_no_lexical | 1938 | 87.15 | 87.15 | - | 109 | 3.06 | 87.15 |
| seen | candidate_gru | 1938 | 86.03 | 86.03 | - | 109 | 2.14 | 86.03 |
| seen | literal | 1938 | 0.00 | 0.00 | - | 109 | 0.00 | 0.00 |
| unseen | readout | 524 | 0.38 | 0.38 | - | 179 | 0.00 | 0.38 |
| unseen | scalar | 524 | 5.22 | 5.22 | - | 179 | 0.00 | 5.22 |
| unseen | selective | 524 | 6.62 | 6.62 | - | 179 | 0.00 | 6.62 |
| unseen | selective_no_lexical | 524 | 12.02 | 12.02 | - | 179 | 1.49 | 12.02 |
| unseen | candidate_gru | 524 | 0.00 | 0.00 | - | 179 | 3.35 | 0.00 |
| unseen | literal | 524 | 0.00 | 0.00 | - | 179 | 0.00 | 0.00 |

Stale means the wrong choice equals previous scored gold, not proof of an internal carry operation.

| Panel | Method | Revision n | Correct | Stale | Other wrong | Stale after correct prior | Revision vs literal pp |
|---|---|---:|---:|---:|---:|---:|---:|
| seen | readout | 241 | 138.00 | 90.67 | 12.33 | 73.67 | 7.47 |
| seen | scalar | 241 | 132.67 | 97.00 | 11.33 | 74.33 | 5.26 |
| seen | selective | 241 | 135.33 | 93.33 | 12.33 | 69.00 | 6.36 |
| seen | selective_no_lexical | 241 | 97.00 | 102.33 | 41.67 | 83.33 | -9.54 |
| seen | candidate_gru | 241 | 136.00 | 94.00 | 11.00 | 71.67 | 6.64 |
| seen | literal | 241 | 120.00 | 57.00 | 64.00 | 57.00 | 0.00 |
| unseen | readout | 203 | 127.00 | 59.67 | 16.33 | 54.00 | -17.24 |
| unseen | scalar | 203 | 147.33 | 44.00 | 11.67 | 35.67 | -7.22 |
| unseen | selective | 203 | 149.33 | 43.00 | 10.67 | 35.33 | -6.24 |
| unseen | selective_no_lexical | 203 | 80.00 | 71.67 | 51.33 | 62.00 | -40.39 |
| unseen | candidate_gru | 203 | 143.33 | 28.33 | 31.33 | 21.00 | -9.20 |
| unseen | literal | 203 | 162.00 | 15.00 | 26.00 | 15.00 | 0.00 |

## Limits

- Exposed development only; this is descriptive subgroup analysis outside the unchanged thirteen-check rule, not a new winner or continuation test.
- Historical extraction and count functions are reused by exact source hash. This is comparable arithmetic, not an independent reimplementation of those definitions.
- No neural execution or hidden-state reconstruction. Internal normalization is authenticated by the completed V2 report, not reverified from saved choices.
- Stale predictions and poor boolean/DONTCARE accuracy are error patterns, not causal proof of memory inertia, lost encoder information, or failed semantic understanding.
- Lexical USER/SYSTEM matches are pinned saved features, not intent labels. SYSTEM text may ask about or reject a value. Raw dialogue text is not reprocessed.
- The prior scored frame can be separated by public steps. No intermediate predictions are invented, and the three seeds share the same examples.
- The original V1 normalization defect compromises operator comparisons. No V1-to-V2 subgroup improvement is attributed to a representation or architecture change here.
