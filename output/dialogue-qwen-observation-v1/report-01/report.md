# Qwen semantic observation baseline

Saved-only complete-cohort observation comparison on exposed official TRAIN with correct previous gold. Candidate log probabilities, exact prompt-order ties, metrics and decisions are reconstructed. Public text/lexical provenance and actual inference/model identity are source-bound execution witnesses, not replayed. Unsaved full-vocabulary logits cannot be reconstructed: their saved partition witnesses are checked for mathematical consistency only. No architecture, autonomous-memory or calibration claim. Historical controls are separately fitted references; aggregate seed comparisons are not row-paired repairs.

| Arm | Changed accuracy | Retained error | Overall NLL | Overall Brier |
|---|---:|---:|---:|---:|
| current | 83.7370% | 17.4009% | 1.587058 | 0.309482 |
| history4 | 82.3529% | 21.6683% | 2.308462 | 0.400953 |

semantic_strength: behavioral FAIL; proper-score nonregression FAIL.

added_history: behavioral FAIL; proper-score nonregression FAIL.

All row, equal-service and equal-dialogue metrics, all historical seeds, sparse-category denominators, per-service results and paired repairs/harms are retained in summary.json. No combined architecture gate.

Preparation 12.963s; pilot 17.564s; full run 3229.396s. Pilot decisions were repeated and paid separately.
