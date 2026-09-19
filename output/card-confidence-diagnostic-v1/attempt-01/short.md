# Saved C-policy confidence and choice diagnostic

Exploratory analysis of all 576 completed games across three families and all three fitted seeds. Existing gates are unchanged.

| Family | Second-card decisions | Mismatches | Unseen selected | Seen-hidden wrong top rank | Seen-hidden correct top rank, wrong pending rank | Visible selected |
|---|---:|---:|---:|---:|---:|---:|
| kalman | 9757 | 5273 | 3837 | 45 | 1391 | 0 |
| innovation_local | 9783 | 5322 | 3882 | 42 | 1398 | 0 |
| gated_delta | 9800 | 5364 | 3869 | 34 | 1461 | 0 |

| Family | Eligible hidden queries | Top-rank accuracy | Raw mean max confidence | NLL (nats) | Brier (13-rank sum) |
|---|---:|---:|---:|---:|---:|
| kalman | 105942/106516 | 99.4611% | 81.0838% | 0.226938 | 0.061477 |
| innovation_local | 106474/107097 | 99.4183% | 80.7222% | 0.231984 | 0.062904 |
| gated_delta | 106878/107406 | 99.5084% | 80.4037% | 0.234913 | 0.063966 |

| Family / mismatch category | Count | Repeats | Mean raw true-rank p | Mean raw pending-rank p | Mean raw max p |
|---|---:|---:|---:|---:|---:|
| kalman / unseen_selected | 3837 | 0 | 0.073449 | 0.083937 | 0.317672 |
| kalman / seen_hidden_top_rank_wrong | 45 | 6 | 0.234897 | 0.295470 | 0.397296 |
| kalman / seen_hidden_top_rank_correct_different_pending | 1391 | 276 | 0.630697 | 0.115563 | 0.630697 |
| innovation_local / unseen_selected | 3882 | 0 | 0.073226 | 0.084819 | 0.312880 |
| innovation_local / seen_hidden_top_rank_wrong | 42 | 4 | 0.213516 | 0.282492 | 0.387266 |
| innovation_local / seen_hidden_top_rank_correct_different_pending | 1398 | 290 | 0.633186 | 0.114882 | 0.633186 |
| gated_delta / unseen_selected | 3869 | 0 | 0.074047 | 0.078646 | 0.295689 |
| gated_delta / seen_hidden_top_rank_wrong | 34 | 4 | 0.226356 | 0.229349 | 0.383277 |
| gated_delta / seen_hidden_top_rank_correct_different_pending | 1461 | 297 | 0.641795 | 0.114860 | 0.641795 |

Raw unseen predictions are not used by C: the picker substitutes uniform 1/13 probabilities. Seen-hidden recall targets come only from earlier public frames. NLL and Brier use fixed row normalization, with no evaluation-set tuning. All per-fit denominators, repeated-mismatch probability summaries, age >32 queries and authenticated episode hashes are retained in results.json.

These are conditional saved-history diagnostics. A small number of recall errors can alter subsequent discovery and repeated choices; probability overlap can matter even when the top rank is correct. No counterfactual policy or architecture benefit was evaluated.

Analysis source SHA-256: `b737467295610b6965ea2f35dc110b907d01690e7d9b674907c1b8e4d6c34586`.
Execution completion SHA-256: `cc0d22c85fe8c9dadb36ce4b6c4fb047ad920c05a929752b1d0df2be94df22ba`.
Results JSON SHA-256: `0eb5a58b13a25cd2b24333a9a1bb738cd3fbfb10cb627a483dcd618d2a5d3107`.
