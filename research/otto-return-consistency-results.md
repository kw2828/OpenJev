# Saved branch consistency: a narrow diagnostic

**All 144 saved model-prefix rows were processed and independently checked.** The minimum-linear model has larger average greedy-backup inconsistencies on these selected states than either MLP control. This motivates a learning-objective comparison; it does not establish why the autonomous study failed.

The scope is exactly steps 0 through 7 of two length-three, initial-hit-one teacher episodes: one TRAIN episode lasting 79 steps and one VALID episode lasting 14 steps. All nine fixed models see the same 16 prefixes. Only eight of the full 1,109 VALID states occur here; this is not representative validation coverage or 144 independent episodes.

[Original design](otto-return-consistency-design.md) · [Terminal-packet correction](otto-return-consistency-repair.md) · [All rows](../output/otto-return-consistency-v2/run-01/rows.jsonl) · [Summary](../output/otto-return-consistency-v2/run-01/summary.json) · [Independent audit](../output/otto-return-consistency-v2/audit-01/receipt.json)

## What was measured

Current error is the predicted physical remaining cost minus the realized teacher return. Teacher and greedy residuals subtract the current value from the corresponding one-step deployed backup. Switching advantage is the teacher-action backup minus the chosen-action backup. Branch values already use physical units; only the stored current scalar is multiplied by 64.

All sixteen observation branches, the 1e-10 mass floor, biased zero-input values, signed predictions and strict eligible-action near-tie rule are retained. These are consistency measurements of the deployed floored arithmetic, not exact physical Bellman errors or observed counterfactual returns. Lower backup cost can indicate possible policy improvement or inaccurate successor predictions; this diagnostic cannot distinguish them.

## Family means across all three fitting seeds

| Family | Split | Current return MAE | Teacher residual MAE | Greedy residual MAE | Mean greedy residual | Mean switching advantage | Mean teacher disagreements / 8 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| min8 | train | 30.935 | 1.817 | 1.104 | -1.104 | 2.585 | 8.000 |
| min8 | valid | 23.123 | 1.751 | 1.661 | -1.661 | 2.136 | 7.667 |
| mlp8 | train | 37.758 | 1.950 | 0.560 | 0.136 | 1.814 | 7.000 |
| mlp8 | valid | 17.446 | 1.187 | 0.572 | 0.473 | 0.615 | 6.000 |
| homogeneous8 | train | 38.307 | 1.960 | 0.576 | 0.140 | 1.819 | 7.000 |
| homogeneous8 | valid | 16.916 | 1.189 | 0.568 | 0.477 | 0.622 | 6.000 |

Values are in movement-cost units. Means equally weight the three fitting seeds. Disagreement is not an error rate. The noisy realized teacher returns are not optimal values or expected action-value labels.

## Every fit and split

| Fit | Split | Current MAE | Teacher residual MAE | Greedy residual MAE | Mean greedy residual | Switching advantage | Teacher disagreements / 8 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| homogeneous8@10101 | train | 38.470 | 1.966 | 0.552 | 0.121 | 1.845 | 7 |
| homogeneous8@10101 | valid | 16.685 | 1.190 | 0.551 | 0.463 | 0.639 | 6 |
| homogeneous8@10102 | train | 37.698 | 1.947 | 0.598 | 0.161 | 1.786 | 7 |
| homogeneous8@10102 | valid | 17.575 | 1.186 | 0.582 | 0.488 | 0.605 | 6 |
| homogeneous8@10103 | train | 38.753 | 1.966 | 0.578 | 0.138 | 1.828 | 7 |
| homogeneous8@10103 | valid | 16.488 | 1.191 | 0.570 | 0.480 | 0.621 | 6 |
| min8@10101 | train | 30.201 | 1.637 | 1.048 | -1.048 | 2.429 | 8 |
| min8@10101 | valid | 23.957 | 1.704 | 2.118 | -2.118 | 2.755 | 8 |
| min8@10102 | train | 30.953 | 1.998 | 1.227 | -1.227 | 2.764 | 8 |
| min8@10102 | valid | 23.023 | 2.002 | 1.637 | -1.637 | 1.924 | 8 |
| min8@10103 | train | 31.652 | 1.815 | 1.036 | -1.036 | 2.561 | 8 |
| min8@10103 | valid | 22.389 | 1.546 | 1.227 | -1.227 | 1.729 | 7 |
| mlp8@10101 | train | 38.233 | 1.955 | 0.542 | 0.122 | 1.833 | 7 |
| mlp8@10101 | valid | 16.913 | 1.189 | 0.560 | 0.466 | 0.630 | 6 |
| mlp8@10102 | train | 36.873 | 1.939 | 0.576 | 0.153 | 1.786 | 7 |
| mlp8@10102 | valid | 18.366 | 1.183 | 0.581 | 0.477 | 0.602 | 6 |
| mlp8@10103 | train | 38.168 | 1.957 | 0.561 | 0.134 | 1.822 | 7 |
| mlp8@10103 | valid | 17.060 | 1.188 | 0.574 | 0.475 | 0.614 | 6 |

Signed means, MAE, per-seed RMS and all branch contributions are retained in the linked complete records. The published family RMS is the mean of per-seed RMS values, not a pooled RMS. There is no significance test or efficacy gate.

## Execution and next experiment

The corrected saved diagnostic took 2.466 seconds with 35,012,608 bytes peak RSS. It made zero model, training, policy or simulator calls. All 44 current-study payloads, 41 teacher payloads and two original-audit payloads were authenticated before reading numerical records. The separate independent audit checks the joins, arithmetic and reporting without generating new model predictions.

The first attempt failed on terminal packets with an empty legal-action list, before producing any diagnostic rows. Its original source, plan and failed receipt remain preserved. The separately frozen correction changes only that terminal-packet contract. [Failed attempt](../output/otto-return-consistency-v1/execution-witness.json) · [Completed execution](../output/otto-return-consistency-v2/execution-witness.json).

The next proposed learning control compares continued Monte Carlo fitting with delayed observation-backup targets in the same ordinary MLP, plus unchanged-checkpoint references. It requires runtime qualification and fresh evaluation cases. [Concrete design](otto-bellman-control-design.md) · [Research rationale and prior art](otto-learning-direction.md).

No new policy has been trained or evaluated by this diagnostic. The original scalar-return study remains **0/54**, and no recurrent-memory, biological-wiring or novel-architecture advantage is established.
