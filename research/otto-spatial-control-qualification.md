# Spatial controllers: deployment qualification passed

All **780 prescribed NumPy/Torch64 comparisons passed**, including exact agreement on the eligible action. This qualifies the numerical deployment path for all fifteen saved heads. It does not establish search competence, efficiency or an architectural advantage.

The [frozen protocol](otto-spatial-control-protocol.md) retains spatial, neighbor-free, CNN, dense128 and statistics heads at all three fitting seeds, 10101-10103. No checkpoint was selected or retrained. Each head received the same **52 input tuples**: the first eight TRAIN and eight VALID public-state prefixes, followed by 36 fixed synthetic tuples. The synthetic set crosses sensing lengths 3, 4 and 5; center, lower-corner and upper-corner positions; and asymmetric, successor-point, zero and subfloor-mass beliefs. Shared arrays were stored once. Cached targets and legacy feature columns were not decoded.

Each tuple produces sixteen observation branches and four action costs. Values use physical remaining-move units, including the factor of 64 applied to normalized predictions. The frozen elementwise tolerance was `1e-8 + 1e-10 * abs(Torch64 reference)` for values and costs, with exact eligible-action agreement required separately. Maximum absolute errors were:

| Comparison | Maximum absolute error |
| --- | ---: |
| Physical branch value | 3.694822225952521e-13 |
| Action cost | 3.126388037344441e-13 |

The original supervised qualification completed once with **15 NumPy head loads, 15 Torch reference restores, 780 NumPy forwards and 780 Torch64 forwards**. Every attempted call returned; each forward evaluated sixteen branches. It performed **zero new environment, training or external model calls**, and completed zero autonomous episodes. The [original terminal](../output/otto-spatial-control-v1/qualification-process-01.terminal.json) records exit 0, no timeout, successful reaping and an absent process group.

| Timing scope | Observed seconds |
| --- | ---: |
| Qualification worker | 13.806553708 |
| Original supervisor, including worker and cleanup | 14.120537625 |
| Separate saved-output audit | 9.959710333 |

The worker and parent intervals are nested, so they should not be added. These are single execution timings, not a deployment-latency benchmark.

The separately completed audit agreed after **780 additional saved NumPy forwards**, covering 12,480 branch rows. Its final receipt records **49,004 comparisons**. It independently reconstructed tuple geometry, branches, comparison arithmetic and evidence joins. Its neural readout shares the qualified NumPy algebra with production; this is not a third independent neural implementation. Torch execution, historical input truth and timing remain authenticated original-execution evidence. The audit ran no training or simulator calls.

The [prior scalar result](otto-spatial-study-results.md) remains unchanged: spatial did not improve mean validation MSE over its matched neighbor-free control. Correct deployment cannot turn that result into an efficacy claim. The next stage is the separately frozen **1,152-episode autonomous comparison**, retaining all fifteen heads and analytic control across 72 paired environmental cases. That comparison is pending here; none of its 66 continuation conditions has been evaluated by this qualification.

Evidence: [qualification plan](../output/otto-spatial-control-v1/qualification-plan-01.json), [worker receipt](../output/otto-spatial-control-v1/qualification-01/receipt.json), [qualification summary](../output/otto-spatial-control-v1/qualification-01/summary.json), [all 780 comparisons](../output/otto-spatial-control-v1/qualification-01/parity.jsonl), [52 tuple identities](../output/otto-spatial-control-v1/qualification-01/qualification-tuples.jsonl), [audit receipt](../output/otto-spatial-control-v1/qualification-audit-01/receipt.json), [audit summary](../output/otto-spatial-control-v1/qualification-audit-01/summary.json), and [original execution witness](../output/otto-spatial-control-v1/qualification-execution-witness-01.json). The [runner](../scripts/study_otto_spatial_control.py) and [saved-output auditor](../scripts/audit_otto_spatial_control.py) are bound by the frozen 219-file source closure.
