# Teacher-continuation costs: saved-record component

The [cost-summary component](../src/openjev/research/otto_teacher_costs.py) passed **18 fabricated-record tests** and Ruff. It prepares one part of the [conditional action-cost follow-up](otto-action-cost-followup.md). It generates no rollouts, trains no policy and establishes no task-performance or architecture result. Real label collection remains unadmitted. The later [completed spatial-control audit](otto-spatial-control-results.md) supplies the prerequisite comparison, not sampler qualification.

The caller supplies the expected replicate IDs, eligible first actions and horizon separately from the records. The reducer rejects missing, duplicate or extra pairs. A completed unsuccessful rollout must reach the full horizon; a missing or interrupted rollout cannot be silently counted as censoring. Success exactly at the horizon remains a success, with the same capped cost as an unsuccessful horizon. Every cost includes the forced first action.

Outputs include mean costs, means centered across eligible actions, success/censoring fractions and every paired action-cost difference. Differences are first-action cost minus second-action cost, so negative is better for the first action. Pairing uses replicate IDs, and declaration or record order does not affect the result. Paired standard error is descriptive and is undefined with one replicate. A constant observed difference can have zero empirical standard error without establishing zero population uncertainty.

For a fixed nonterminal anchor, costs are in `[1, H]` and differences in `[-(H-1), H-1]`. With `N` independent replicate vectors and `M >= 1` unordered action pairs, the component uses the Hoeffding radius

```text
2 * (H - 1) * sqrt((log(2 * M) - log(alpha)) / (2 * N))
```

With only one eligible action, the component returns no pair intervals. Otherwise the intervals intersect the known difference support. The union bound covers all declared action pairs at this one anchor, conditional on independent, identically distributed replicate vectors from the declared public belief and observation model, one fixed teacher/horizon, and a sample allocation fixed in advance. Common randomness across actions within a replicate is allowed. The reducer cannot establish those provenance or sampling assumptions. The bound does not cover adaptive sampling, selection among anchors, or policy effectiveness, and it can be too conservative to resolve useful gaps.

Independent source review found a precision defect before the first test execution: subtracting a rounded large mean distorted the paired standard error. Exact integer sufficient statistics now preserve both variance and centered costs before final floating-point division. The regression fixtures include paired costs near the maximum supported integer horizon, where the correct standard error is `0.5`, plus small differences on large shared offsets.

The [tests](../tests/test_otto_teacher_costs.py) also cover hand-computed means and uncertainty, all six pairs for four actions, success at the cap, one-action/one-replicate panels, horizon-one ties, very small positive alpha, invalid types and incomplete panels. These are mathematical and accounting checks on fabricated records. No empirical records, models, simulators, training or label collection were invoked.

Evidence: [original engineering receipt](../output/otto-teacher-cost-engineering-v1/attempt-01/receipt.json), [test log](../output/otto-teacher-cost-engineering-v1/attempt-01/pytest.log), [code-check log](../output/otto-teacher-cost-engineering-v1/attempt-01/ruff.log). The original captured command completed with exit 0, tool chunk `bb8d3a`. The receipt SHA-256 is `044aa77818c5648f5f54eb2afd06d58a1bb16f5a1c59c7becea14c5d7b25f9a4`.

A future study still needs a qualified public-only rollout sampler, fixed anchor and sample allocations, complete cost accounting, a separately frozen admission rule and fresh autonomous evaluation. This component does not supply those missing parts.

The subsequent [analytical budget check](otto-teacher-cost-budget.md) shows why the range-only interval should remain an uncertainty reference rather than a mandatory confident-label gate for a small pilot. No real continuation records were needed for that calculation.
