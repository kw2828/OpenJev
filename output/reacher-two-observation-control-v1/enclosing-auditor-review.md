# Independent enclosing-auditor review

Status: no remaining material integration blocker found in this bounded static review. This supports proceeding to a retained whole-tree engineering rehearsal; it is not a scientific result or permission to skip that rehearsal.

## Reviewed source identities

| File | SHA-256 |
| --- | --- |
| `scripts/audit_reacher_two_observation_study.py` | `89be8baa6f2bd0bdfe82172a83d374f0204cd76f9a6327f518d15940f2b7fcf6` |
| `tests/test_audit_reacher_two_observation_study.py` | `d7929aff6e2ce9c1a5c837a57b7f43ca079f9632dcb368ee37342eafa78fd653` |
| `scripts/reacher_two_observation_study.py` | `e3a8dcf2e69f6cb6fee01feca5e1984e728fb5a65d0cfbb7326c119cae2c232d` |
| `tests/test_reacher_two_observation_study.py` | `e5a5b773a5fe797e37bb11f151bdb522f53be6c099a94eb59b7c076e993204ea` |

The reviewer independently rehashed these exact local bytes when recording this memo. The reviewer authored the runner, but did not author or edit the enclosing auditor or its tests.

## Finding and correction

The history episode helper records the controller's outer `score_search` duration in the saved search trace. Scoring metadata separately records an inner duration measured earlier. Requiring those durations to be equal would reject a valid execution.

The final auditor requires the inner duration to be finite and no greater than the outer trace duration, allowing the documented `1e-6` comparison tolerance. It separately requires the controller's recorded outer duration to match the trace exactly. Existing controller and episode source bytes were preserved.

The added regression reaches `audit_history_scoring`, the original rejection path: an inner duration of 0.20 seconds and an outer duration of 0.21 seconds are accepted; 0.22 seconds versus 0.21 seconds are rejected. Geometry replay functions are stubbed in this synthetic test. The reviewer inspected the correction and this regression without executing them.

## Scope and limits

The static comparison covered runner/auditor artifact schemas and exact membership, inherited and fresh fit lineage, actual-class restoration and before/after snapshots, real public-history and issued-command alignment, candidate and selected score diagnostics, the complete 12-observation/11-transition reconstruction accounting, reference input boundaries, phase chronology, failure handling, and cost reconciliation without double-counting inherited ancestry. A failed qualification result remains a completed, failed result rather than being suppressed.

The auditor author reports 37 focused synthetic tests passing in 1.26 seconds and Ruff passing. The reviewer did not rerun those tests, inspect production inputs or outcomes, derive scientific streams, or call learned models, optimizers, or native simulation. The runner's separately reported 45 tests exercise synthetic orchestration; they do not establish runtime capacity or efficacy.

Saved hashes, receipts, counters, and public-state arithmetic support artifact consistency. They do not independently prove every learned forward, gradient, optimizer transition, or neural hidden-state value. A complete retained engineering rehearsal and its independent saved-output audit remain necessary before a prospective scientific freeze and run.
