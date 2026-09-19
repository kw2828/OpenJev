# Common-root report and audit review

**Review clear after the corrections below.** This is a source review with synthetic scalar tests before the diagnostic run, not a result or permission to infer learned adaptation. No production outcomes were read. The reviewer did not run a simulator, planner, learned model, RNG, or the report renderer.

## Bound sources

| Repository-relative path | SHA-256 |
|---|---|
| `output/reacher-common-root-diagnostic-v1/report_results.py` | `56395fff05c2a243686ef74bedc5634075d38df4bad12994bc3c3eebf1ff5911` |
| `output/reacher-common-root-diagnostic-v1/run_diagnostic.py` | `e082c9e517bd33c6d10831316226483a628dbfbe3cf751fb9d13a2b52e896790` |
| `output/reacher-common-root-diagnostic-v1/protocol.json` | `dc1a48b81f7e9a29b610d53dd97825afebc650722d9d20b75b1f1a1f30cb9561` |
| `output/reacher-common-root-diagnostic-v1/report-arithmetic.json` | `b8a37d706210273889efba2db8656303ac4f87f02e9ae01f51294abc97f58788` |
| `tests/test_reacher_common_root_report.py` | `a7c05077309fb30cca7a5427889db2b7c883bbb06987609e3a262f90a37298c4` |
| `src/openjev/research/reacher_common_root_audit.py` | `5cf5910567082d048054c54f465e8d94c57dcdcd35540a8fbeff6875c2042f97` |
| `tests/test_reacher_common_root_audit.py` | `b7b7a067b7f7e153e430eb066a8a812b794000a3a190f685d9343dc8d529937c` |

## Findings and corrections

The initial reporter could reject an exactly 3% decimal improvement because of floating-point averaging. Using rounded floating totals alone also merged a threshold with the next representable worse value. The final, separately declared arithmetic clarification uses exact sums of `Fraction(str(float(slot_mean_cost)))` and exact decimal margins for the gate. Displayed statistics use `math.fsum`. There is no numerical slack, fixed-place rounding, changed margin, or regenerated draw. The driver binds the clarification to the completed preparation receipt and includes it in the copied source set.

Two statements were corrected: union-score winners are chosen by frozen modeled-score rules, but need not be temporally computed before native branch evaluation; the descriptive pool minimum is the minimum **four-branch mean** over union candidates, not a separate best choice within each noise branch.

## Report semantics

All twelve declared role/root slots and their audits are required. The six selected identity slots map correctly to nominal and actual-gain short A, best(A,B), and long A, in that order. Short sequences are held to 24 actions by the branch helper. The search contrast compares actual-gain best(A,B) to A; the horizon contrast compares actual-gain long A to best(A,B). The gain contrast selects on each modeled full-24 score vector over the same deduplicated union. `argmax` retains earliest first-occurrence ties. No native cost enters a deployable selection rule.

Native cost is the negative, unclipped native reward averaged over 24 actions, then equally over the four shared noise branches. The separate first-12 diagnostic uses exactly twelve actions. Each gate requires all four conditions: at least 3% pooled improvement, at least 0.001 cost/action pooled improvement, nonworse mean on every source trajectory, and at least eight nonworse slots. Coverage is exactly four slots per role. Missing, nonfinite, negative candidate costs, or nonpositive baselines are rejected.

The decision tree requires the gain contrast to pass. Gain plus horizon permits preparation of only a separate longer-horizon closed-loop engineering correction; otherwise gain plus search permits only the extra-short-search correction. All other combinations stop this controller/task recipe as evidence for learned adaptation. Search cannot relabel a failed horizon comparison as a horizon success. No branch of this decision tree authorizes a scientific launch or neural fitting.

Repeated startup roots are retained as declared identity slots and checked for identical branch artifacts, not treated as independent samples. The report identifies privileged state/gain, the exposed case, held-tail open-loop execution, and the common union populated by both gain-conditioned searches. These limits prevent a claim of closed-loop improvement, online gain identification, matched runtime, or architectural novelty.

## Verification and remaining boundaries

Reviewer-executed command: `PYTHONPATH=src .venv-robotics/bin/python -m pytest -q tests/test_reacher_common_root_report.py`: **35 passed in 0.14s; Ruff clean**. The tests compile only the actual report's pure `useful` function and scalar decision tree from its AST, without importing its transitive backend modules. Coverage includes exact margins, next-representable worse scalar values, independent trajectory/slot failures, malformed coverage/costs, contrast separation, and all eight decision combinations.

The independent native auditor was read statically, including its corruption tests. It authenticates complete slot/branch membership and externally supplied roots/noise; reconstructs the score-selected union independently; checks shared A-prefix proposals; replays nominal reset-state searches/cross scores and full-integration-state noisy branches separately; verifies sequential float32 clipped scores and cached 3D native reward with applied effort once; and reconciles work and disjoint timing. Its author reports **39 tests passed in 6.88s, Ruff clean**. This reviewer did not rerun those native tests.

The driver preserves A-on-restart ties, full candidate/selected accounting, event-free root eligibility and complete payload evidence. Maximum scheduled transitions are 294,912 search candidates, 72 selected advances, 40,320 cross-scored transitions and 80,640 branch transitions; actual unique-union counts reduce the last two. The latest driver includes initial authentication/setup inside its cooperative cap and includes payload hashing in each slot wall time. Final slot-receipt writes are charged by the enclosing execution clock. Runtime and original source/innovation provenance remain enclosing-driver responsibilities; this static review cannot certify a future execution.
