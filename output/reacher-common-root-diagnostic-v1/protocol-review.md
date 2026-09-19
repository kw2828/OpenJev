# Independent pre-run protocol review

**Clear for the proposed bounded engineering diagnostic, subject to final component tests and the enclosing source freeze.** I found no remaining design or static implementation blocker in the files below. This review did not prepare inputs, derive seeds, run tests, call a simulator/model/planner, or inspect new diagnostic outcomes. It used source and completed earlier engineering records only.

| Reviewed file | SHA-256 |
|---|---|
| [protocol.json](protocol.json) | `dc1a48b81f7e9a29b610d53dd97825afebc650722d9d20b75b1f1a1f30cb9561` |
| [prepare_inputs.py](prepare_inputs.py) | `c3616c383cc610a4338be7334b1c6b16424f8a66abf88eabd7ade9f2439d0e47` |
| [Arithmetic amendment](report-arithmetic.json) | `b8a37d706210273889efba2db8656303ac4f87f02e9ae01f51294abc97f58788` |
| [run_diagnostic.py](run_diagnostic.py) | `e082c9e517bd33c6d10831316226483a628dbfbe3cf751fb9d13a2b52e896790` |
| [report_results.py](report_results.py) | `56395fff05c2a243686ef74bedc5634075d38df4bad12994bc3c3eebf1ff5911` |
| [Native branch/union component](../../src/openjev/research/reacher_common_root_branches.py) | `d24c24d3aa5dc71fea4de6264f661787e3d0bd959c49d68b9f5d7131bcfdb7eb` |
| [Independent numerical auditor](../../src/openjev/research/reacher_common_root_audit.py) | `5cf5910567082d048054c54f465e8d94c57dcdcd35540a8fbeff6875c2042f97` |

The earlier [follow-up design](../reacher-proposal-memory-engineering-v1/followup-design.md) remains prospective. The separate input-preparation receipt remains `c5aa5252075d86d93dff635594da21a05db5648aeda191e58490a09e28c74543`; I rehashed that receipt and the unchanged original protocol/preparer while reviewing the amendment. This review does not substitute for preparation or later execution/audit receipts.

## State and information boundaries

The 12 slots are roots 0/50/100/150 from all three cold trajectories, without selecting favorable states. Root provenance is reconstructed from the hash-bound source episode. Each root uses `audit.decision_states[t]`, after public target installation. The preceding transition's `native_after` retains the old target and would be wrong here. Event-free eligibility checks all 25 target states and all 24 action gains.

I checked the completed nominal source record: the clocks at 50/100/150 are `1.0000000000000007`, `2.0000000000000013`, and `2.99999999999998`. The branch helper retains these clocks and the complete 49-element integration state. It resets private data, restores that state and calls forward; the saved original vector and post-forward initialized state are separately retained. It does not use the old 50-step-limited restore helper. Current gains are explicitly 0.7/0.7/1.3/1.3, rather than inferred from a model whose startup gear is still nominal.

All searches receive identical native qpos/qvel within a slot. Their nominal solver reset is deliberately different from full-state native branch continuation. Both gain assumptions are privileged state diagnostics; neither is a public observer. Targets and gains remain current and fixed inside each interval. No future event, branch noise or native outcome enters search.

## Matching and selection

Short A is shared with short A+B and paid once in actual execution. A+B chooses solely by modeled H12 score, with A winning an exact tie. H24 A and H12 A+B each pay 6,144 candidate transitions, but differ in callback count, selected advances and action-space dimension. Fresh bank setup is charged. The comparison does not claim matched latency or identical optimization difficulty.

The first four blocks of every A input field retain the original saved innovations, including unused fields. New tails and B are prepared separately, paired across roles/gain assumptions, then loaded without RNG draws during execution. Initial proposals share the expected prefixes; later adaptive proposals need not agree.

Union slots are exactly initial64 plus six score-selected plans. Short plans hold their final command through action24. Signed-zero normalization, first-occurrence deduplication and the identity mapping preserve every selector. The numerical audit independently reconstructs these choices and exact saved-score CEM selection. Four common explicit noise branches evaluate the resulting union; outcomes never choose a reported method. The later gain-ranking selection uses modeled scores only, even though its IDs are computed in the report. The descriptive union minimum is the minimum **four-branch mean**, not a different winner per noise branch.

## Fixed continuation rule and costs

Each named contrast independently requires at least 3% pooled improvement, at least 0.001 native cost/action improvement, all three trajectory means nonworse, and at least 8/12 slot means nonworse. Search, horizon and gain flags remain separate. Repeated startup states and shared disturbances are explicitly dependent observations, not 12 independent trials.

Before any full diagnostic search or branch evaluation, boundary tests exposed that rounded floating-point totals could merge a threshold with the next representable worse slot cost. The separately bound arithmetic amendment takes explicit precedence over the original protocol's aggregation clause. Gate totals now sum `Fraction(str(float(slot_mean_cost)))`, using exact decimal `0.03` and `0.001` thresholds and inclusive comparisons; trajectory nonworse checks use the same exact sums. Slot order comparisons retain their equivalent finite-float ordering. `math.fsum` is used for displayed means and improvements only. This defines arithmetic on canonical decimal representations of the computed slot costs, not exact real-valued simulator rewards. It introduces no rounding to a fixed precision, epsilon, changed margin or best-subset exception. Roots, prepared draws, work, caps and decision precedence remain unchanged. The driver binds the amendment in its source snapshot and checks its original preparation-receipt hash before evaluation; the reporter verifies that bound source map.

The predeclared next-design precedence is horizon only if both horizon and gain pass; otherwise extra short search only if search and gain pass; otherwise close this recipe as evidence for learned adaptation. This chooses a possible later engineering correction, not a demonstrated generalized controller. No result launches fresh qualification or changes an old gate.

The fixed work is 294,912 search candidate transitions plus 72 selected advances. Deduplicated union work is at most 40,320 nominal cross-scoring and 80,640 noisy native transitions, totaling at most 415,944. Independent replay is additional. The 300-second execution and separate 300-second audit caps include their setup/checking work; there is no automatic extension. Complete slot timing includes payload hashes, with final completion writing covered by enclosing execution time. Model construction, copies and serialization prevent interpreting these timings as deployment latency.

## Failure handling and remaining scope

Requested fixes are present: exclusive preparation/attempt allocation precedes guarded lineage validation; execution source checks are inside its cap; slot and outer completion demotions preserve the original exception; late failure cannot leave an accepted completion; slot timing follows payload hashing. The reported test provenance is included in the new source closure. I independently rehashed the protected 116-source, 136-source and prior 20-source maps: all matched. No protected source was edited.

This is an open-loop diagnosis on exposed histories. Holding a short plan's tail is not H12 MPC replanning. Ranking a union populated by both gain-conditioned searches does not measure the utility or cost of an online identifier. A finite-pool gap is not a global-optimality bound. Even a positive result establishes neither useful recurrent learned state nor architectural novelty. Completed component tests, actual source freeze, successful execution and independent numerical replay remain separate prerequisites for interpreting any new result.
