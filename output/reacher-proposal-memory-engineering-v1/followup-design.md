# One common-root planning diagnostic before learned adaptation

**Recommendation: stop tuning proposal memory. Separately freeze one small common-root comparison of search effort, planning horizon and gain knowledge.** This is a prospective diagnostic on exposed engineering histories, not a new scientific qualification or permission to train. No simulator, model, planner or RNG was called for this note. Existing protocols and failed criteria remain unchanged.

The completed [nine-row report](review-01/report.md) passes only **2/6** fixed comparisons. Shifted plans worsen full cost versus cold by 12.38% for nominal dynamics and 0.30% for public true gain. With true state, they improve versus cold by 4.72% but lose 2.26% to repeating the last action. Exact cold replay passed for all three roles. There is no consistent full-plan-memory advantage to build an architecture claim on.

The earlier [candidate diagnosis](../persistent-dynamics-diagnosis-v1/candidate-results-01/results.json) found no clipping, small first-step distance errors, and better prediction yet worse control for the true-state reference. Accurate one-step dynamics do not certify adequate long-range planning. Closed-loop rows also visit different states, so their costs cannot isolate search, horizon and information effects. Another centering coefficient, terminal penalty or observer change on this case would obscure that distinction.

## Fixed roots and one controlled comparison

Use decision roots **0, 50, 100 and 150** from each of the existing `nominal--cold`, `public_gain--cold` and `true_state--cold` rows: 12 identity slots. These are all four public-goal onsets, chosen by the task schedule rather than observed cost. Startup duplicates remain declared duplicates, not independent cases. Each subsequent 24-action interval ends before any next target change or the saved gain switch at 80.

At each slot, give every condition the **same copied native qpos/qvel and current public target**. Compare assumed gain 1.0 with the true current gain. This is explicitly a privileged diagnostic of dynamics knowledge with state held fixed; it is not a deployable observer or an oracle upper bound. Neither condition receives future goals, future gains, branch noise or a switch flag.

| Condition | Cold searches | Paid candidate transitions at a root | Question |
|---|---|---:|---|
| Short baseline | H12, one CEM256 run A | 3,072 | Current search recipe |
| More short search | H12, independent CEM256 runs A and B; select their best scored sequence | 6,144 | Does more search help at the same objective? |
| Longer planning | H24, one CEM256 run A | 6,144 | Does allocating the same candidate-transition budget to a longer horizon help? |

Reuse run A as the nested short baseline; do not secretly run a third short search. Both restarts retain the unchanged four paid 64-candidate stages, seven anchors, elite count, variance floor, paid mean, earliest global-best rule, geometry score, expected actuator cost and block3. The two-restart comparator is not a new 512-candidate CEM algorithm. Its eight scoring calls and two selected advances differ from the long condition's four calls and one selected advance. Match candidate transitions and report complete costs, not equal latency.

Pair the full H24 innovation tensors across gain assumptions. For A, retain the saved H12 innovations as the first four action blocks and append independently declared tail innovations. B uses a separate declared stream. The H12 runs take the first four blocks. Later CEM banks may differ because scores differ. Bind every input before execution; use one new engineering namespace, no retries or outcome-based extra starts.

## A common candidate set makes the diagnosis interpretable

Construct a fixed union of the common initial 64 H24 sequences and the six selected sequences from the three conditions under two gain assumptions. Extend each H12 sequence to 24 actions by holding its final command. This is an explicit diagnostic continuation rule, not a claim about what receding-horizon MPC would do. Keep all 70 identity slots and an exact first-occurrence deduplication map.

Score each unique sequence under both dynamics assumptions from the same root, retaining separate first-12 and full-24 sequential float32 clipped scores. Also execute each sequence from the same saved native integration state under **four shared, fresh actuator-noise branches**. Restore the complete native state for each branch; charge actual clipped-action cost once. The event-free intervals allow the current target and gain to remain fixed without revealing future events to a planner. Save raw native distance and effort separately. Branch cost per action and modeled clipped score are different quantities.

This permits three concrete diagnostics:

1. **Search:** if the H24 winner's first 12 commands outperform the short search under that same 12-step modeled objective, the short optimizer missed a feasible better prefix. This is an observed finite-pool gap, not a global-optimality certificate.
2. **Horizon:** if the short objective favors one sequence but the 24-step objective and native branches favor another, report that rank reversal. Compare H24 to the doubled short-search budget before attributing it to horizon. Tail extension and open-loop execution limit the inference; it is not a closed-loop control improvement.
3. **Gain knowledge:** on the identical union, select by nominal versus correct-gain scores, then compare their independently evaluated native costs. This separates dynamics-dependent ranking from changes in the candidate pool. Report native regret within this finite union and branch spread; do not select using branch outcomes.

Retain every root, both gains, all condition winners and the zero anchor already in the common bank. Do not average away a failing trajectory or report duplicate startup slots as extra evidence.

## Bound the work and stop

With all 12 slots executed, search pays **294,912 candidate transitions and 72 selected advances**. At most 70 unique sequences per slot add **40,320** nominal cross-scoring transitions and **80,640** native branch transitions. The total is at most **415,944 transitions**, before independent replay, setup, copying, geometry, storage and hashing. Each transition still uses the existing two native integration steps. Audit work is additional. These are checked design counts, not measured runtime or a proposed timing cap. Freeze finite execution/audit caps only after engineering sizing.

This is one diagnostic, not an optimizer sweep. If the extra short search or longer horizon fails to expose a useful native-cost difference and correct gain still gives no consistent within-pool benefit, **stop this task/controller recipe as evidence for neural adaptation**. Do not add another horizon, terminal coefficient or favorable root. If a specific limitation is visible, retain it as engineering evidence and choose at most one separately declared controller change for later fresh-case qualification. No diagnostic outcome automatically reopens the original screen or changes its thresholds.

The architecture hypothesis remains conditional: retained public action/observation history could identify a persistent actuator change and improve prediction or control. CEM search memory is optimizer state; exact-state/true-gain scoring is supplied physics. Neither establishes that recurrent learned state, fast weights or a biological motif is useful. Before training such a model, a fixed controller must use gain knowledge effectively on fresh cases, and a public finite-window classical identifier must be the comparator. This diagnostic alone qualifies neither recurrence nor novelty.

Evidence read: completed report receipt `95e8e4304394e83487c80d0f6977a26d6a5ec767f7d8d9655e89444ec0e337d3`, with its three report members rehashed; summary `87a3f8a98d970d4936176cbb2344965fdc61a8add4f3b55c2845c99d68e794c1`; earlier candidate diagnosis `e8d45fab596ea26b5dbee0f2b054551a411dec61dfbdc14b82b24651fe0e6618`. This was a saved-results/design review, not a repeat native audit.
