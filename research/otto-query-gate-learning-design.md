# Learn query allocation without confusing annotations with queries

**PROPOSED, not frozen, admitted or executed.** This is one next comparison after the [native integration qualification](otto-query-gate-native-protocol.md), which establishes mechanics only. Freeze the following recipe, implementation, seed review and budgets before collecting TRAIN data. The intervention learns when to use the unchanged restricted OTTO neural planner; it does not learn a new planner or establish architecture novelty.

## One small cohort with consistent query histories

Collect **60 complete TRAIN trajectories**: sensing lengths 3 and 4, six fresh paired cases per setting, and five fixed virtual deployment schedules per case. Initial hit is `1 + case % 3`, giving two cases per hit. Rotate schedule order by `(regime_index * 6 + case) % 5`. At pre-action step `t`, use:

| Schedule | Query decision |
|---|---|
| Always | Every step |
| Never | No steps |
| Period two | `t % 2 == 0` |
| Period eight | `t % 8 == 0` |
| Initial only | `t == 0` |

The scheduled query determines the **actual** selected endpoint and action. Thus each observation sequence is consistent with its declared query history. Always/never provide the endpoints; the periodic schedules cover intermediate ages, and initial-only supplies long ages with `has_queried=true`. This replaces the earlier two-endpoint-only collection proposal rather than adding another cohort.

Obtain exactly one original neural score annotation at every visited decision, but keep it outside deployment query accounting. On a scheduled query, the genuine planner result also supplies that state's annotation. On a skipped query, finish the analytic choice first, then obtain a separate score-only annotation before the native step. Annotation never changes the committed action, gate state, `has_queried`, or `last_query`. Record `scheduled_query` and `annotation_call` separately. Build features before annotation; the gate receives no current neural score.

Retain every pre-action row, every actual action and observation, and the terminal or capped final update through **H=2,188**. No success filtering, loop stopping, state quotas or replacement cases. Maximum collection is 60 resets and 131,280 native steps/neural annotations. An interrupted trajectory makes collection incomplete; do not train on its surviving prefix as a substitute.

Candidate TRAIN seeds are 18100001–18100006 and 18200001–18200006. A source/protocol search found no declarations for these or the candidate EVAL/fit seeds below; this is not ledger-level freshness evidence. A scoped review of actual known seed ledgers and frozen reservations remains required before admission.

## A matched learning comparison

The binary target is whether the analytic action lies outside the neural planner's **eligible float32 near-minimum set**, using its existing strict `abs(score - minimum) < 1e-10` arithmetic. Preserve the continuous planner cost gap as a diagnostic. This is disagreement with a fixed score function, not observed continuation regret or proof that querying improves the environment outcome.

Train a **GRU32 plus scalar output** and a **31→190→1 Tanh MLP**, using exactly the helper's 31 public features. With ordinary two-bias GRUCell semantics these have **6,273 and 6,271 parameters**, respectively. The MLP receives the same explicit age, prior-action and change features; only the GRU carries additional learned state. Three paired fit seeds are proposed: 40101–40103. Initialize scalar output weights to zero and bias to `log(19)`, so both start with query probability 0.95. Fix query threshold **0.05**, including equality, with no held-out adjustment.

Use all collected rows, binary cross-entropy, 80 epochs, Adam 0.0003 and gradient norm clipping at five. If episode `e` has `T_e` rows and the cohort has `N` rows, assign every row weight `N/(60*T_e)`. The reported objective is the weighted loss sum divided by `N`, so episodes have equal weight and long tails do not dominate. No class balancing or cost-gap filtering.

Initialize one `default_rng(seed + 20000)` per fit and draw successive epoch permutations, identically across families. Batch eight episodes, process chronological 32-step windows, carry and detach GRU state between windows, and reset only at episode boundaries. Pad short windows; retain every real tail row. Each update divides the weighted real-row loss sum by the fixed 8×32 slots; padding contributes zero. Both families receive the same windows, weights, update counts and final-checkpoint rule. No early stopping or checkpoint selection.

The five schedules prevent the specific error of treating every annotation as a recent query. **They cannot guarantee the learned gate's state distribution.** Its choices can create new query histories and new physical trajectories. Record query-age histograms and maximum ages on TRAIN and EVAL descriptively, without adding post-hoc thresholds or dropping rows.

## One fresh autonomous test and complete costs

Evaluate all six final gates plus standalone always-neural and always-analytic controls on **24 paired cases per setting**, lengths 3/4 primary and length 5 a declared transfer probe: **576 episodes, 72 cases**. Candidate starts are 17100001, 17200001 and 17300001; hit is `1 + case % 3`, block is `case // 3`, and eight-arm order rotates by case index. Reuse source and hit stream identities across arms, not observations after their paths diverge. No EVAL paths enter training.

For each learned seed and primary setting, require weighted success at least 95%, capped moves no more than 105% of **each** fixed control, and complete controller seconds at most 50% of always-neural. Report these absolute checks for both families. A useful recurrent result must also preserve analytic success while reducing its family mean moves by at least 5%; a recurrence advantage additionally requires no worse family success/moves and at least 10% lower complete cost than the matched MLP in both primary settings. Report all eight paired blocks and all transfer cells; primary success cannot establish transfer. These are prospective admission criteria, not promises based on classification accuracy.

Propose **900 seconds for collection plus fitting** and **3,600 seconds for autonomous evaluation**, one CPU thread and 4 GiB RSS; output caps 2 GiB and 6 GiB, respectively. Qualify runtime before the full frozen plan. These are hard caps, not guarantees that the worst-case tails fit. Failure preserves the attempt without sampling fewer rows or extending the budget.

Charge every annotation, analytic score, feature build, gate decision, neural query, filter update, checkpoint load and framework setup. Pay the shared TRAIN cohort once, report per-family allocations separately, and retain raw physical totals. Evaluation uses fresh gate loads and includes first inference; standalone controls avoid unnecessary gate overhead. Report query counts alongside full controller time and amortized training costs. Successful disagreement prediction alone cannot establish competence, savings or the value of recurrence.
