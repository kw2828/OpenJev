# Autonomous query-gate comparison

Freeze all six completed trained gates, successful collection/training parent
receipts, sources, runtime, seed review and this protocol before EVAL. No EVAL
feedback changes training, checkpoints, thresholds, cohorts or budgets.

Eight arms: three GRU32 gates, three MLP190 gates (seeds 40101-40103), the
standalone original restricted neural planner and standalone analytic control.
Standalone controls avoid gate/feature work they do not need. Every learned
gate uses the qualified QueryGateActor, with threshold sigmoid(logit) >= 0.05.
Its NumPy state resets per episode and advances only on actual decisions.

Use 24 paired cases at each sensing length 3, 4 and 5: 576 episodes total.
Starts are 17100001, 17200001 and 17300001, respectively, with seed=start+case,
hit=1+case%3 and block=case//3. Rotate arm order by global case index modulo 8.
The native source and observation RNG streams are paired; actual observations
may diverge with different paths. Stop only at discovery or 2,188 moves and
retain all failures, terminal updates and censored tails. No replacement seeds.

Use the established sensing-length-specific initial-hit mixture weights from
the original environment. Report weighted success, capped moves and complete
controller seconds for every arm and setting, all eight blocks, all raw counts,
query counts/ages, and equally weighted family means across the three fit seeds.
Keep length 5 as a separately reported transfer probe; do not pool it into the
primary result at lengths 3 and 4.

Report 62 Boolean criteria, grouped as follows:

1. GRU absolute (24): at each primary setting and each of three fit seeds,
   success >= 95%, moves <= 105% of analytic, moves <= 105% of neural, and
   complete controller seconds <= 50% of neural.
2. MLP absolute (24): the same four checks for each seed/primary setting.
   These diagnose the control and are not required for GRU continuation.
3. Useful recurrence (4): at each primary setting, GRU family success >=
   analytic success and GRU family moves <= 95% of analytic moves.
4. Recurrence comparison (6): at each primary setting, GRU family success >=
   MLP family success, moves <= MLP moves, and complete time <= 90% of MLP time.
5. Working references (4): both standalone controls reach >= 95% success in
   each primary setting.

The primary continuation rule is the conjunction of 38 conditions: groups
1, 3, 4 and 5. Report every individual criterion and group count regardless of
outcome. A failed MLP criterion cannot by itself reject a successful GRU.
Passing this small comparison would motivate confirmation, not establish a
novel algorithm, biological wiring advantage or broad robustness.

Count analytic scoring, feature building, learned gate inference, requested
neural computation, actor construction and belief updates. Keep raw physical
setup and loading totals. Allocate common native/evaluator setup across all
576 episodes, original TensorFlow/model setup across the 504 episodes of seven
neural-consuming arms, and each freshly loaded gate's setup across its own 72
episodes. Do not charge TensorFlow setup to analytic control or reuse the old
192-episode allocation. Include first inference and gate cost. Report deployed
query counts alongside full time, and separately retain shared collection and
per-family training costs. Journal overhead is separated using the same
declared accounting; preserve whole-process time including that overhead.

One original suspend-inclusive supervisor permits 3,600 seconds, one CPU
thread, 4 GiB RSS and 6 GiB output. Maximum 576 resets and 1,260,288 native
steps. No teacher annotation is permitted on skipped queries during EVAL.
No time-cap extension, early success promotion, automatic retry or omission of
slow/failing episodes. Incomplete evidence cannot pass the continuation rule.
