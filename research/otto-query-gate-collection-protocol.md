# Fixed query-gate TRAIN collection

This protocol operationalizes the prospective [learning design](otto-query-gate-learning-design.md).
Freeze sources, runtime, seed reservation and completed native-qualification
receipts before collection. No training or evaluation occurs in this phase.

Collect 60 trajectories: two sensing lengths (3 and 4), six paired cases each,
and five scheduled policies. Initial hit is `1 + case % 3`; TRAIN seeds are
18100001-18100006 and 18200001-18200006 respectively. Use the existing native
seeded environment, 53 by 53 grid, four hit values and the original sensor model.
Rotate the five schedules by `(regime_index * 6 + case) % 5`:

- always query;
- never query;
- query when pre-action step modulo 2 is zero;
- query when pre-action step modulo 8 is zero;
- query only at pre-action step zero.

Each schedule controls the actual action and therefore its actual trajectory.
Stop only at discovery or after 2,188 steps. Retain terminal/censored updates,
all real pre-action rows, and all failed attempts. No quotas, replacement cases,
loop filters, success selection or truncated-prefix substitution.

Use the unchanged qualified original TensorFlow planner and QueryGateActor.
Exactly one original neural annotation is obtained at every decision. A
scheduled query supplies the annotation through the gate's backend. If skipped,
first commit the analytic choice, then compute a separate score-only annotation
before stepping the environment. Annotation cannot change the selected action,
query age, has-queried flag, carried gate state or public belief. The gate sees
only its declared 31 public features, computed before the annotation.

The binary target is whether the analytic action is outside the eligible
float32 near-minimum set under the original strict `abs(cost - min) < 1e-10`
arithmetic. It is planner disagreement, not measured continuation regret.
Store the continuous planner cost gap separately. Preserve public/native belief
byte equality after resets and all updates; no secondary reference forward is
needed after the completed integration qualification.

`training-data.npz` contains exactly: features float32[N,31], labels bool[N],
neural_costs float32[N,4], analytic_action int64[N], masks bool[N,4],
episode_offsets int64[61], scheduled_query bool[N], neural_gap float32[N].
Rows follow recorded episode order. Record every episode's seed, regime, hit,
schedule, actual steps, annotation count and deployed-query count, plus complete
public action/observation journals and belief witnesses. Record query-age
histograms descriptively; never alter the cohort based on these statistics.

One original suspend-inclusive supervisor permits 600 seconds, one CPU thread,
4 GiB RSS and 2 GiB output. Caps: 60 native resets, 131,280 native steps and
131,280 actual TensorFlow policy forwards. Include setup, filtering, features,
both score backends, all annotations and evidence in the physical runtime.
Original setup allocation metadata is historical; retain new physical costs.
Fitting receives a separate 300-second allocation, so the two caps sum to the
proposed 900 seconds. No automatic extension or scientific retry.

Completion requires all 60 paths and exact declared data/receipt inventory,
closed operation journals, unchanged frozen inputs/sources and successful
original parent closure. Incomplete collection cannot admit fitting. This
establishes a TRAIN cohort, not learned competence or architecture efficacy.
