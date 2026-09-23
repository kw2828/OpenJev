# Query-corrected recurrent score prediction

Prospective forecast screen. This is a new score-propagation mechanism, not a
retry of the failed local-advantage labels or a learned query-allocation study.
The prior sparse-query result remains FAIL, 14/16. Freeze sources, qualification,
runtime, teacher weights, cohorts and this protocol before collecting data.

## Mechanism and prior art

Can a small recurrent model update a remembered planner decision after actual
actions and observations, across three steps without a new planner query?
Use the unchanged original OTTO planner as teacher and its four finite action
costs as targets. These targets are teacher predictions, not true action values
or measured search improvement. Its sixteen action-hit branch values are scalar
successor-state values; they do not constitute a four-action plan for the next
state. Analytic scores use a different objective and cannot be subtracted as if
they were in the teacher's units.

Related mechanisms include [Value Prediction Networks](https://arxiv.org/abs/1707.03497),
[Recurrent Kalman Networks](https://proceedings.mlr.press/v97/becker19a.html) and
[PredNet](https://arxiv.org/abs/1605.08104). These motivate prediction and correction,
not a claim that their combination is novel. The exact public posterior already
performs hidden-source inference. This pilot predicts expensive planner outputs;
it is not a learned environment simulator, a new RL algorithm or biological wiring.

The candidate `residual_gru` retains four normalized raw costs and a 29-unit GRU
state. Costs are divided by the fixed power-of-two scale 64. At each period-four
query boundary, copy all four teacher costs exactly, reset hidden state, then
encode the current query features. At each of the next three actual transitions,
feed the current 31 public features and centered previous predicted costs to the
shared GRU, and add its centered four-output increment to score memory. Crucially,
skipped-step annotations supply loss targets only: the recurrent model consumes
its own previous prediction, never a teacher-forced skipped score.

All query outputs retain the original raw float32 teacher selection when later
used in a controller. Blocked directions remain in score memory and targets;
the current legal mask controls prediction loss and action selection. There is
no normalization repair of inherited public beliefs.

Controls receive the same query anchors and available public information:

- `hold`: retain the four raw query costs with the current legal-action mask.
- `current_mlp`: current and query features plus the centered query costs;
  one tanh hidden layer of width 82.
- `history_mlp`: all four query-to-current feature slots, causal padding masks
  and centered query costs; one tanh layer of width 43.
- `direct_gru`: the same GRU29 and output head as the candidate, but every step
  receives the fixed query-cost anchor and predicts an offset from that anchor.

The recurrent models have identical 5,862-parameter layouts and paired initial
weights. The two MLPs have 5,826 and 5,895 parameters. All output heads start at
zero, exactly reproducing hold-Q on valid rows. Inputs outside the valid prefix
are never exposed. Models reset at identical query boundaries, so the history
MLP has the complete intervening history. This cannot establish long-term memory.

## Fresh collection

Collect 54 TRAIN paths: nine paired cases at each sensing length 3 and 4, under
analytic, always-neural and period-four-held-Q control. Seeds are 22100001-22100009
and 22200001-22200009. Then collect 36 VALID paths: six paired cases per setting
under the same three controllers, seeds 22300001-22300006 and 22400001-22400006.
Initial hit is `1 + case % 3`. Rotate the base arm order analytic, neural,
period4_hold left by global case index modulo three. Review these seeds before
freezing. TRAIN has 18 independent cases and VALID has 12, not 54 and 36.

Each path uses the qualified 53-by-53 public filter and original float32 teacher,
with eight-way symmetry averaging and original in-bounds tie selection. Preserve
all paths to discovery or 2,188 moves, including the final update. Pair native
source and indexed observation uniforms across the three arms in each case.

Make exactly one physical teacher call per visited preaction state. Reuse a
deployed call as its label. At analytic and held-Q skipped decisions, fix the
action before the annotation; the annotation must not modify cache or action.
Only actual period-four deployed queries update held-Q control memory.

Distinguish physical `deployed_query`, `annotation_only`, and virtual
`correction_scheduled`. Forecast windows always start at steps 0,4,8,... under
every collector. The unchanged feature builder receives virtual last-query
index `step - step%4`: feature 16 is `step%4 / 2188`, and feature 17 is one,
including at a query row whose teacher anchor is supplied to the model. This
is a deliberate forecast convention, not the earlier gate's prequery age.
Actual previous actions and hits remain public inputs. No teacher annotation
is encoded in features.

Save separate TRAIN/VALID arrays and complete per-step witnesses. Every disjoint
window has one to four rows; retain short tails and query-only windows. Never
fabricate future transitions. No row or originating case can cross the split.

## Training and forecast evaluation

Fit every learned family with seeds 225001,225002,225003. Use CPU float32,
80 epochs, Adam learning rate 0.003, batches of 32 windows, gradient clipping at
norm 5, no weight decay, and the final checkpoint only. Shuffle each epoch with
the same seed-derived permutation across families. Every window and tail is
exposed once per epoch. No early stopping, checkpoint selection, tuning or
winning-seed promotion. Save all twelve final models before reading VALID arrays.

Train mean squared error after centering each prediction and target over that
row's legal actions, on every legal action at every available nonquery row, in
units divided by 64. Each TRAIN episode has total mass 1/54,
divided equally among its nonquery rows. An episode without such rows contributes
zero and stays in the coverage denominator. A minibatch sum uses the fixed row
weights, rounded to float32 for training, times `total_windows / batch_windows`;
report the actual weighted loss and exposure counts. Every minibatch takes one
Adam step. An exact zero connection to every parameter supplies zero gradients
for query-only minibatches, preserving Adam's ordinary momentum evolution and
the full declared exposure. VALID metrics use scalar float64 episode means.
Targets are never inputs to skipped predictions.

Score all final fits and hold-Q once on complete VALID trajectories. This is
forced-path forecasting, not autonomous policy evaluation: trajectories and
future cheap observations were generated by the declared collectors. Report
every regime, collector, fit and query age. Later ages condition on a path
surviving long enough to supply that observation; their support is explicit.

Primary metrics are agreement with the teacher's eligible near-minimum action
set and raw teacher-score gap. Eligibility and near ties use original float32
subtraction and strict `abs(q-min) < 1e-10`. Gap is the nonnegative float64
difference of the original float32 teacher costs, with no rounding or clipping.
Prediction ties select the first eligible near-minimum in original order.
Centered score MSE is secondary. Within each reported setting/age, give every
originating episode equal total mass and divide it among its applicable rows.
Zero-support episodes keep zero contribution; show their count and the effective
weight mass. Identical choices, zero gaps and all tails remain included.

All 45 continuation conditions are required:

1. Complete original collection, all twelve fixed fits, saved predictions and
   independently checked metrics/lineage within the original limits (one).
2. At least four distinct VALID originating cases support each of the three
   nonquery ages in each setting (six).
3. For every candidate fit and setting, agreement is at least hold-Q agreement
   and mean score gap is at most 80% of hold-Q gap (twelve).
4. For every candidate fit, setting and age 1,2,3, score gap is no worse than
   hold-Q (eighteen).
5. Averaged over the three fixed fit seeds in each setting, candidate agreement
   is no worse than both history-MLP and direct-GRU, and candidate gap is at most
   90% of each respective control's gap (eight).

A zero comparison gap requires a zero candidate gap, without epsilon ratios.
These are practical screen margins, not significance or confidence guarantees.
Three fit seeds do not create independent VALID cases. A pass permits proposing
a separately frozen fresh autonomous comparison under the existing competence,
move-quality and fully paid compute requirements. It does not itself establish
useful control, an advantage over analytic behavior, a world model or novelty.
A failure retains every fit and closes this training recipe under this allocation.

## Resources and evidence

One original collection supervisor: 900 seconds, CPU1, 4 GiB RSS, 2 GiB output;
at most 90 resets and 196,920 moves/teacher calls. Deployed calls are at most
82,050 and annotation-only calls at most 114,870; their sum equals physical
teacher calls. All 49,230 possible virtual corrections are accounting events,
not additional inference. Returns remain provisional until the complete episode
journals and row are durably closed. Preserve attempted calls, pending actions
and partial files on failure. No resume, replacement cases or time extension.

One original training/forecast supervisor: 600 seconds, CPU1, 4 GiB RSS, 2 GiB
output. No simulator or teacher calls. One separate saved-output audit allocation:
120 seconds, CPU1, 2 GiB RSS, 128 MiB output. Preserve collection setup, annotation,
fitting and scoring costs; no operational compute-saving claim follows from a
forecast loss. Freeze and qualify source changes before their execution, keep
exclusive output directories, and publish failures as well as successful checks.
