# Paired value-of-query signal pilot V2

Prospective, 22 September 2026. Freeze this protocol, execution and audit sources,
qualification receipts, source/runtime/input hashes, seed review and resource
limits before collecting any new trajectories. This tests a training signal, not
a learned controller. No optimizer, learned-gate fitting, VALID or EVAL work is
authorized by this pilot. Earlier experiments and their failed rules remain
unchanged.

## New allocation after an engineering failure

V1 remains incomplete at its original 900-second limit. Its partial outcomes are
not inputs to this protocol, its label panel order is not resumed, and its
scientific rule was not evaluated. This is a prospective fresh cohort with the
same scientific recipe and criteria, after a separately measured implementation
change. Native paths, label draws and bootstrap streams are all new.

Before this allocation may be frozen, the [durable logging qualification](otto-query-logging-qualification-protocol.md)
must finish under its original supervisor, pass its failure contracts and exact
byte checks, achieve at least 2.0 median paired speedup, and improve five of six
blocks. The scientific plan binds that original qualification and its terminal.
The new logger still flushes and fsyncs every record before acknowledgment. It
encodes once, keeps streams open and accounts bytes incrementally, with explicit
reconciliation after external publications, episodes, panels and final close.
The combined setup reservation remains 1 MiB, split equally over the two small
setup artifacts. Failed writes remain uncertain and cannot be resumed.

The corrected audit qualifies the exact absolute producer script invocation.
Original V1 sources and records remain unchanged. A fabricated logging speedup
does not guarantee that this new cohort finishes. The 900-second limit is a new
prospective feasibility cap, not a measured completion bound. Any V2 miss closes
this allocation with no continuation, replacement or automatic budget increase.

## Question and fixed collection

Does the benefit of one neural-selected action, followed by analytic control,
vary repeatably across public states? In particular, does such variation remain
among states where the neural and analytic actions differ? The earlier
disagreement target and fixed 0.05 threshold produced only queries.

Collect 72 full TRAIN paths: sensing length 3 (`base`) and 4 (`shift`), 12 cases
per setting and three physical schedules (`always`, `never`, `period2`). Native
seeds are 19500001-19500012 and 19600001-19600012. Initial hit is
`1 + case_index % 3`. Rotate the three schedules left by global case index modulo
three. Use the existing 53-by-53 environment, original qualified TensorFlow
planner with eight-way symmetry averaging and restricted in-bounds action rule,
and the unchanged analytic endpoint and public filter. Actual paths end only at
source discovery or 2,188 moves, including the final public update. Keep every
path, failure and capped tail. No replacement seeds or shortened paths.

Predeclare five possible anchors per episode at pre-action steps 0, 4, 8, 16 and
32. Anchor ID is `episode_index * 5 + prefix_slot`, leaving gaps where a path
already found its source. Record all 360 slots as available or unavailable due
to discovery; missing anchors are not replaced. Every episode supplies its
initial anchor. Save the complete preceding public history, exact already
assimilated float64 belief, public packet, 31 features, endpoint actions and
score witnesses. Snapshot validation must accept every available anchor before
any label generation. Unsupported beliefs fail the attempt without repair.

Physical query schedules determine the actual chosen action and query age.
Reuse a deployed neural query when it occurs at an anchor. On a skipped anchor,
choose the analytic action first, then obtain one separate score-only neural
annotation. Verify that it cannot change the actor state, pending action,
belief, public history or query age. Do not annotate other skipped states.
All 72 paths and their saved anchor arrays close before the first label source
draw. Hidden native truth remains in a separate evaluator journal and cannot
enter anchor construction, endpoint features or counterfactual initialization.

## Paired intervention labels

At each available anchor, use replicate IDs 0-15 and sampling seed 19700001.
Preserve the qualified sampler's RNG namespace, source/hit channel identities,
categorical arithmetic and analytic continuation. The additive pair adapter
changes only the first-action set. Qualify it against the unchanged all-action
sampler on fabricated fixtures in the actual native runtime before collection.

For each replicate, draw one source from the anchor's public posterior. Force
the analytic action in one branch and the neural action in the other, then use
the same analytic policy. The horizon is **32 total moves, including the forced
first move**. Restart the same observation RNG stream for both branches, using
each branch's source-conditioned observation law. Observations need not match
when positions differ. No odor draw follows discovery. Assimilate the final
packet, including discovery on move 32 or a nonterminal move 32.

If both endpoint actions are identical, execute one real continuation per
replicate and reference it for both roles. This is a structural zero advantage,
not an uncertain estimated winner. Otherwise execute both distinct actions in
ascending action order. Preserve all records, draws, attempted/returned work,
censored ties, discovered-at-cap outcomes and negative advantages.

For each paired replicate define `D = moves_analytic - moves_neural`. Report
the 16 differences, their mean, sample variance, standard error and each branch's
discovery, censoring and at-cap rates. These are local capped benefits under a
declared analytic continuation. They are not full-episode regret, optimal action
values or benefits under repeated gated neural queries. They exclude the price
of the neural call; actual generation and controller costs remain reported.

## Fixed signal analysis and continuation rule

Split replicates once into IDs 0-7 and 8-15. For anchor i let x_i and y_i be the
two half means. Within each setting, give each of its 36 episodes equal total
weight, divided equally among that episode's available anchors. Report both
all-anchor and different-action groups; the latter inherits these weights and
renormalizes, without selecting on returns.

For each group compute centered cross-half covariance
`C = sum(w_i * (x_i - x_bar) * (y_i - y_bar))`, pooled centered half variance
`V = 0.5 * sum(w_i * ((x_i - x_bar)^2 + (y_i - y_bar)^2))`, and descriptive
repeatability `C/V`. Keep negative estimates. Undefined ratios stay undefined;
do not clip, smooth or replace them with evidence of signal. Also retain
uncentered moments, noise estimates, sign/tie counts, per-case contributions and
censoring. Center within settings so a setting offset cannot supply apparent
state discrimination.

The saved-only independent analysis fixes 2,000 stratified cluster-bootstrap
draws with standard-library `random.Random(19800001)`. In every draw, process
`base` then `shift`; sample 12 originating case indices per setting using
`randrange(12)`, with replacement. Carry each sampled case's three schedules and
their complete anchors together. Case multiplicity determines original weights;
renormalize only within the declared different-action subgroup. A resample with
no different-action anchors receives C=0 for that group and is counted, not
dropped. Average the two setting covariances equally. The reported one-sided
90% percentile lower value is exactly the 200th sorted value, index 199, of
2,000 draws. Preserve every sampled case list and statistic.

All **11** conditions must pass to admit a separately frozen matched fitting
study:

1. Technical completion, original process closure and independent saved-record
   agreement (one condition).
2. At least 12 different-action anchors and six originating cases containing
   them in each setting (four conditions).
3. Positive C for both all-anchor and different-action groups in each setting
   (four conditions).
4. Positive pooled bootstrap lower values for both groups (two conditions).

This is an exploratory screening rule, not a guarantee of confidence coverage,
label accuracy, controller efficacy or novelty. A pass admits only the matched
target/architecture study. A miss closes this label recipe under this allocation;
do not automatically enlarge replicates, change horizons, tune thresholds or
substitute anchors. Earlier exposed evaluation cases remain development evidence
and cannot serve as untouched confirmation for any later controller.

## Resources and evidence

One run, one numerical thread, unchanged `.venv-otto-released-native` interpreter:
900 seconds including authentication, native collection, all annotations, label
generation, reductions, I/O and closure; 4 GiB peak RSS; 6 GiB output. The
suspend-inclusive supervisor is the original authority for time and process
cleanup. No runtime installation or mid-run budget extension.

Maximum work is 72 native resets, 157,536 actual native steps, at most one neural
score per visited state, 360 anchors, 5,760 posterior source draws, 11,520 distinct
action continuations and 368,640 continuation moves. Count identical endpoints
once physically. The pair adapter emits at most 3,019,680 events, each bounded
by 1,024 bytes. Bound all other artifacts separately within 2 GiB, leaving the
full envelope for those records and the final receipt. These are hard ceilings,
not throughput promises. Record raw phase times and overlapping operation times
separately, with setup, filtering, feature, query, annotation and logging costs.

Use exclusive outputs, bound complete sources and input manifests, preserve
pending operations and partial artifacts on failure, and join the original
supervisor terminal before claiming completion. Independently audit saved
identities, all pair coverage, source/observation stream pairing, action roles,
public update chronology, arithmetic, weights and cost counts. Freeze that
analysis source before collection. Its separate limit is 120 seconds, 2 GiB
RSS and 128 MiB output; it makes no new native, neural, sampler or optimizer calls.
Inherited numerical filter/planner truth remains qualified producer evidence
unless explicitly replayed. Publish all signal results and limitations, including
a failed continuation rule.
