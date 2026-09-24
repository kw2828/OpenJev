# Frozen-feature residual estimators: prospective execution protocol

**Not yet admitted.** This protocol implements the
[mechanism-screen design](otto-residual-estimator-design.md). It does not change
the failed previous experiment or permit its unused TEST data to be read.
Registration, qualified sources and original process closures are required in
addition to this document.

## Fixed comparison

The sole candidate is unattenuated full-covariance RLS. Use the nine methods,
four-ratio grid, numerical convention, minimax selection, 13 usefulness conditions
and separate mechanism contrasts in the design without modification. There is
no fitting, learned gate, process noise, forgetting, seed selection or adaptive
teacher schedule in this screen. The first-query exclusion and P4 schedule apply
to every method.

Reuse all nine original final checkpoint files: `pretrained`, `joint_aux` and
`trace_delta`, each for seeds 309000001, 309000002 and 309000003. The trace model
supplies only its trained projection; its slow weights must equal the same-seed
pretrained weights. Bind these bytes and their pre-DEV lineage to the original
training plan SHA256
`9ea6b7422116ca3a708e93d2339f4209ddad8a695feac9d31c590f758aac140a`.
The old producer and independent audit completed technically, with a scientific
**DEV FAIL 6/13**. Neither that failure nor any checkpoint identity is rewritten.

## Cohort and phases

Freeze one master roster of 54 paths before collection. Development uses
lambda3 seeds 314000001-314000003 and lambda4 seeds 315000001-315000003.
Confirmation uses lambda3 seeds 316000001-316000006 and lambda4 seeds
317000001-317000006. Each originating case has analytic, neural and period-four
held-teacher paths. Rotate collector order by global case index modulo three;
use `initial_hit = 1 + local_case % 3`. Development episode indices are 0-17 and
confirmation indices are 18-53. The
[seed review](../output/otto-residual-estimator-v1/seed-review-01.json) preserves
both reservation attempts and resolves the replacement scan's non-seed matches.

The first implementation admits **development only**, with 18 complete paths
and 72 evaluation views. It explicitly rejects confirmation execution. After a
development pass, a separately qualified and registered confirmation runner may
use the already reserved 36 paths and 27 views at the frozen selected ratio.
No replacement cases, continuation after an interrupted collection, shortened
episode, resampling or cap extension is allowed.

Each path has at most 2,188 steps. Collect one teacher answer at every retained
state, including annotation-only states. Annotation never changes a collector's
action or held teacher cache. Save the full census, legal masks and actual
actions. Only actual scheduled answers reach the frozen model and estimator;
other teacher answers remain evaluator targets.

Confirmation collection itself waits for the completed development decision and
both original successful producer/auditor closures: array serialization is
verified by reloading it, so collection would otherwise decode held-out data.
External confirmation artifacts use `confirm`. Any later compatibility mapping
to the existing metrics label `test` must retain the new master identities and
must never resolve a path or roster from the old study.

## Extraction and evaluation

Authenticate sources, runtime, checkpoint lineage, complete collector receipt
and original collector supervisor before any numerical import or array decode.
Project the development census once at P4. Extract one cache per fit seed using
batch size one and chronological 32-step chunks, retaining the complete episode
between chunks. Each cache charges both the pretrained and joint-model forwards.
There are six supplied-model constructions and three complete cache builds.

Evaluate all 24 method/ratio settings for each of the three seeds, preserving
all 72 outputs. Each replay is executed independently, including the three
strict attenuation controls; posterior reuse is not silently counted as free.
Save action scores, prewrite correction, uncertainty values, selected operation
counts, array state sizes and the shared offsets. Save all three caches and
their exact metadata. Serialize arrays with explicit types and no pickle, then
verify their byte-for-byte roundtrip.

Compute all existing full, initial, later, age, collector and originating-case
action summaries. Use action-centered MSE as a secondary metric. Do not attach
the pretrained shadow prior to joint-model results; no common prequery-MSE
comparison is reported. Candidate selection and decision-gap conditions use the
unchanged floating-point arithmetic in the qualified gate.

The producer records a provisional decision with technical completion false.
A separate process audits saved outputs after the original producer closes
successfully. It makes no neural-model, optimizer, teacher or simulator calls.
It reconstructs each estimator replay from the authenticated saved cache and
checks exact outputs/counts, then independently computes scalar action metrics,
aggregation, ratio selection and the complete decision rule. Replaying an
estimator is real audit computation and remains in the measured audit cost.
Recurrent hidden-state causality and cache correctness remain source-tested
claims; saved arrays alone cannot independently prove them.

The audit's decision remains provisional until its own original supervisor
closes successfully. A final closure record joins both original processes,
source/input hashes and the audited decision. Any technical or scientific
failure closes development and leaves confirmation unused. Controls, ratios and
individual fit seeds cannot be promoted after seeing the result.

## Fixed resource bounds

| Phase | Wall-time cap | Peak RSS cap | Output cap |
| --- | ---: | ---: | ---: |
| Development collection | 3,600 seconds | 4 GiB | 1 GiB |
| Development extraction/evaluation | 900 seconds | 4 GiB | 1 GiB |
| Independent development audit | 900 seconds | 4 GiB | 256 MiB |

Collection caps are 18 resets and 39,384 teacher/native steps. These are bounds,
not a target episode length or a promise of success. Use the original
suspend-aware supervisor and its native continuous clock, one numerical CPU
thread, exclusive output paths, complete failure receipts and process-group
cleanup. Authentication, serialization, gate computation, array verification
and audit are all paid work. Confirmation needs its own phase admission; these
bounds do not authorize it.

Worker callbacks check the native deadline on every call. Peak-RSS and output
scans run at a fixed 250-millisecond interval and at explicit IO boundaries;
the supervisor independently enforces the original deadline. The capacity probe
uses the same callback, including its native clock and resource scans.

Before empirical collection, qualify the integrated numerical path on a
fabricated worst-horizon development roster with synthetic model weights.
Measure one fit's cache, all 24 views and their independent audits, then use
`2 * 3 * measured_seconds + 120` as the projected time for each 900-second
evaluation/audit phase. Require each estimate at most 675 seconds. This fixed
twofold margin and 120-second overhead are a capacity heuristic, not a runtime
guarantee or a performance result. Capacity failure does not permit increasing
the empirical caps. Source or implementation repairs require a new recorded
engineering attempt before admission.

## Interpretation

Passing development is a selection result, not held-out evidence. Only a
confirmation usefulness pass permits a separately registered autonomous
utility-versus-total-compute experiment. Full covariance earns a mechanism claim
only if its own confirmation contrast passes too. A tie or advantage for weaker
reads must remain visible. This screen does not establish a new learning rule,
connectome advantage, calibrated action probabilities, RLCD, conformal coverage,
world model or ICLR-level novelty.
