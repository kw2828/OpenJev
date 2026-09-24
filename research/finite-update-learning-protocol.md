# Equal-update recurrent transition comparison

Status: prospective. This protocol defines a new question after the closed
[equal-time comparison](finite-rounded-learning-results.md). That experiment
remains failed with 14/21 conditions. No saved fit is resumed or extended.

## Question and scope

Does rounded transport improve blind four/eight-step decisions when the
three models receive exactly the same prefix and joint training updates?
The earlier equal-time comparison changed the number of accepted updates as
well as the transition parameterization. This experiment removes that
exposure difference. It does not establish equal runtime, FLOPs, gradient
geometry or decision efficiency.

The three arms remain `original_free`, `matched_free` and `rounded`. Keep the
352 stored parameters, four fixed normalization sweeps, slack 1e-8 and final
row/column residual tolerance 1e-12. Original free and rounded share initial
raw transition logits; matched free shares rounded's effective initial
transition within 1e-12. Initial emissions, hazards and readouts are paired.
The existing model and probability construction files remain unchanged.

## Training exposure

The preselected scientific exposure is exactly **1,024 prefix updates, then
3,072 joint updates per fit**. Prefix learning updates only dynamics using
full public-history likelihood and the existing 0.001 log prior. At the
boundary, save the prefix optimizer and model, create fresh joint Adam, and
start the joint batch cursor at zero. Joint learning retains the unchanged
H1/H2 endpoint losses plus the globally normalized prefix likelihood, without
the prior. Adam learning rate is 0.003; gradient clipping is 5; batch size is
64. Orders are paired by seed and epoch. Independently check every accepted
joint batch's attempted indices, eligible rows and event exposures across
all three arms, not only the update totals.

Each fit's 120-second controller cap is a failure limit, not a training eligibility
window. Time starts before validation and model construction. All snapshots,
checks, optimizer work, boundary saves and final summary work are charged.
Every successful fit must complete exactly the requested exposure. Any
exception or cap failure preserves its partial trace and ends the original
attempt. In-flight numerical updates are rolled back; they are never retried.
A failed fit cannot proceed to DEV or be replaced by another seed. The
durable allocation trace is written afterward and included in full fit seconds;
final fit-row publication is included in outer producer time. The outer native
supervisor retains startup, publication and cleanup costs. These measured costs
remain visible even though they are outside the controller's cap.

## Engineering admission

Freeze all sources and this protocol before running qualification. Use one
supervised qualification with lint, the declared tests and one exposure
probe, in that order. The smoke fixture uses namespace 942001, seed 942101,
8 TRAIN / 8 DEV attempts, batch size 3, exactly 3 prefix / 4 joint updates
and a 30-second per-fit cap. Other fabricated profile labels in the tests
are not scientific seeds.

The exposure probe uses namespace 942201, seed 942301, 512 TRAIN / 8 DEV
attempts, batch size 64, exactly 32 prefix / 64 joint updates and a 30-second
per-fit cap. All three arms run once and receive a saved-output audit.
Only measured runtime determines feasibility. For each arm, multiply prefix
stage seconds by 1024/32 and joint stage seconds by 3072/64, double their
sum, then add measured non-stage fitting time. All three projected times
must be below 90 seconds. This deliberately conservative estimate is not a
runtime guarantee or a speed result. It admits the preselected counts; it
cannot change them or use engineering performance metrics to select a model.

A failed test, incomplete probe, failed audit or failed time projection stops
admission. Do not repair a frozen attempt in place. Retain its failure and
source snapshot. Any engineering successor needs a separate identity and a
specific correction, rather than silently repeating the same run.

## Scientific data and order

After original qualification closure, freeze the scientific registration
with identical sources and configuration. Use fresh namespace 433260924 and
fit seeds 433261001, 433261002 and 433261003. Generate 512 TRAIN attempts and
128 DEV attempts. Preserve all attempts, including histories that terminate
early. Required eligible cases remain at least 256 TRAIN and 64 DEV.

Rotate arm order by seed as before. All nine final checkpoints must exist
before DEV generation. Train only on H1/H2 targets; evaluate H1/H2/H4/H8.
Retain the known-dynamics uniform-state reference, exact target witness,
cyclic history shuffle and both blind and observed-filtering predictions.
Learned arms receive public histories; oracle states only support independent
target and reference validation.

## Decision rule and costs

`UPDATE_MATCHED_ADVANCE` has 19 conditions:

- Rounded passes the unchanged all-seed SHORT, BLIND and OBSERVED criteria.
- Against each free control, all six paired H4/H8 regret differences are
  nonpositive, and mean regret is at least 10% lower at both horizons with
  positive control means.

Any failed condition means no advancement. No average, omitted seed or
substituted control rescues a failure. Mean full fitting times and their
ratios against both controls must be reported, but there is no time-ratio
condition in this equal-update rule. A pass cannot overturn the failed
equal-time rule, imply efficient learning or establish statistical
significance. Any efficiency claim needs its own fresh comparison.

## Execution and independent audit

Phases use `scripts/supervise_dialogue_observation_v2.py` and the new
`scripts/finite_update_learning_worker.py`: qualification cap 300 seconds,
producer cap 1,200 seconds and audit cap 600 seconds. The worker requires
single-thread execution, no ambient pytest plugins or bytecode writes,
4 GiB RSS and 512 MiB phase output limits. The original process must close
successfully before the next phase uses its results. Frozen checksums bind
the source, runtime, configuration, output paths and native supervision.

The auditor reads only saved arrays and metadata. It reconstructs targets,
metrics, gates, all work records, exact update counts, common batch exposure,
model/Adam boundaries and actual transition constructions. Retain all 27
model and 27 optimizer boundaries. Rounded diagnostics use independent
four-sweep plus correction reconstruction; free diagnostics use their
actual column softmax. No numerical replay of historical training is
claimed. Historical losses and timings remain source-qualified records.

Publish every arm and seed, runtimes, failures and the complete evidence.
The archived equal-time result and its sources must remain unchanged. A
positive outcome would still require untouched replication, a world without
the known doubly stochastic prior and a second environment. Privileged
starting readouts and the synthetic task remain limitations. This experiment
does not test connectome wiring, RLCD, calibrated text probabilities or a
new architecture.
