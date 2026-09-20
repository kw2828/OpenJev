# Conditional cost coverage over the closed training ledger

Status: prospective design only. The shared-state-columns sixteen-cell timing
screen is frozen but remains unlaunched while another CPU training job is
active. This proposal becomes eligible only if that screen completes and passes
every unchanged requirement. It neither replaces that screen nor authorizes
timing, fixture extraction, new training, a retry or a cap extension.

Earlier training and implementation qualifications remain closed. The purpose
of a later check would be to cover more of the recorded public workload before
considering another training attempt. Passing artificial update measurements
would still not establish that twelve full fits finish within a study cap.

## Available evidence and the missing prerequisite

The closed `slot_readout-4101` fit has 1,280 ledger rows: twenty epochs of 64
updates, comprising 1,260 batches of 32 dialogues and twenty final batches of
one dialogue. The aligned training cohort has 2,017 dialogues. Its ledger hash
was checked against its closed fit receipt. Only indices, epoch/update identity,
actor shapes and observation-work fields were used here; losses, predictions
and fit-quality outcomes were not analyzed.

Paths below are relative to the repository. These are existing local inputs,
not a claim that a new representative fixture has been prepared.

| Existing path | Permitted role in a future metadata-only preparation |
|---|---|
| `runs/dialogue-token-v1/study-01/fits/slot_readout-4101/completed.json` | Closed fit identity and ledger binding, not successful completion of the enclosing timed-out study |
| `runs/dialogue-token-v1/study-01/fits/slot_readout-4101/batches.jsonl` | All 1,280 ordered training batches, dialogue indices and recorded logical work |
| `runs/dialogue-copy-v1/lexical-01/index.json` | Training dialogue order, public query IDs and `[turns, queries, maximum_candidates, 10]` layouts |
| `runs/dialogue-token-v1/features-01/index.json` | Matching training dialogue order and public-context index ranges |
| `runs/dialogue-token-v1/features-01/context-indices.npy` | Training context references, using only the training ranges |
| `runs/dialogue-token-v1/features-01/offsets.npy` | Referenced token lengths from adjacent offsets |
| `runs/sgd-state-v1/features-02/packet.json` | Missing per-query candidate cardinalities, projected only for training query IDs from the supplied candidate table |
| `output/dialogue-token-packing-v1/geometry-01.json` | Already exported exact geometry for update 1,237 only |

The receipt and ledger SHA256 values are respectively
`9c9bb4ca7a6bf2d197fb540267315007bfedf4ff63d1847c6d4ab6a0ecf98665`
and `017b24dd01d1366a53bb6d5e1bfd606fd0f7097b0eb8f4a2b0c6b01c17785213`.
The existing single-geometry fixture has SHA256
`286e209b2ed22a98a51894c0e1438ac084de7c28a3c2f5771f8d6dc4521bcc49`.
Its provenance records the other parent receipts and payload hashes, including
packet `14dd89a633bfa406394945d279c44ce6796d3b8dc412f772e2abda794ec943cb`.
A future preparation must authenticate those exact bindings again.

The aggregate ledger alone cannot reconstruct per-dialogue candidate support
or token-length distribution. The lexical layout supplies maximum candidate
width, not each query's cardinality. Therefore an exact all-training metadata
export is a missing prerequisite. It must retain only per-dialogue ordered
token lengths and candidate counts, plus source/row provenance. Reconstruct
prefix token masks, real-turn masks and candidate masks using the original
assembler, including one supported NONE candidate for each padded dummy query.
Require its derived actor shapes and all applicable observation-work fields to
match every ledger row exactly before selecting or timing anything.

This export needs no token states, sentence embeddings, lexical values,
trained weights, predictions, encoder or new corpus fetch. The packet is a
mixed-content file containing more than cardinalities; the future reader must
project an explicit metadata whitelist, not call the full study loader or
consume text, target labels, transition bins or quality outputs. Do not inspect
development/test cohorts or use their contents to select fixtures. No such
export was executed for this note.

## Deterministic coverage, not quality selection

Propose seven non-overlapping frequency strata covering all 1,280 rows. Put
the twenty one-dialogue batches in one stratum. Sort the 1,260 full batches by
recorded `pooling_score_positions`, breaking ties by update index, then divide
them into three contiguous groups of 420. Within each group, sort by exact
`valid_token_positions / padded_token_positions`, again breaking ties by
update index, and split into two groups of 210. Use rational comparisons for
the density sort. Timing, losses and fit outcomes must not enter any ordering.

Within each stratum, sort by score work and update index and select its lower
median row. Those seven rows represent weights 20 and six times 210. Record
every row-to-stratum assignment and the selected row identity, not just the
seven geometries. These choices provide deterministic coverage of attention
work and token padding; they are not random sampling or an unbiased estimator.

Add at most five tail guards: the earliest maximum of dense attention score
positions, public attention-group count, dense head positions, supported head
positions including dummy NONE, and sequential turn count. Derive the last
four from the validated metadata with the frozen public layout formulas.
Deduplicate by selected ledger update, never by observed timing. Tail guards
carry zero additional frequency weight because their population rows are
already assigned to the seven strata. Report their coverage separately. The
known maximum-score fixture should reappear at update 1,237 and must match the
existing export exactly. The final selected list must be frozen before values
are generated or any model is constructed.

The result contains seven to twelve geometries, each exercised in all four
slot/candidate and readout/scalar arms. These masks come from one closed
seed's batch ordering and one training cohort. They do not represent all
possible seeds, tasks, devices or deployment requests.

## A later bounded synthetic check

After the prerequisite pass and metadata validation, use one new prospective
protocol and exclusive directory. Retain the original implementation as the
comparator and freeze the exact passing shared-columns source. Generate only
artificial values, priors and labels with independently declared engineering
streams. Retain actual mask geometry but do not use actual training values or
labels. State the synthetic supervision rule explicitly because loss sparsity
and input values can affect cost; do not claim exact training replay.

Keep float32 CPU execution, thread settings, optimizer, numerical tolerances,
all input/parameter gradient membership and actual monitor checks from the
sixteen-cell screen. For each arm/geometry, require parity, one warm update per
path and four alternating measured pairs. Restore the same original post-warm
model and optimizer state before each measured update. At most 48 cells imply
480 optimizer updates, 96 parity forward/backward passes and 576 operation
records. Freeze the exact count after metadata selection; incomplete coverage
cannot pass.

Measure complete update time, including assembly, validation, all mask/layout
and work-counter computation, forward/loss, backward/clipping, optimizer and
scalar readback. Report those broad phases separately. Charge construction,
initialization, synthetic-value preparation, warmup, state restoration and
hashing, external counter comparison and receipt serialization to whole-command
time, while reporting them separately from measured updates. No cost may
silently disappear between these scopes. Defer execution while a known heavy
competing job is active; record runtime and host-contention limitations without
claiming control over every source of scheduling variation.

Proposed engineering requirements retain the per-cell 0.90 original/new ratio
floor for one-dialogue batches and 1.10 for full batches, numerical agreement,
6 GiB process-lifetime peak RSS and one 300-second whole-command cap. Also report
for each arm the ratio of stratum-size-weighted sums of median original and new
update durations. Do not average cell speed ratios or give tail guards duplicate
population weight. Publish all pairs, stratum/tail coverage, logical work and
failed conditions. No selective retiming or threshold adjustment.

A normalized stratum-weighted mean duration, multiplied by 1,280, gives only a
synthetic geometry-based projection under that run's host conditions. It omits genuine
twenty-epoch parameter/optimizer evolution, actual supervision and values,
evaluation, checkpoint I/O and any unmeasured contention. It is not a guaranteed
study-wall forecast or recovery of the closed 3,600-second attempt. Even a pass
would require a separate prospective end-to-end budget protocol before any
new scientific launch.
