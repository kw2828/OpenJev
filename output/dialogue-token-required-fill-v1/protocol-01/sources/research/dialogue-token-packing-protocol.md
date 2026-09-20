# Mask-packed pooling: prospective implementation qualification

The preceding [time-batched pooling qualification](dialogue-token-batching-results.md)
made progress by rejecting an insufficient optimization. Its measured failure
and the earlier 3,600-second scientific timeout remain unchanged. This is a
new implementation hypothesis, not a retry, resumption, cap extension or
relaxation of those attempts.

## Question

Does removing padded token and candidate work improve complete training-step
cost while preserving the existing model? Follow the
[support-packing design](dialogue-token-packing-design.md). Enumerate valid
tokens and supported schema pairs from public masks, including dummy question
slots' valid NONE candidate. Project valid tokens once, group real turns by
token length rounded up to width 16 (capped at the input's padded length) and
exact supported-pair count, then pool and differentiably scatter evidence.
Keep the original dense actor assembly, parameter initialization, validations,
normalized head and every ordered recurrent update and monitor call.

Softmax remains over the supported tokens of one turn, followed by the same
L2 normalization. Grouping uses no labels, scored-time eligibility, task
predictions or attention values. No trainable tensor or gradient may be
detached or cached across optimizer updates. All-padded inputs must preserve
the original zero-versus-absent gradient membership. This is a systems
control, not evidence of a new memory or world-model architecture.

## Sixteen fixed synthetic cells

Retain all twelve preceding cells with identical synthetic generation, case
seeds 91201-91212, shapes, and slot/candidate by readout/scalar arms:
`[B,T,Q,C,L] = [1,6,1,7,53], [32,15,10,12,80], [32,23,10,12,90]`.
Do not replace dense workloads with favorable sparse cases.

Add four cells using one fixed public mask geometry, with seeds 91301-91304
in slot/readout, slot/scalar, candidate/readout, candidate/scalar order. Select
the **first maximum-attention-score batch** from the complete closed
`slot_readout-4101` training ledger. Selection uses integer work counts only,
not timings, labels, model quality or prediction arrays. Authenticate the
ledger against its completed fit receipt. Derive each dialogue's public turn
token lengths and per-question candidate counts from existing cache and schema
metadata. Preserve their order. Publish only those lengths/counts and source
provenance in `output/dialogue-token-packing-v1/geometry-01.json`.

The selected geometry is update 1237, the unique maximum among 1,280 closed
ledger rows. Its maximum shape is `[32,23,10,12,90]`, with 374 real turns,
11,304 valid tokens, 1,966 real question updates and 11,781 real candidate
updates. Including the required 1,774 dummy NONE updates gives 13,555 active
candidate rows. Retain that geometry unchanged. Its SHA256 is
`286e209b2ed22a98a51894c0e1438ac084de7c28a3c2f5771f8d6dc4521bcc49`.

Generate fresh artificial float32 vectors and valid artificial labels for
that exact geometry. Reuse the unchanged original batch assembler so padded
turns, candidate masks and dummy question slots are constructed normally.
The geometry is a single preselected shape example, not a representative
sample of all training. No real text, task predictions, encoder outputs,
trained checkpoint, external model call or official test data enters the
qualification. The source-bound old plan supplies only fixed loss weights.

## Correctness and timing

Use the original 173,186-parameter model and the additive packed model, with
384-dimensional input, projection/hidden width 64, GRU width 16, float32 CPU,
four intra-op threads, one inter-op thread and deterministic Torch operations.
Keep AdamW at .001 learning rate, .0001 weight decay and clipping at 1.

For every cell, identical initial tensors and actor inputs must pass output
agreement (absolute `1e-5`, relative `1e-4`), loss agreement (`1e-6`, `1e-5`),
and all input/parameter gradient agreement (`1e-5`, `1e-4`). Require finite
loss and gradients, exact masks and absent-gradient membership. Retain the
original `2e-6` probability checks and complete monitored-update coverage.
Prior small tests must cover mask holes, dense and sparse layouts, all-padded
inputs, exact state carry, causal prefixes, unrelated queries and permutations.

Each path receives one warm update. Use the original path's post-warmup model
and AdamW state as the common measured start, restored and hash-verified for
both paths outside each measured update. Run four alternating pairs in order
original/packed, packed/original, original/packed, packed/original.

Measure the original assembler, grouping/gather/scatter and work accounting,
validation, monitored forward, weighted loss, backward, clipping, AdamW and
scalar readback. Preserve every phase and pair. Report original padded work
and actual packed key rows, attention-score positions, group counts and
logical payload counts. Counter collection is included in timing; it must
not silently move packing outside the measured region. Logical payload is
not physical I/O or peak memory. Keep process-lifetime RSS separately, with
no per-method memory-saving claim. Construction, reset and parity comparison
are included in the whole-command time, outside update timings.

An independent mask-formula reference and comparisons against saved counters
may run outside update timing, with their cost retained in the whole command.
The packed model computes its own actual counters inside its forward path.
Do not execute hypothetical packing in the baseline's timed region just to
collect comparison metadata. Receipt formatting may also be outside timing.

The complete attempt has **160 optimizer updates**, **32 parity
forward/backward passes**, and **192 operation records**. A shorter prefix
does not qualify. These are single-update costs, not a training trajectory.

## Fixed continuation rule

Require all sixteen correctness cells and complete work, plus:

1. Median of four paired original/packed time ratios at least **0.90** in
   each of the original four minimum cells.
2. Median paired ratio at least **1.10** in each of the original eight
   medium/large cells, unchanged from the earlier qualification.
3. Median paired ratio at least **1.10** in each of the four added geometry
   cells.
4. Process-lifetime peak RSS at most **6 GiB**.

The entire qualification has a **300-second** execution cap. Preserve failed
and partial work; no retry, resume, threshold change, favorable repetition or
hardware switch. A pass permits a separately predeclared representative cost
check before a new scientific study. It does not reopen either stopped
attempt or establish that full training fits its limit.

Freeze and publish the complete 46-file source/protocol/geometry closure and
runtime before the sole actual run. Preserve the prior 40 sources byte for
byte. Use new exclusive preparation and run destinations and recheck source
identities at completion. Run no other model, corpus or test workloads during
timing. Publish all sixteen cells, failure status and receipts; verify saved
arithmetic independently without model replay. Numerical equivalence and
compute results do not establish a decision-quality or architecture gain.
