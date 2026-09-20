# Shared state columns: prospective cost qualification

The [required-fill qualification](dialogue-token-required-fill-results.md)
passed all sixteen numerical comparisons and 15/16 speed checks. Minimum
candidate/readout failed its fixed 0.90 floor. That attempt and all earlier
qualifications and incomplete training remain closed. This protocol tests the
execution change in the [shared-column design](dialogue-shared-state-columns-design.md).

## One change: share a differentiable parameter view within each forward

Create `state_columns = feature.weight[:, -2:]` once per forward in the
caller's current gradient context. Pass that same view explicitly to every
ordered state-feature call using a private typed record. Retain every actual
state linear call, `tanh`, readout reset or scalar transition, normalization,
padded-row operation and monitoring check. Argument zero to the feature
container remains the actual live belief/entropy tensor consumed by the head.

The current required-fill implementation constructs this same slice separately
at every turn. Sharing its differentiable ancestry may reduce backward work,
but that is a hypothesis. The shared object is a view of the existing parameter,
not copied weight storage. Do not detach or clone it, enter an extra no-grad
context, register a new parameter/buffer, or retain any tensor between calls.
Each forward creates a fresh view, including after loading weights or an
optimizer update. Do not mutate a parameter between a forward and its backward.

Reuse the original Linear and Tanh child objects, names, parameter order and
initialization draws. A typed adapter must delegate ordinary full-feature
calls and inherited one-step behavior unchanged. No width heuristic, case-size
route, outcome-dependent branch, readout shortcut or monitor relaxation.
Preserve all 173,186 parameters, training supervision and dense input assembly.

All grouped attention, raw-token pooling and 384-dimensional normalization
(`eps=1e-12`), supported-row consolidation/projections, required schema fill,
scatter and recurrent equations are inherited unchanged. Preserve all-padding
empty graph connections and exact zero-versus-absent gradient membership.
The monitor-first MRO retains its original `2e-6` tolerance and turn order.
Sharing gradient ancestry can change floating addition order. Bitwise optimizer
trajectories and universal extreme-finite-input equivalence are not claimed.
No new thread-safety claim: existing monitored wrappers have mutable counters.

## Unchanged sixteen workloads

Inherit the exact required-fill plan at
`output/dialogue-token-required-fill-v1/protocol-01/plan.json`, SHA256
`871d3e5633290f9015af02aaab7ec136adf3743b5dec1f0ff5e16b4c7498987e`.
Keep its artificial generators, seeds, shapes, masks and loss weights. The
first twelve use seeds 91201-91212, `[B,T,Q,C,L]` shapes `[1,6,1,7,53]`,
`[32,15,10,12,80]`, `[32,23,10,12,90]`, each in slot/readout, slot/scalar,
candidate/readout and candidate/scalar order.

The last four use seeds 91301-91304 and unchanged mask fixture SHA256
`286e209b2ed22a98a51894c0e1438ac084de7c28a3c2f5771f8d6dc4521bcc49`.
Its first maximum-attention-work batch is update 1,237 among all 1,280 rows of
the closed `slot_readout-4101` ledger. It remains one extreme geometry rather
than a representative distribution. No new fixture selection.

Values and labels are artificial. No corpus, real text/features, trained
weights, encoder, external-model service or official test data enters this run.
The inherited training plan supplies only fixed loss weights. Preserve every
cell, including the previously failing minimum workload.

## Numerical checks and complete timing

Compare the original model directly with shared columns. Use float32 CPU,
four intra-op threads, one inter-op thread, deterministic Torch, input width
384, projection/hidden widths 64 and GRU width 16. AdamW uses learning rate
.001, weight decay .0001 and gradient clipping at 1.

Every cell must pass outputs at absolute `1e-5`, relative `1e-4`; loss at
`1e-6`, `1e-5`; all input/parameter gradients at `1e-5`, `1e-4`. Require exact
masks and absent-gradient membership, finite loss/gradients and full monitor
coverage. Before freezing, focused tests cover all four arms in float32 and
float64, generic widths, masked/dummy NONE candidates, nonzero biases, token
holes, mixed/all padding, causal prefixes, permutations and ordinary monitored
steps. Witness the same view object at every split call within a forward and a
fresh view across forwards. Check two forward graphs before a combined backward,
freshness after weight updates/loads, exceptions without a retained view, and no
new registration or initialization draw.

Give each path one warm optimizer update. Before every measured update, restore
and hash-check the same original post-warmup model and AdamW state outside
timing. Run four alternating pairs: original/shared_columns,
shared_columns/original, original/shared_columns, shared_columns/original.
Preserve every pair and phase. Require **160 optimizer updates, 32 parity
forward/backward passes and 192 operation records**. A shorter prefix cannot
qualify.

Complete-update timing charges original assembly, all inherited packing/fill
work, view creation, typed records/dispatch, every state operation, counters,
validation/monitoring, loss, backward, clipping, AdamW and scalar readback.
Charge new allocation and gradient lifetimes. The original baseline must not
execute hypothetical shared-column work for bookkeeping. Independent counter
comparison, construction, restoration, parity and receipt formatting remain
inside whole-command time outside measured-update intervals.

## Source work and logical view geometry

Retain the seventy required-fill counters, their reference-only fields and
boolean/int64 accounting. Add source-level state-column slice expressions,
the prior per-turn reference count, and logical view dimensions/elements.
For the fixed minimum/medium/large/geometry shapes the source expression count
changes from 6/15/23/23 to one; state linear calls remain 6/15/23/23.
The default view is 64-by-2, with 128 logical elements. Neither these elements
nor their dtype-scaled logical bytes are a new copied/allocated payload.
Keep them separate from the existing float payload map.

Check expected counters independently from public masks without model execution
on the baseline path. Do not call source-level counts measured backward nodes,
physical traffic, saved bytes, peak memory or speed. Preserve all inherited
feature MAC counts. Report combined process-lifetime peak RSS; it cannot identify
per-method peak memory or attribute timing changes to the proposed mechanism.

## Fixed decision and execution boundary

Require complete numerical/work coverage and all four conditions:

1. Every minimum cell's median of four paired original/shared_columns time
   ratios is at least **0.90**.
2. Every original medium/large cell's median paired ratio is at least **1.10**.
3. Every real-mask cell's median paired ratio is at least **1.10**.
4. Combined process-lifetime peak RSS is at most **6 GiB**.

Use one new exclusive attempt with a **300-second** whole-command cap. Preserve
failures and partial work. No retry, resume, favorable repetition, threshold
relaxation or hardware switch. Run no concurrent model, corpus or test workload
while timing. Freeze and publish **71** source/protocol/fixture identities,
snapshots and runtime before execution. Preserve all 66 inherited sources;
the five additions are the model, model tests, harness, harness tests and this
protocol. Recheck identities at completion.

Independently audit saved file closure, operation/work counts, paired arithmetic
and thresholds without new model calls. Unsaved output/gradient tensors remain
authenticated execution witnesses rather than independent replay. A pass permits
only a separately declared representative cost check before a new scientific
study. A failure stops this implementation under this protocol. Neither outcome
reopens the incomplete training run or establishes better decisions, architecture
novelty, biological learning or recurrent-world-model advantage.
