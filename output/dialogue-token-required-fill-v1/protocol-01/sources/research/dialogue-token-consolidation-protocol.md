# Consolidating supported rows: prospective cost qualification

The [factored-head qualification](dialogue-token-factoring-results.md) passed
14/16 speed checks and all numerical comparisons but failed its fixed rule.
The failed cells were large candidate/scalar and real-mask slot/readout. All
earlier qualifications and incomplete training attempts remain closed. This
protocol tests the distinct execution change in the
[published design](dialogue-supported-row-consolidation-design.md).

## Mechanism and preserved model

Retain the existing public-mask groups, attention scores, softmax, weighted
raw-token sums and 384-dimensional L2 normalization (`eps=1e-12`). Concatenate
all supported normalized evidence rows in deterministic group/turn/schema
order into one `S x 384` tensor. Preserve aligned schema and destination indices.
Apply the original biased turn projection and `tanh` once to these rows.
Gather projected query/candidate and lexical values in that same order, form
`[u, q, c, u*q, u*c, q*c, lexical]`, and apply the existing first-layer
observation columns and bias once. Scatter the resulting 64-wide preactivation.

The original full-schema fill remains once per forward: projected schema
and interactions, `u=tanh(turn_projection.bias)`, zero lexical values. Keep
supported dummy NONE candidates and all padded-cell validation. Empty inputs
must retain the original zero-versus-absent gradient membership, including
the empty turn-projection call. No raw-token replication across candidates.

Retain the factored implementation's sequential state path unchanged. It
computes the actual belief and entropy, applies the last two weight columns,
adds the observation term and then applies the original feature `tanh`, output
layer, transition and normalization. Preserve the exact parameter objects,
names, initialization draws and inherited raw step/pool API. Keep an explicit
precomputed-input marker even when input/projection/hidden widths coincide.

The monitor-first MRO retains incoming, actual consumed feature, output and
scalar departure-mass checks at `2e-6`, in turn order. No detached audit
substitute, disabled check, label-dependent grouping, future-state access or
activation cache across updates. All 173,186 parameters, the loss and dense
input assembler stay unchanged. Consolidated matrix shapes change floating
reduction order; this protocol claims neither bitwise trajectories nor
universal finite-domain equivalence.

## Identical sixteen workloads

Inherit the exact factored plan at
`output/dialogue-token-factoring-v1/protocol-01/plan.json`, SHA256
`c717e81f88e1c6c3e3fcc43b681fb8f5e6e4c8725b380d3b4ebc56ae9c71d536`.
Keep all sixteen workloads and the original generators and assembler. The
first twelve retain seeds 91201-91212 and `[B,T,Q,C,L]` shapes
`[1,6,1,7,53]`, `[32,15,10,12,80]`, `[32,23,10,12,90]`, each in
slot/readout, slot/scalar, candidate/readout, candidate/scalar order.

The last four retain seeds 91301-91304 and public mask fixture SHA256
`286e209b2ed22a98a51894c0e1438ac084de7c28a3c2f5771f8d6dc4521bcc49`.
It describes the first maximum-attention-work batch among all 1,280 rows of
the closed `slot_readout-4101` training ledger, update 1,237. It is one extreme
geometry, not a representative distribution. Its selection is unchanged.

All values and labels are artificial. No corpus, real text/features, trained
weights, encoder, external-model service or official test data enters this
run. The inherited training plan supplies only fixed loss weights. Previously
failed cells must not receive a special execution path or selective retiming.

## Agreement and timing

Compare the original model directly with the consolidated implementation.
Use float32 CPU, four intra-op threads, one inter-op thread, deterministic
Torch, input width 384, projection/hidden widths 64 and GRU width 16. AdamW
uses learning rate .001, weight decay .0001 and gradient clipping at 1.

Every cell must pass output agreement (absolute `1e-5`, relative `1e-4`),
loss agreement (`1e-6`, `1e-5`), and all input/parameter gradients
(`1e-5`, `1e-4`). Require exact masks, absent-gradient membership, finite
loss/gradients and complete monitor coverage. Focused pre-freeze correctness
tests cover float32/float64, all four arms, irregular masks, dummy NONE,
schema fill/nonzero bias, all padding, causal prefixes, permutations, generic
dimensions, inherited monitored steps and absence of cached activations.
Verify that each supported-row projection really executes once per forward,
without counting the unchanged schema fill as a supported-row projection.

Give each path one warm optimizer update. Restore and hash-verify both paths'
model and AdamW states from the original path's same post-warmup state before
every measured update, outside update timing. Run four alternating pairs:
original/consolidated, consolidated/original, original/consolidated,
consolidated/original. Preserve every phase and pair. Require all **160
optimizer updates, 32 parity forward/backward passes and 192 operation
records**. A shorter prefix cannot qualify.

Charge the original assembly, mask grouping, gathers, attention, normalization,
all concatenation and aligned-index construction, projection, feature
construction, weight slicing/materialization, schema fill, scatter, sequential
state operations, internal counters, validation/monitoring, loss, backward,
clipping, AdamW and scalar readback inside complete-update timing. Include
allocation and gradient work from the larger intermediate. The baseline must
not execute hypothetical consolidation for counter collection. Independent
counter comparisons, formatting, construction, state restoration and parity
checks remain in whole-command time outside measured updates.

## Work and memory accounting

Attention groups and mathematical multiply-accumulate counts are unchanged
from factoring. Count the actual supported-row projection calls as one for
nonempty support, zero for empty support; count empty projection calls and
the one full-schema fill separately. Retain sequential state call counts.
`turn_projection_calls` includes the empty zero-row call, identified by
`empty_turn_projection_calls`; subtracting the latter gives supported calls.
`group_feature_calls` counts the consolidated supported feature call, not
the unchanged attention-group count in `pooling_groups`.
Record supported pooled concatenation scalars `S * input_width` and their
float32 byte estimate. Existing first-layer row/MAC and hidden-scatter counts
remain valid; prior group-call counts must not be presented as executed calls.

For the fixed geometry, 69 calls for each supported-row projection become
one; the pooled concatenation has 20.82 decimal MB instead of the previous
3.47 MB hidden concatenation. The large synthetic case consolidates only six
calls and introduces 106.69 MB of pooled concatenation. These logical payloads
are not measured traffic or peak-memory sums. Gathered schema, lexical and
feature buffers also remain charged. Report combined process-lifetime peak
RSS without inferring per-method memory savings. Counter formulas are checked
independently from public masks, not by invoking the candidate on the baseline.

## Fixed decision and execution boundary

Require all numerical comparisons, full work coverage, and all four conditions:

1. Each minimum cell's median of four paired original/consolidated time ratios
   is at least **0.90**.
2. Each original medium/large cell's median paired ratio is at least **1.10**.
3. Each real-mask cell's median paired ratio is at least **1.10**.
4. Combined process-lifetime peak RSS is at most **6 GiB**.

Use one new exclusive attempt with a **300-second** whole-command cap. Preserve
failures and partial work. No retry, resume, favorable repetition, threshold
change or hardware switch. Run no concurrent model, corpus or test workload
while timing. Freeze and publish the **61** source/protocol/fixture identities,
snapshots and runtime before execution. Preserve the 56 inherited sources;
the five additions are the model, its tests, harness, harness tests and this
protocol. Recheck source identities at completion.

Independently audit saved file closure, operation/work counts, paired timing
arithmetic and thresholds without new model calls. Unsaved output/gradient
tensors remain authenticated execution witnesses, not independent replay.
A pass permits only a separately declared representative cost check before a
new scientific study. Failure stops this candidate under this protocol.
Neither outcome reopens the incomplete training run or establishes better
decisions, novel architecture, biological learning or world-model advantage.
