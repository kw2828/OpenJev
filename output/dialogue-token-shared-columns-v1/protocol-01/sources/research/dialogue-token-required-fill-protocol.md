# Required schema fill: prospective cost qualification

The [consolidation qualification](dialogue-token-consolidation-results.md)
passed all sixteen numerical comparisons and 15/16 speed checks, including all
twelve larger workloads. Minimum candidate/readout failed its fixed 0.90
floor. That run and every earlier qualification or incomplete training run
remain closed. This protocol tests the different execution mechanism in the
[required-fill design](dialogue-required-fill-design.md).

## One change: omit fill that is completely overwritten

Retain grouped attention, raw-token pooling and 384-dimensional normalization
(`eps=1e-12`), supported-row concatenation, turn projection plus `tanh`, the
supported observation projection and aligned destination scatter. Change only
which schema rows receive a skipped-position fill. The exact public-mask rule
is:

```text
needs_fill[b,q,c] = not all_t(valid[b,t]) or not candidate_mask[b,q,c]
```

Every omitted fill row must be overwritten at every turn before the recurrent
head consumes it. A padded turn requires fill for every schema coordinate of
that dialogue. Unsupported candidates retain fill; dummy NONE candidates use
ordinary support. Token holes do not change this rule because the original
validation already requires nonempty token support for each real turn.

For `F` required rows among `K=B*Q*C`, use the exact existing fill inputs:
`u0=tanh(turn_projection.bias)`, projected query and candidate vectors, their
interaction features, and zero lexical values. Apply the existing observation
weight columns and bias. With no required rows, skip this projection and use
initialized zero storage. With all rows required, the original full fill path
may be retained. For partial support, gather only the required schema rows,
project them and place them at their original indices in initialized storage.
Expand across time and perform the unchanged supported-row scatter.

These branches depend solely on the public predicate. No method name, case
size threshold, timing outcome, label, prediction or future recurrent state
may choose a path. Do not omit any fill row merely because its output is later
masked. No uninitialized storage, cached mask, activation, outcome or gradient
may be carried across calls or optimizer updates. Preserve all-padding empty
projection and safe zero-gradient connections, and exact zero-versus-absent
gradient membership for every input and parameter.

Inherit the normalized recurrent state path, actual consumed belief/entropy
feature hook, readout reset, scalar departure-mass checks and ordinary step/pool
API unchanged. The monitor-first MRO retains its original `2e-6` tolerance and
turn order. No detached audit substitute or disabled check. Keep all 173,186
parameters, initialization draws, names, supervision and dense input assembly.
Floating reduction and extreme-finite-input failure behavior are not claimed
to be bitwise identical over every possible input.

## Unchanged sixteen workloads

Inherit the exact consolidation plan at
`output/dialogue-token-consolidation-v1/protocol-01/plan.json`, SHA256
`f2a1e01dbde82b180c1bca26df61e67756a952b308bc85a17123baf6c3c03ac8`.
Keep the same artificial generators, seeds, shapes, masks and loss weights.
The first twelve use seeds 91201-91212, `[B,T,Q,C,L]` shapes `[1,6,1,7,53]`,
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
cell, including the previously failing small workload.

## Numerical checks and complete timing

Compare the original model directly with the required-fill implementation.
Use float32 CPU, four intra-op threads, one inter-op thread, deterministic
Torch, input width 384, projection/hidden widths 64 and GRU width 16. AdamW
uses learning rate .001, weight decay .0001 and gradient clipping at 1.

Every cell must pass outputs at absolute `1e-5`, relative `1e-4`; loss at
`1e-6`, `1e-5`; all input/parameter gradients at `1e-5`, `1e-4`. Require exact
masks and absent-gradient membership, finite loss/gradients and full monitor
coverage. Before freezing, focused tests cover float32/float64, all four arms,
no/partial/all required fill, masked candidates, dummy NONE, nonzero biases,
token holes, mixed/all padding, generic widths, causal prefixes, permutations,
inherited monitored steps and no cached activations. Witness actual fill calls
and row contents; verify all omitted destinations are overwritten.

Give each path one warm optimizer update. Before each measured update, restore
and hash-check the same original post-warmup model and AdamW state outside
timing. Run four alternating pairs: original/required_fill,
required_fill/original, original/required_fill, required_fill/original. Preserve
every pair and phase. Require **160 optimizer updates, 32 parity forward/backward
passes and 192 operation records**. A shorter prefix cannot qualify.

Complete-update timing charges original assembly, grouping, mask reduction,
predicate checks, index construction, every gather/concatenation, attention,
normalization, projection, zero initialization, fill scatter/expansion, supported
scatter, sequential state operations, internal counters, validation/monitoring,
loss, backward, clipping, AdamW and scalar readback. Charge allocation and
gradient lifetimes from all new paths. The original baseline must not execute
hypothetical required-fill work for bookkeeping. Independent counter comparison,
receipt formatting, construction, restoration and parity remain in
whole-command time outside measured-update intervals.

## Actual work and references

Record `F` executed fill rows, zero or one fill projection call, corresponding
feature/input/hidden payloads and first-layer multiply-accumulates. Preserve
unchanged supported-row and state work. Record the public predicate and index
payloads, separating boolean and int64 byte estimates from float32 estimates.
The old full-schema row/payload/MAC counts must be explicitly reference-only.
Counters must describe actual branches, including no indices for a retained
all-required dense path if none are constructed. Check formulas independently
from public masks without invoking the model on the original baseline.

The four fixed layouts require `F=0,3025,3025,3789` out of `K=7,3840,3840,3840`.
They remove respectively 176,512; 20,551,040; 20,551,040; and 1,286,016 first-layer
multiply-accumulates. These estimates do not identify the cause of a timing
result or establish a speedup. Dense hidden storage, attention, supported
projections, recurrence and monitoring remain. Added mask/index work may exceed
the savings. Report combined process-lifetime peak RSS; do not infer per-method
memory savings from logical payload counts.

## Fixed decision and execution boundary

Require complete numerical/work coverage and all four conditions:

1. Every minimum cell's median of four paired original/required_fill time ratios
   is at least **0.90**.
2. Every original medium/large cell's median paired ratio is at least **1.10**.
3. Every real-mask cell's median paired ratio is at least **1.10**.
4. Combined process-lifetime peak RSS is at most **6 GiB**.

Use one new exclusive attempt with a **300-second** whole-command cap. Preserve
failures and partial work. No retry, resume, favorable repetition, threshold
relaxation or hardware switch. Run no concurrent model, corpus or test workload
while timing. Freeze and publish **66** source/protocol/fixture identities,
snapshots and runtime before execution. Preserve all 61 inherited sources; the
five additions are the new model, its tests, harness, harness tests and this
protocol. Recheck identities at completion.

Audit saved file closure, operation/work counts, paired arithmetic and thresholds
independently without new model calls. Unsaved output/gradient tensors remain
authenticated execution witnesses rather than independent replay. A pass permits
only a separately declared representative cost check before a new scientific
study. A failure stops this implementation under this protocol. Neither outcome
reopens the incomplete training run or establishes better decisions, architecture
novelty, biological learning or recurrent-world-model advantage.
