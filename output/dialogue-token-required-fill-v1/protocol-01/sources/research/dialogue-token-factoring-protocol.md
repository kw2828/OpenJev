# Factored observation/state head: prospective cost qualification

The [projected-pooling comparison](dialogue-token-projection-results.md)
completed with 10/16 speed cells passing, including all four real-mask cells,
but failed its fixed continuation rule. Its results and all earlier failed or
incomplete attempts remain closed. This is a distinct execution mechanism for
the same model, following the [published design](dialogue-exogenous-head-design.md).

## Mechanism

Preserve the public-mask token groups, 384-dimensional attention pooling and
L2 normalization (`eps=1e-12`), followed by the original turn projection with
bias and `tanh`. Within each group, form the original observation features
`[u, q, c, u*q, u*c, q*c, lexical]`. Apply the existing first feature-layer
weight's first 394 columns and its bias. Scatter this 64-dimensional
preactivation instead of the projected turn vector.

At each sequential head update, form the actual normalized own belief and
entropy using the original equations, including the readout reset. Apply the
same feature weight's final two columns to `[belief, entropy]`, with no second
bias. Add the precomputed observation term, then apply the original feature
`tanh`, output layer, transition and state normalization in the original order.
Splitting this linear sum must not move either nonlinearity or detach a term.

Skipped positions require their correct schema-dependent observation term:
`u=tanh(turn_projection.bias)`, the original projected query/candidate and
interaction features, and zero lexical values. Compute a full-schema fill
once per forward and broadcast it across turns, then replace supported
real-turn positions. Do not use zero fill or skip the supported NONE candidate
of dummy questions. Preserve all-padded input and parameter gradients as zero
or absent exactly as in the original model. Retain original finite-input,
prior and shape validation, including padded entries.

The feature-container adapter must preserve the same Linear/Tanh objects,
parameter names, order and initialization. Ordinary full-feature calls retain
the inherited step API. The split call receives the actual consumed
`[belief, entropy]` tensor as its first argument and the observation term as
its second. The existing feature pre-hook must inspect that actual own belief
through `[..., -2]`; no fake full tensor, detached substitute or disabled check.
An explicit internal marker distinguishes precomputed forward inputs from
ordinary step evidence, including when input/projection/hidden widths coincide.

Every incoming-state, feature, output and scalar departure-mass check remains
inside the original monitor-first MRO, in turn order, at tolerance `2e-6`.
No computed tensor or gradient survives an optimizer update. Packing and
prefill depend only on public masks, observation/schema inputs and current
weights, never labels, scored-time eligibility, selected answers or future
recurrent state. All 173,186 parameters, supervision and the original dense
input assembler remain unchanged.

## Identical sixteen workloads

Inherit the exact projection plan at
`output/dialogue-token-projection-v1/protocol-01/plan.json`, SHA256
`442fb17865fc95e9ca4a72ef6721e12f5cdb77f65a02c0ad8eb6da80a2dab2a3`.
Keep all sixteen cells, including the six previously failing speed cells.
The original twelve use seeds 91201-91212 with `[B,T,Q,C,L]` shapes
`[1,6,1,7,53]`, `[32,15,10,12,80]`, `[32,23,10,12,90]`, each in
slot/readout, slot/scalar, candidate/readout, candidate/scalar order.

The last four retain seeds 91301-91304 and the unchanged mask fixture SHA256
`286e209b2ed22a98a51894c0e1438ac084de7c28a3c2f5771f8d6dc4521bcc49`.
It describes the first maximum-attention-work batch among all 1,280 rows of
the closed `slot_readout-4101` training ledger, update 1,237. It is a single
extreme geometry, not a representative sample. Do not revise selection based
on the already observed timings.

Use the exact original synthetic generators and assembler. All values and
labels remain artificial. No real text, feature vectors, trained checkpoints,
corpus, encoder, external model service or official test data enters this run.
The inherited training plan supplies only the fixed loss weights.

## Agreement, work and timing

Compare the original model directly with the new factored implementation.
Use float32 CPU, four intra-op threads, one inter-op thread, deterministic
Torch, input width 384, projection/hidden widths 64 and GRU width 16. AdamW
retains learning rate .001, weight decay .0001 and gradient clipping at 1.

Each cell must pass original/factored output agreement (absolute `1e-5`,
relative `1e-4`), loss agreement (`1e-6`, `1e-5`), and every input/parameter
gradient (`1e-5`, `1e-4`). Require exact masks and absent-gradient membership,
finite loss/gradients and complete normalization/monitor coverage. Before
freezing, focused tests must cover float32/float64, all four arms, irregular
masks, dummy NONE, nonzero biases and schema fill, all padding, causal prefixes,
permutations, inherited step/pool, parameter identity and the actual monitored
feature inputs on both full and split paths.

Give each path one warm optimizer update. For every measured update, restore
and hash-verify the original path's same post-warmup model and AdamW state
outside the timed region. Run four alternating pairs: original/factored,
factored/original, original/factored, factored/original. Preserve every pair
and phase. Complete all **160 optimizer updates, 32 parity forward/backward
passes and 192 operation records**; a shorter prefix cannot qualify.

Update timing includes the original assembly, public-mask grouping, token and
schema gathering, pooling/normalization, turn projection, lexical gathering,
feature concatenation, all weight slicing or materialization, observation
projection, full-schema fill, scattering, state projection, addition, internal
work counters, validation, monitored recurrence, supervised loss, backward,
clipping, AdamW and scalar readback. Do not charge hypothetical factoring to
the baseline. Independent counter comparisons and receipt formatting may be
outside update timing; their cost remains in whole-command time alongside
construction, state restoration and parity checks.

Report actual group, schema-fill and sequential state-feature positions,
linear-call counts and logical payloads. Record layer multiply-accumulates
as algebraic estimates with explicit dimensions and scope, not hardware FLOPs
or measured traffic. Mark obsolete dense/scatter counter fields as references,
not executed work. Preserve zero-gradient-support and fill overhead. Report
combined process-lifetime peak RSS without claiming per-method peak savings.

## Fixed admission and execution boundary

Require all sixteen numerical comparisons, complete work, and:

1. Every minimum cell's median of four paired original/factored time ratios
   is at least **0.90**.
2. Every original medium/large cell's median paired ratio is at least **1.10**.
3. Every real-mask geometry cell's median paired ratio is at least **1.10**.
4. Combined process-lifetime peak RSS is at most **6 GiB**.

Use one new exclusive attempt with a **300-second** whole-command cap. Retain
failures and partial work. No retry, resumption, favorable repetition,
threshold relaxation or hardware switch. Run no concurrent model, corpus or
test workload while measuring.

Freeze and publish all **56** source/protocol/fixture identities, snapshots
and runtime before the actual run. Preserve all 51 inherited sources byte
for byte. The additions are the new model, its tests, the new harness, its
tests and this protocol. Recheck identities at completion. Independently
audit saved file closure, counts, paired arithmetic and thresholds without
model replay. Unsaved parity tensors remain authenticated execution
witnesses, not independently recomputed proof.

A pass permits only a separately declared representative cost check before
a new scientific study. Failure stops this implementation candidate under
this protocol. Neither outcome reopens the incomplete training attempt or
establishes a full-study runtime, better decisions, a novel architecture,
biological learning or a recurrent-world-model advantage.
