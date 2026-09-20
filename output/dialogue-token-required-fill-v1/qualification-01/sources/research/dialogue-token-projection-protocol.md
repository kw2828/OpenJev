# Project-before-scatter: prospective cost qualification

The previous [mask-packing experiment](dialogue-token-packing-results.md)
preserved numerical agreement but failed its speed rule. Its failed result,
the earlier batching result and the incomplete one-hour training study remain
closed and unchanged. This is a different execution mechanism for the same
model, following the [published design](dialogue-projected-packing-design.md).

## Mechanism and invariants

Retain public-mask token packing and its deterministic groups of token length
rounded up to a multiple of 16 (capped by the supplied padded length) and
exact supported-candidate count. Keep dummy question slots' supported NONE
candidate. In each group, compute attention, the weighted 384-dimensional raw
token sum and its original L2 normalization with epsilon `1e-12`. Then apply
the existing 384-to-64 turn projection, including bias, followed by `tanh`.
Scatter that 64-dimensional result into the dense sequential head input.

Skipped positions must contain the original projection of zero evidence,
`tanh(turn_projection.bias)`. Preserve zero versus absent gradients, including
all-padded inputs. Projection-before-normalization, normalization in 64
dimensions, or moving `tanh` through the attention sum are prohibited because
they change the model. No trainable computation may be detached or cached
across optimizer updates.

Use an explicit preprojected head adapter, with the same feature equations,
state normalization, transition equations and ordered updates. The existing
monitor must still wrap every `_advance` and inspect the actual feature
tensor, incoming state, outgoing state and scalar departure mass. Retain the
original `2e-6` normalization tolerance. The ordinary single-step and pooling
APIs must remain compatible with their original 384-dimensional evidence.
Do not accidentally project twice or bypass monitoring in the optimized path.

Keep all 173,186 parameters, their construction order, names and initialization
draws, the original dense input assembly, lexical features and supervision.
Only public masks determine packing. No labels, scored-time eligibility,
attention values, predictions or recurrent state may choose groups.

## Same sixteen workloads

Inherit the exact parent plan
`output/dialogue-token-packing-v1/protocol-01/plan.json`, SHA256
`60ca972cc71451185d2b564e3a954863103c750bbb002ee58e5a6c02735a8b8d`.
Keep all sixteen cases and their order, including the thirteen previously
failing speed cells. The original twelve cells retain seeds 91201-91212 and
shapes `[B,T,Q,C,L]` of `[1,6,1,7,53]`, `[32,15,10,12,80]` and
`[32,23,10,12,90]`, each in slot/readout, slot/scalar, candidate/readout,
candidate/scalar order.

The final four cells retain seeds 91301-91304 and the unchanged public-mask
fixture, SHA256
`286e209b2ed22a98a51894c0e1438ac084de7c28a3c2f5771f8d6dc4521bcc49`.
This is the first maximum-attention-work batch among all 1,280 batches of the
closed `slot_readout-4101` ledger, update 1,237. It contains 374 real turns,
11,304 valid token rows and 13,555 supported candidate/turn positions,
including dummy NONE. It is a single extreme batch, not a representative
sample of training cost. No selection is revised after earlier timings.

Reuse the exact synthetic generators and original assembler. Values and
labels are artificial throughout. No corpus, real feature values, model
weights, encoder, external model service or official test set enters this
qualification. The inherited plan supplies only the fixed loss weights.

## Numerical agreement and cost

Compare the original model directly with the new projected implementation.
Use float32 CPU, four intra-op threads, one inter-op thread, deterministic
Torch, input width 384, projection/hidden widths 64 and GRU width 16. Retain
AdamW learning rate .001, weight decay .0001 and gradient clipping at 1.

For each cell, identical initial parameters and input tensors must pass:

- Output agreement: absolute tolerance `1e-5`, relative tolerance `1e-4`.
- Supervised loss agreement: absolute `1e-6`, relative `1e-5`.
- Every input/parameter gradient: absolute `1e-5`, relative `1e-4`.
- Exact mask identity and absent-gradient membership; finite loss/gradients.
- Complete monitored-update coverage and all original normalization checks.

Before freezing, run focused implementation and harness tests covering all
four arms, float32 and float64, sparse and irregular masks, dummy NONE,
nonzero projection bias, all padding, causal prefixes, inherited step
equivalence, and actual monitor coverage. A small-test failure can be fixed
before the qualification is frozen; it is not a scientific timing result.

Each path receives one warm optimizer update. Use the original path's
post-warmup model and AdamW state as the common start for every measured
update, restored and hash-verified outside its timed region. Run four pairs:
original/projected, projected/original, original/projected,
projected/original. Retain every pair and phase. This is **160 optimizer
updates, 32 parity forward/backward passes and 192 operation records**.

Time original input assembly, all grouping/gathering, 384-dimensional pooling
and normalization, groupwise projection and `tanh`, bias fill, 64-dimensional
scatter, internal work counters, validation, sequential monitored head,
weighted loss, backward, clipping, AdamW and scalar readback. Baseline timings
must not execute hypothetical packing for comparison metadata. Independent
mask-count comparison and receipt formatting may run outside update timing;
their cost remains in the whole-command total, as do construction, state
restoration and parity comparisons.

Record original padded work and actual packed/group/projected counts, with
explicit widths and logical payload bytes. Distinguish the hypothetical
384-wide dense intermediate from the actual 64-wide scattered result.
Account for bias fill and zero-gradient support rather than silently
omitting them. Logical payload does not establish memory traffic or measured
peak allocation. Report process-lifetime RSS separately for the combined
process, without claiming per-method peak memory savings.

## Unchanged admission rule and stop condition

Require all sixteen numerical comparisons and complete work, plus:

1. Each of the four minimum cells has a median of its four paired
   original/projected time ratios of at least **0.90**.
2. Each of the eight original medium/large cells has a median paired ratio
   of at least **1.10**.
3. Each of the four geometry cells has a median paired ratio of at least
   **1.10**.
4. Process-lifetime peak RSS is at most **6 GiB**.

The complete attempt has a **300-second** wall cap. Use a new exclusive
destination and run once. Preserve failures and partial work. No retry,
resumption, threshold relaxation, favorable repetition or hardware switch.
No other numerical model, corpus or test workload may run during timing.

Freeze and publish all **51** source/protocol/fixture identities, source
snapshots and runtime before the sole actual run. Preserve all 46 inherited
sources byte for byte. The five additions are the new model, its tests, the
new qualification harness, its tests and this protocol. Recheck identities
at completion. Independently audit saved counts, timings, thresholds and
file hashes without replaying the model; disclose that unsaved parity
tensors are authenticated execution witnesses rather than recomputed proof.

A pass permits only a separately predeclared representative cost check before
a new scientific study. It cannot reopen the incomplete training attempt,
establish a full-study runtime, or supply a decision-quality, architecture,
biological-wiring or recurrent-world-model result. Failure stops this
implementation candidate under this protocol.
