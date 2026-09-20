# Time-batched token pooling: prospective engineering qualification

The [shared-token study](dialogue-token-protocol.md) stopped at its unchanged
3,600-second limit, with seven of twelve fits complete and one partial fit.
Keep that attempt and every frozen source unchanged. This qualification tests
an additive implementation change before considering another scientific run.
It neither resumes the failed attempt nor changes its scientific criteria.

## Change and claim boundary

Pool each turn's token evidence in one batch across time, then retain the
original sequential recurrent updates and V2 probability monitor. The schema
queries, token prior, softmax axis, masks, parameter shapes, initial tensors,
loss and optimizer are unchanged. No future token contributes to another
turn's evidence. Floating-point batching can still change rounding and
gradient accumulation. Require measured numerical agreement, not an assertion
of bitwise-identical training trajectories.

This is a systems control, not a new memory, connectome or world-model
architecture. A pass permits designing a representative training-cost check;
it does not authorize or establish that a twelve-fit study fits its cap.

## Fixed synthetic workload

Use all four arms: slot/readout, slot/scalar, candidate/readout,
candidate/scalar. Test each at the following `[B,T,Q,C,L]` shapes, selected
from already closed workload ledgers without inspecting task quality:

- `[1,6,1,7,53]`: minimum attention-score work.
- `[32,15,10,12,80]`: upper-median attention-score work.
- `[32,23,10,12,90]`: maximum attention-score work.

There are twelve cells. Generate synthetic float32 token/schema vectors and
valid labels with seed 91201 plus the fixed case index. Vary public sequence,
query, candidate and token lengths while ensuring the declared maximum shape
is attained. Use the original batch assembler, including its padding and
dummy query slots. No real dialogue, task prediction, trained checkpoint,
encoder, external API, or official test data may be read. The old saved plan
supplies only fixed loss weights and source identities. Hash synthetic inputs.

Both paths use CPU, four intra-op threads, one inter-op thread and deterministic
Torch operations. Use the original 384-dimensional observations, 64-dimensional
projections/hidden layer, GRU width 16 and all 173,186 parameters. Retain AdamW
at learning rate .001, weight decay .0001 and gradient clipping 1.

## Correctness before timing

In each cell, start both paths with identical parameters and actor tensors.
Perform one forward/backward comparison per path. Require:

- Outputs: absolute tolerance `1e-5`, relative tolerance `1e-4`.
- Weighted supervised loss: absolute `1e-6`, relative `1e-5`.
- All floating input gradients and parameter gradients: absolute `1e-5`,
  relative `1e-4`, with identical absent-gradient membership.
- Identical masks, finite support and actual monitor coverage; all internal
  probability checks retain the original `2e-6` tolerance.

Prior unit tests must cover exact carry on padding, causal prefixes,
independence of unrelated queries, unchanged initialization/state dictionaries,
and monitored versus unmonitored gradients. No tensor may be detached or
cached between optimizer updates to obtain a speed benefit.

## Timing and continuation

For each implementation and cell, run one warm update and four measured
updates. The paired order is original/batched, batched/original,
original/batched, batched/original. Use the original path's post-warmup model
and AdamW state as the common measured starting state. Restore that exact
snapshot for both implementations before each pair, outside the timed
interval. This includes initialized optimizer state without allowing weights
to drift across pairs. These are single-update costs, not a training
trajectory. Input gradients are disabled
in timing, as in the actual training study.

Include original batch assembly, monitor setup, input checks, forward,
weighted loss, backward, clipping, optimizer update and scalar readback in
the measured interval. Retain phase timings, every pair, original logical
work counts, and process-lifetime RSS high-water. The latter is not a
per-method peak-memory comparison. Construction/reset, synthetic generation,
parity comparisons and reporting remain in the whole-command cost, separately
from timed updates. Record all 120 warm/measured optimizer updates and the
24 parity forward/backward passes.

Require all correctness checks, all expected work, and:

1. Median paired original/batched time ratio at least **1.10 in each of the
   eight medium/large cells**.
2. Median paired ratio at least **0.90 in each of the four minimum cells**.
3. Process-lifetime peak RSS at most **6 GiB**.

The ratios are engineering thresholds, not confidence intervals or universal
speed claims. Publish every cell, including regressions. The command has a
**300-second** total cap. Timeout, numerical failure or missing work stops
admission; preserve partial ledgers and a failure receipt. No automatic retry,
resume, threshold adjustment, hardware switch or favorable repetition.

## Provenance

Freeze the complete source/test/protocol closure and runtime before the real
qualification. Publish its plan before execution. Each destination must be
new; the run accepts only its frozen plan. Recheck identities at completion.
No other tests, model calls or timing experiments run concurrently. Source
review and saved-output audits may proceed. Report implementation correctness
and speed separately from scientific efficacy. The failed original training
and its uninspected task predictions stay preserved.
