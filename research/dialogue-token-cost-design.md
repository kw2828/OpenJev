# Dialogue token pooling: bounded cost control

This is a source and closed-receipt diagnosis, written while the twelve-fit study was still running. It uses only the four completed seed-4101 fit receipts and batch ledgers. No predictions, accuracy, live fit logs, model execution, or encoder calls were inspected or run. It does not establish the study's terminal status or change its fixed 3,600-second cap or scientific rule.

## Measured work, not a kernel profile

All four completed fits have exactly the same training and evaluation work counts. Each fit has 20 epochs and 1,280 updates. Each ledger's SHA matches its closed fit receipt, and summing its work fields reproduces that receipt. Evaluation batch work likewise sums to the recorded evaluation totals.

| Per fit | Training | Final evaluation |
|---|---:|---:|
| Batches | 1,280 | 74 |
| Pooling/head time-step invocations | 23,746 | 1,084 |
| Valid raw token positions | 12,223,820 | 707,593 |
| Padded raw token positions | 55,568,312 | 2,554,630 |
| Emitted raw-token tensor bytes | 85,352,927,232 | 3,923,911,680 |
| Logical raw-token gather bytes | 18,775,787,520 | 1,086,862,848 |
| Attention score positions | 5,836,415,330 | 152,905,732 |
| Pooled candidate positions | 79,148,484 | 2,028,002 |
| Real candidate updates | 10,872,080 | 669,083 |

Training allocates 4.546 padded token positions per valid position; only 22.0% of those positions are valid. Pooled candidate positions exceed real candidate updates by 7.280 times. These are distinct padding effects; their ratios must not be multiplied into a claimed useful-work fraction. Across the four fits, training plus evaluation emits 357.107 GB of token tensors, gathers 79.451 GB of raw token payload, and evaluates 23.957 billion attention score positions. These are cumulative logical payload/work counts, not peak memory, physical disk traffic, measured memory bandwidth, or a hardware FLOP profile.

Recorded training times are 647.180, 725.147, 707.202, and 657.453 seconds for slot-readout, slot-scalar, candidate-readout, and candidate-scalar. Their evaluation times range from 4.719 to 23.539 seconds despite identical logical counts. Whole-fit time before receipt writing sums to 2,805.760 seconds. These observations cannot separate arithmetic, allocations, validation, Python dispatch, library scheduling, or shared-host effects. They do not justify a stable throughput estimate or extrapolated speedup.

## One implementation control worth qualifying

The current [token model](../src/openjev/research/dialogue_token_memory.py) computes schema attention queries once, but calls `_pool` separately at every padded time index. `_pool` uses only that turn's raw tokens, prior and masks, plus fixed schema queries. It never reads recurrent belief, labels, other turns, or the head's output. Therefore pooling can be computed across time before retaining the exact original monitored `_advance` loop.

The additive implementation would project keys as `[B,T,L,64]`, compute scores with `bqck,btlk -> btqcl`, apply softmax strictly over `L`, and pool with `btqcl,btld -> btqcd`. The original normalized head still consumes one evidence slice at a time and retains all state, padding and monitor checks. Computing independent slices together does not mix future evidence into earlier slices. This is an implementation control, not a new memory mechanism.

For these closed training ledgers, pooling invocations would fall from 23,746 to 1,280, an 18.55-fold reduction in calls. Token projection, score, weighted-sum and candidate counts remain unchanged. The head and its checks still run 23,746 times. The existing schema projection is already outside the loop. Reducing padded arithmetic would require a separate valid-row packing change; it is not a benefit of batching time alone.

Batching has a memory tradeoff. The largest recorded batch score tensor would contain 31.795 MB of float32 scores, and the largest pooled evidence tensor 135.660 MB. These maxima are not a full peak-memory estimate: weights, other intermediates, gradients, allocator behavior and retained autograd graphs also matter. The present training graph already retains information across time until backward, so neither a constant-memory claim nor a simple multiplication of current peak RSS is justified.

## Bounded next admission

If the current study times out, preserve it as the failed fixed-cap attempt. Do not resume it or extend its cap. Before another real training attempt, freeze a small synthetic implementation qualification comparing the existing path with time-batched pooling. Fixed `[B,T,Q,C,L]` shapes selected only from the closed work ledger are `(1,6,1,7,53)`, `(32,15,10,12,80)`, and `(32,23,10,12,90)`, the minimum, upper-median, and maximum attention-score workloads. Exercise both pooling modes and both heads, with identical parameters, public inputs and masks.

Require output, supervised loss, input-gradient and parameter-gradient agreement under prospectively specified float32 tolerances; exact state carry on padding; prefix and unrelated-query independence; and unchanged internal normalization coverage. No tensors may be detached or cached across optimizer updates. Changed matrix batch shapes or gradient accumulation order can change floating-point results, so mathematical equivalence does not promise bitwise training trajectories.

Measure complete packing, validation, forward, backward and optimizer work, separately identify pooling where possible, and report peak memory and all logical work. Predeclare repetitions and execution order so host variability is visible. A measured end-to-end benefit with acceptable memory is needed before another full-run admission. Fewer calls alone is not evidence that the twelve-fit study will fit its cap. No such qualification has been implemented or executed here.

## Closed evidence identities

All paths below are under `runs/dialogue-token-v1/study-01/fits/`. Receipt hashes identify the inspected closed files; this is not authentication of a completed enclosing study.

| Fit | `completed.json` SHA256 | `batches.jsonl` SHA256 |
|---|---|---|
| slot_readout-4101 | `9c9bb4ca7a6bf2d197fb540267315007bfedf4ff63d1847c6d4ab6a0ecf98665` | `017b24dd01d1366a53bb6d5e1bfd606fd0f7097b0eb8f4a2b0c6b01c17785213` |
| slot_scalar-4101 | `0bd943705fae0e038d68d908ccbb9a817b61b6d58db72c0cc19544d922e102ba` | `1cc1509531c0fba34a379b4e2fa7d3fba227dda97920ac75722fea41fcfb5835` |
| candidate_readout-4101 | `af5f01ecf1264f619db1a54841f578aab137003ba2dd2e5775b54bf973707c32` | `b6815388f4f8eef020998b30496c4bd67b02d723de1d3a1394191c052c02e833` |
| candidate_scalar-4101 | `a3066c9b8eb908c1ca008d8e3806e9dd2f9328b6fbdc8b3aff7653f1ece1c42a` | `8d2b679775d5b1c5977edd2abc94345665d972e52d1f73126d9a99b8bde0aa54` |

Source identities:

- Token model: `07a9b5d11ba4569417ede208bf1b6498cfac3488db8c8daec1062053a8bdc5eb`.
- Joint normalized head: `30a3e9180916537b261dfb7b0450730ff97542bf07b259fe4a88cd5d2ba0ba5e`.
- Token runner and logical counters: `88e28a1455862c02f562798b31c1b9992c844b7abcb35c4a8649e7ff37075bee`.
- V2 internal monitor: `e7ba25714685e905071ba7c4381c9ebc5253039773c8daafc21dbffb9ffe3d39`.
