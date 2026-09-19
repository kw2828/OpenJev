# Shared-prefix inference: fixed systems pilot

This tests the useful serving idea in the supplied Jev-inspired diagram against
OpenJev's existing scorer. It does not train a model, reproduce proprietary
RLCD, or test recurrent learning, calibration, biological wiring or accuracy.
See the [upstream source review](qwen-parallel-source-review.md).

## Comparison fixed before execution

Keep the pinned Qwen3-4B-Instruct-2507 4-bit weights, tokenizer, complete prompt
tokens, candidate ordering, final-position readout, full-vocabulary projection
and candidate probability semantics. Four methods:

1. Serial complete prompts, the current execution pattern.
2. Batched complete prompts without prefix reuse.
3. One shared prefix, followed by independent serial suffix branches.
4. One shared prefix, followed by batched suffix branches.

Find the exact longest common **token** prefix, leaving at least one suffix
token. Right-pad batches and gather each last real hidden position before
vocabulary projection. Discard suffix caches after each request. No branch
ever continues through padding. Cache replication consumes memory; this is
not a zero-copy or cross-request cache experiment.

The fixed 54 synthetic English requests cross 1/2/4 questions, short/long
context and 2/4/12 candidates, with three requests per cell. They exercise
unequal suffix lengths. Their text and exact token arrays are saved before any
large-model execution. Synthetic obvious answers are not a labeled benchmark.

## Execution and acceptance

For every request, make one allocator-cold call per method with weights resident,
then five warm calls per method, rotating method order deterministically. Total:
216 allocator-cold and 1,080 warm request evaluations. The former are initial
shape calls and warmups, not independent process-cold measurements. Record model
loading separately. No timing-based retries or selection of fastest requests.

Every result is compared with that request's first serial result. Require exact
selected IDs, maximum absolute candidate probability difference <=0.005 and
candidate vocabulary mass difference <=0.005. Record raw candidate-logit
differences without a separate threshold. These are prospective engineering
tolerances, not proof of distributional identity. Retain mismatches; do not
relax tolerances after reading results. A method can earn a scoped speed claim
only if **all** its recorded requests pass. Report every method and cell even
if global parity fails. No automatic production rollout follows this pilot.

Synchronize around timing. Include prompt tokenization, prefix work, KV
replication, padding, vocabulary readout, CPU conversion and answer formatting.
Exclude request validation, filesystem recording and comparison calculations.
All methods use the same 256 MiB allocator-cache limit. Record actual transformer
calls, prefix tokens, padded token slots, peak active MLX memory and cached
allocator bytes. Memory is device allocator memory, not complete process RSS.

Primary descriptive report: per-cell median/p95 warm end-to-end milliseconds,
paired per-request median speed ratios, parity and memory. Report the one-question
overhead. These deterministic workloads are not independent accuracy trials;
no confidence interval or general 5.6-7x claim is planned.

Strict causal/cache tests use a tiny CPU float32 Qwen fixture. Initial tiny GPU
tests differed by up to about 0.0017 raw-logit units across batch shapes; the CPU
fixture separates mathematical checks from the real GPU parity experiment.
The measured quantized GPU model keeps the fixed acceptance thresholds above.

Source, tests, requests, tokenizer/model files and runtime versions are bound
in the protocol and completion receipts. Run only once into a new directory.
Any failure remains recorded. No upstream inference code is copied or executed.
