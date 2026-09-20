# Mask-packed token pooling: a bounded implementation hypothesis

The sole time-batching qualification completed all 12 synthetic cells, 120 optimizer updates and 24 parity forward/backward passes in 32.725 seconds. All parity checks passed, but only five cells met their fixed speed requirement. Engineering admission failed. This note does not revise that outcome, authorize a retry, or reopen the timed-out scientific study. Task quality remains uninspected.

## What the saved timings establish

The table gives medians of four saved phase timings in milliseconds. `O/B` means original/time-batched. The final column is the median of the four **paired whole-update ratios**, not the ratio of separately median update times. The phase medians need not sum to a median whole-update time.

| Size and arm | Forward, validation, loss O/B | Backward, clipping O/B | Paired ratio | Speed criterion |
|---|---:|---:|---:|---|
| Minimum slot/readout | 2.193 / 1.704 | 1.285 / 0.919 | 1.242 | pass |
| Minimum slot/scalar | 3.695 / 3.187 | 1.686 / 1.346 | 1.168 | pass |
| Minimum candidate/readout | 2.351 / 1.876 | 1.424 / 1.024 | 1.265 | pass |
| Minimum candidate/scalar | 2.737 / 2.151 | 1.544 / 1.103 | 1.262 | pass |
| Medium slot/readout | 89.187 / 76.960 | 79.914 / 93.685 | 0.936 | fail |
| Medium slot/scalar | 94.505 / 90.581 | 85.736 / 99.924 | 0.972 | fail |
| Medium candidate/readout | 91.204 / 81.973 | 84.906 / 98.213 | 0.983 | fail |
| Medium candidate/scalar | 103.202 / 84.439 | 88.762 / 100.879 | 1.026 | fail |
| Large slot/readout | 222.677 / 126.831 | 153.459 / 170.067 | 1.279 | pass |
| Large slot/scalar | 168.476 / 138.115 | 156.561 / 181.671 | 0.971 | fail |
| Large candidate/readout | 156.052 / 145.298 | 143.606 / 181.439 | 0.921 | fail |
| Large candidate/scalar | 153.298 / 149.382 | 145.796 / 165.639 | 1.061 | fail |

Every medium/large cell has a smaller median forward phase but a larger median backward phase under time batching. Assembly medians were 3.23-3.52 ms for medium and 4.77-5.15 ms for large cells; optimizer medians were below 1.09 ms. These are broad instrumented regions, not a causal kernel profile. They cannot attribute the backward increase to a particular contraction, memory layout, allocator, or runtime mechanism. Process-lifetime peak RSS was 3,193,815,040 bytes, below 6 GiB, but is not a per-method peak-memory comparison.

The two implementations had identical logical work in every pair:

| Shape | Valid/padded token positions | Real/padded candidate updates | Attention scores |
|---|---:|---:|---:|
| Minimum | 303 / 318 (95.28%) | 42 / 42 (100%) | 2,226 |
| Medium | 31,983 / 38,400 (83.29%) | 43,194 / 57,600 (74.99%) | 4,608,000 |
| Large | 57,798 / 66,240 (87.26%) | 68,794 / 88,320 (77.89%) | 7,948,800 |

The synthetic cases reproduce closed-ledger maximum dimensions, not the distribution of real masks. In the four closed first-seed training ledgers, only 22.00% of raw token positions and 13.74% of candidate update positions were real. That difference motivates examining padding work; it does not prove packing will be fast. The actual combined token-and-candidate score density cannot be recovered by multiplying those marginal fractions.

## One additive control: support packing with bounded token buckets

Keep the original constructor, parameters, input validation, normalized head and ordered monitor calls. Replace only the pooling implementation:

1. From public masks, enumerate each real `(batch, turn)` and its supported token indices. Project the flattened valid raw tokens once through the existing token-key matrix. Never repeat raw tokens over question/candidate axes.
2. For each batch member, gather every `candidate_mask`-supported `(question, candidate)` pair. **Keep dummy question slots' valid NONE candidate.** The existing assembler computes those slots and the monitor checks them. Do not use scored-label masks, query eligibility, outcomes, attention values, or gold annotations to select work.
3. Group real turns by `(token_bucket, supported_pair_count)`, where `token_bucket = min(input_L, 16*ceil(valid_token_count/16))`. Use deterministic group and within-group index order. Gather each group's tokens, projected keys, priors and schema queries into compact batched matrices. Bucket padding remains masked exactly; softmax still normalizes over that turn's supported tokens only.
4. Compute grouped attention and weighted raw-token sums, apply the same L2 normalization, then differentiably scatter evidence into the original dense `[B,T,Q,C,384]` layout. Fill unsupported candidates and padded turns with zero evidence. Run every original `_advance` and monitor call in its original order, including padded time indices.

This reduces actual key projection and attention arithmetic, unlike time batching alone. It preserves one implementation rule for slot and candidate modes; it does not add a separate slot-specific simplification. Grouping does not mix evidence across turns. All validation remains in place, including rejection of nonfinite values in masked input entries. No trainable tensor or gradient may be detached or cached across updates.

Applying only the frozen synthetic generator's integer length/layout formulas, without regenerating inputs or executing a model, gives these prospective operation counts:

| Shape | Packed key rows | Grouped score positions / old | Groups |
|---|---:|---:|---:|
| Minimum | 303 | 2,191 / 2,226 (98.43%) | 2 |
| Medium | 31,983 | 3,489,040 / 4,608,000 (75.72%) | 3 |
| Large | 57,798 | 6,092,260 / 7,948,800 (76.64%) | 6 |

These counts include dummy NONE support. They are source-derived arithmetic estimates, not measured runtimes. Exact-length grouping would use 39 groups in medium/large cases; the fixed width-16 buckets trade some padded arithmetic for fewer groups. Gather/scatter work, grouping overhead and changed backward kernels can still outweigh the savings, especially in the minimum case.

## Limits and next decision

This proposal leaves the original dense actor assembly, dense evidence destination, head projections and recurrent monitor work unchanged. It therefore does not eliminate the previously recorded 85.353 GB of cumulative emitted token tensors per training fit or promise a speedup equal to the padding ratio. Peak memory can increase through compact copies and index buffers. Packing gradients and reduction order also need numerical verification.

A future implementation would first need the same fixed output/loss/input-gradient/parameter-gradient parity checks, causal prefix and unrelated-query checks, exact padding carry, and full internal monitor coverage. Its cost qualification must charge grouping, gathering, scattering, original assembly, complete forward/backward/optimizer work and memory. Include the prior dense cases and a separately predeclared sparse-mask case based on already exposed workload geometry; do not replace failed cells with favorable ones. Any prospective thresholds and budget must be fixed before that new run. The old 0.90/1.10 requirements, failed qualification and 3,600-second study stop remain unchanged. No full scientific training is authorized by this design.

## Evidence and scope

All payloads in `runs/dialogue-token-batching-v1/qualification-01/completed.json` were rehashed against its manifest. Completion SHA256: `3f92748c153bded89a8c2b49eb39340e7ee5b67a3b879b1a693b5bbb0cca9485`. The 12 case files supply the phase timings and logical counters above. Relevant frozen sources are:

- Qualification harness: `2fb43630e604d6894454594d5c330838e1c25c5eb854afb5032e74226ddb3985`.
- Time-batched model: `450bfdcc467a51c44bbfcbd02cde9376ebe90118eca017d2b3647dd248e022d3`.
- Original token model: `07a9b5d11ba4569417ede208bf1b6498cfac3488db8c8daec1062053a8bdc5eb`.

The closed real-work comparison and its receipt/ledger hashes are recorded in [the preceding cost note](dialogue-token-cost-design.md). This review used saved JSON, source and integer arithmetic only. No implementation, tests, model/encoder calls, corpus reads, task-quality inspection, or old-file edits were performed.
