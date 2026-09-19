# Candidate-memory normalization defect and additive correction

The later [15-fit corrected replication](dialogue-copy-v2-results.md) validates the recorded normalized states and retains the scientific failure, 7/13. This page describes the preceding numerical correction and its synthetic validation.

The frozen candidate-copy comparison has a numerical recurrence defect. Reproducing its saved predictions does not establish that its internal state implements the stated probability transition. The first qualification fit, `scalar-4101`, matched all 62,329 saved predictions, then stopped because its captured factors violated the prospectively fixed normalization checks. The [failed receipt](../output/dialogue-evidence-qualification-v1/replay-01/failed.json) records 74 completed forwards and no completed qualification fit. This correction does not rerun that attempt or relax its tolerance. See the [independent audit and figure](dialogue-evidence-qualification-results.md).

## Why scalar drift grows

The [old scalar implementation](../src/openjev/research/dialogue_copy_memory.py) computes retained log mass as:

```text
log_b + logsumexp(log_b + log_keep)
```

Let `b=exp(log_b)`, `S=sum(b)`, `m=sum(b*r)`, and let the writer sum to one in exact arithmetic. Its actual update is `b' = b*(S-m) + m*w`, so:

```text
S' = S*(S-m) + m
S'-1 = (S-1)*(S+1-m)
```

The intended scalar formula assumes `S=1`. Near that point, normalization error is amplified by approximately `2-m` each turn. At initial departure `sigmoid(-3)`, this is about 1.95. Small float32 errors can therefore grow during an ordinary dialogue. Selective transport does not have this same radial amplification: with an exactly normalized writer its total is `S-m+m=S`. Both methods nevertheless require normalized belief inputs.

The parent `forward` calls `_advance` directly, so the public `step` state's normalization check does not guard every internal forward step. Final softmax normalizes the emitted predictions, allowing saved outputs to remain valid categorical distributions and replay faithfully while the internal state drifts. That final softmax cannot repair earlier feature inputs: for `b=S*q`, the old own-probability feature is scaled by `S`, and its unnormalized entropy is `S*H(q)-S*log(S)` before the candidate-count divisor. Released mass is also affected.

Consequently, the old scalar-versus-selective result cannot isolate the intended retention mechanism. Its saved metrics remain arithmetic facts about those frozen implementations, and its failed continuation decision remains failed. The discovery neither establishes corrected performance nor proves how many errors normalization caused. Only the first scalar fit was inspected by the stopped replay; do not extrapolate its measured drift to every checkpoint.

## The separate V2 implementation

[DialogueCopyMemoryV2](../src/openjev/research/dialogue_copy_memory_v2.py) subclasses the immutable implementation and introduces no parameters. It normalizes `log_b` with `logsumexp` before inherited belief features and transition computation, and normalizes the returned state after each real update. It restores the original state's values exactly on padding, preserves masked zero support, and leaves gradients attached. It adds no probability floor or mass clipping. Normalization is to the working floating-point precision, not an assertion of exact real-number sums.

Configuration explicitly names `dialogue-copy-memory-v2-normalized`. Matching parameter shapes permits mechanical tensor loading, but normalization changes recurrent computation and training gradients. Loading old weights would not produce a corrected trained comparison. Any empirical follow-up needs a prospectively fixed protocol and separately identified results; no old source, checkpoint, prediction, or failure artifact is changed here.

## Synthetic validation only

The [45 focused tests](../tests/test_dialogue_copy_memory_v2.py) passed in 1.28 seconds; Ruff passed. They cover an 18-update float32 drift reproducer with a deliberately tiny initial normalization error, 512-turn normalized forwards for all five methods, 256-turn near-one-hot/small-departure mass checks, actual feature-input normalization, exact padding ownership, forward/step agreement, causal and candidate/query permutation behavior, copy/write endpoints, and finite float32 gradients. No corpus, fitted-checkpoint replay, optimizer step, or encoder call was used. An independent source-only review found no remaining implementation blocker; it did not rerun the tests.

Bound identities:

- Failed qualification receipt: `542bb59858c757292971b9f8d5c708264829bedf1a11de783326745f7acb17dd`.
- Original model source: `182c71a944c6d787533bf129bca1631ff3b366b8559e3db7567e7fd9b19a8717`.
- V2 source: `3fc1e84e5fe9da0076e8d83e7d5c67e61607d67b49ce2b0357a81e2206d3714b`.
- V2 tests: `3d8e4b41425d892ede8293261057c8c905942da80e2cd743537c9dd7d8d44dde`.
