# Shared token cache: preparation result

**The full shared-token cache completed in 17.88 seconds and passed independent
verification.** This establishes a usable input representation for the
[planned attention comparison](dialogue-token-protocol.md), not an accuracy gain.

The [frozen setup](https://github.com/kw2828/OpenJev/commit/bbba56c) was published
before the capacity probe. It retains the same public dialogue context and
frozen MiniLM weights, while preserving contextual token vectors instead of
discarding them after mean pooling. Candidates will query these shared vectors
through a learned attention adapter.

| Measurement | Result |
| --- | ---: |
| Fixed 512-context capacity probe | 4.2479 seconds |
| Projected full preparation | 33.1556 seconds |
| Admission threshold | 720 seconds, passed |
| Actual full preparation | 17.8845 seconds |
| Full execution cap | 900 seconds, passed |
| Unique contexts | 43,553 |
| Public context occurrences | 44,410 |
| Retained valid token vectors | 1,304,325 |
| Recorded encoder calls | 341 |
| Truncated tokens | 0 |
| Cache payload excluding completion receipt | 2,011,987,502 bytes, about 1.87 GiB |
| Cache cap, including all metadata | 4 GiB, passed |
| Maximum pooled-embedding coordinate error | 2.0862e-7, below the 2e-5 tolerance |

The capacity projection includes raw-vector transfer, retention, flushing,
pooling checks and hashing. Its deterministic cold sample is a heuristic,
not representative latency. The actual full runtime is separately measured.
The probe is additional preparation work; it is not included in the full-run
17.88 seconds. Neither timing includes training or live application decisions.

The independent saved-artifact audit verified **87 input hashes**, twelve
preparation sources, six encoder files, token/chunk geometry and every weighted
normalized mean. It independently reconstructed the time and disk admission
arithmetic. Encoder execution, text assembly and elapsed time remain
authenticated producer records; the audit did not rerun the neural encoder.

- [Capacity receipt](../output/dialogue-token-v1/capacity-01/completed.json).
- [Full preparation receipt](../output/dialogue-token-v1/features-01/completed.json)
  and [public metadata manifest](../output/dialogue-token-v1/features-01/publication-manifest.json).
- [Independent audit summary](../output/dialogue-token-v1/cache-audit-01/result-01/summary.json)
  and [audit receipt](../output/dialogue-token-v1/cache-audit-01/result-01/receipt.json).
- [Execution guide](dialogue-token-reproduction.md).

The [earlier per-candidate encoding screen](dialogue-joint-capacity-results.md)
remains rejected. Its projected full cost was never measured, so the two
results do not establish a measured speedup for an equivalent model. The
shared-token representation changes the computation and still needs its own
matched quality comparison. No official SGD test contents were accessed.
