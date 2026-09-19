# Candidate-conditioned dialogue encoding: cost screen

**The probe completed successfully, but the proposed full encoding did not pass
the fixed cost rule.** A 512-text measurement projected **20.91 minutes** for
the full workload, above the **12-minute admission threshold**. Full encoding
and all twelve planned fits were not launched. There is no new accuracy result.

![Measured probe and projected full encoding against the fixed time and disk limits](../output/dialogue-joint-v1/figure/capacity.png)

## Why this control

The [corrected memory study](dialogue-copy-v2-results.md) found weak transfer on
unseen TRUE and DONTCARE values. The proposed control places each supplied
candidate and the current public dialogue context together inside the same
frozen MiniLM encoder. It would compare that representation with independent
sentence vectors, crossed with unchanged readout and scalar-memory heads.
This tests an observation limitation before changing the memory mechanism.

The [protocol](dialogue-joint-protocol.md), preparation source and plan were
[published before the probe](https://github.com/kw2828/OpenJev/commit/51f6c45).
The preparation enumerated **1,202,932 unique candidate/context strings** across
the already exposed training and development cohorts. Official SGD test data
remain untouched.

## Measured and projected cost

The fixed first 512 strings were encoded once on Apple M5 Max using MPS,
float32, batches of 128 and the existing pinned MiniLM revision. Tokenization
covered the full workload, but neural encoding covered only the sample.

| Quantity | Result | Status |
| --- | ---: | --- |
| Total capacity probe | 25.4443 seconds | Measured |
| Sample neural encoding | 0.5582 seconds | Measured, four synchronized calls |
| Sample padded token positions | 41,984 | Recorded |
| Full workload padded token positions | 92,480,952 | Reconstructed from saved lengths |
| Full neural encoding | 1,229.5973 seconds | Projected |
| Full total encoding | 1,254.4835 seconds, or 20.91 minutes | Projected |
| Time admission threshold | 720 seconds, or 12 minutes | Failed |
| Full cache | 1,868,030,878 bytes, or 1.74 GiB | Projected |
| Cache limit | 4 GiB | Passed |

The frozen projection scales the measured sample encoder time by the ratio
of full to sample padded token positions, then adds the observed non-encoding
overhead of 24.8861 seconds. The distinct full-run execution cap was 900 seconds;
it was never exercised because admission failed. The cost rule was not relaxed,
and no replacement backend or retry was used.

The sample was deterministic and included cold-start effects. This is a
heuristic admission estimate, **not measured full latency or proof that a full
run could never fit**. It does not establish general inefficiency of joint
encoding. No comparative task quality was measured by this probe.

## Verification and retained evidence

An independent saved-artifact audit confirmed ten frozen source identities,
six encoder/tokenizer files, four attempted and returned encoder calls, all
512 finite normalized embeddings, and the full workload arithmetic. All
1,202,932 texts were within the 254-content-token chunk limit; there was no
truncation. The audit passed while correctly preserving
`encoding_permitted: false`.

- [Capacity receipt](../output/dialogue-joint-v1/capacity-01/completed.json)
  and [public metadata manifest](../output/dialogue-joint-v1/capacity-01/publication-manifest.json).
- [Independent audit](../output/dialogue-joint-v1/capacity-audit-01/summary.json)
  and [audit receipt](../output/dialogue-joint-v1/capacity-audit-01/receipt.json).
- [Preparation plan](../output/dialogue-joint-v1/protocol/plan.json)
  and [execution guide with recorded stop status](dialogue-joint-reproduction.md).

Sample embeddings, raw dialogue strings and individual token lengths remain
local. Published metadata retain their hashes. The runner and reporter are
implemented and covered by synthetic engineering tests, but neither has run
this twelve-fit scientific comparison. Its fifteen quality checks are
**unmeasured**, not failed accuracy checks.

## Next experiment

The [shared-token proposal](dialogue-token-evidence-design.md) encodes each
dialogue context once, then learns a small attention adapter that pools its
token vectors for each supplied candidate. Its matched control pools by slot
alone, using the same cache and registered parameter count. Both are crossed
with the existing readout and scalar-memory heads.

This adapts an established attention approach described in
[SUMBT](https://aclanthology.org/P19-1546/). It needs a separate prospective
protocol, cost screen and quality comparison. Reduced Transformer repetition
is a design property; better total cost, accuracy, novel memory, and world
modeling remain unestablished.
