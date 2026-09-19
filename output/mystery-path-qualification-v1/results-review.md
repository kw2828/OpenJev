# Independent Mystery Path qualification review

**The completed qualification fails: 16 of 17 exact checks pass. Keep `CLOSE_THIS_RECIPE`.** The full public map succeeds on 195/256 layout/order pairs (76.171875%), below the frozen 80% requirement. At this sample size, 205 successes are required: the full map is ten successes short. Every declared pooled and per-order advantage over the three memory controls passes. Those gains do not rescue the absolute success requirement.

| Controller | Successes / 256 | Success rate | Orders 0, 1, 2, 3 | Mean actions, all episodes | Mean falls |
|---|---:|---:|---|---:|---:|
| full | 195 | 76.171875% | 49, 48, 49, 49 | 79.652344 | 8.820312 |
| last16 | 58 | 22.656250% | 16, 15, 13, 14 | 102.093750 | 20.941406 |
| last32 | 76 | 29.687500% | 20, 18, 18, 20 | 95.375000 | 16.605469 |
| erase_on_failure | 34 | 13.281250% | 10, 10, 7, 7 | 112.226562 | 29.859375 |
| reference | 256 | 100.000000% | 64, 64, 64, 64 | 13.265625 | 0.000000 |

| Full map versus | Both succeed | Full only | Comparator only | Neither | Pooled advantage | Per-order net successes |
|---|---:|---:|---:|---:|---:|---|
| last16 | 58 | 137 | 0 | 61 | +53.515625 pp | 33, 33, 36, 35 |
| last32 | 76 | 119 | 0 | 61 | +46.484375 pp | 29, 30, 31, 29 |
| erase_on_failure | 34 | 161 | 0 | 61 | +62.890625 pp | 39, 38, 42, 42 |
| reference | 195 | 0 | 61 | 0 | -23.828125 pp | -15, -16, -15, -15 |

The three memory controls never succeed on a pair where full memory fails. The observed full-map advantages are 53.515625 percentage points over last16, 46.484375 over last32 and 62.890625 over erase-on-failure. This supports the descriptive importance of retained public information for these controls. It establishes neither a trained-model result nor architectural novelty, and this particular qualification recipe remains closed.

All 17 checks were reconstructed by integer cross-multiplication, independently of the reporter implementation: reference requires 244/256; full requires 205/256; each pooled advantage requires 39 net successes; each of twelve order-specific advantages requires four net successes out of 64. All counts, 2x2 pairing tables, gates and the stop decision match the frozen saved report. All 64 distinct saved layouts and all four orders are retained. The four orders are correlated repeats, not four independent datasets.

Verification authenticated 35 live source/evidence bindings against prospective Git commit `540ec3071ba5112be150c57aad6425f97ab0a0f6`, all 14 installed upstream Python files and package/Python runtime metadata. Earlier 116- and 136-file source sets remain unchanged. All 1,280 NPZ hashes (9801488 bytes), the exact 1286-file attempt membership, execution and audit ledgers, and report sidecars match. Saved public actions, ignored failure returns, success rewards, terminal limits, falls, retained-memory peaks and timing arrays were checked without calling a controller or environment. Full raster replay belongs to the completed official audit; this review checks each payload hash and frame header, not an independent image decoder.

The four-seed preflight is separately bound, with 87 reference/adapter actions and one forced failure/return per seed. Its seeds 0-3 and saved layout hashes are disjoint from the qualification. The qualification executes 103,069 native actions; the completed audit replays the same 103,069. Preflight actions are excluded from both totals. All run/audit/report process exit receipts are zero and their logs are hash-bound. These local records establish consistency, not an external attestation of the process.

Execution took 18.858658 seconds; official replay audit took 15.526502 seconds; the outer run/audit/report process took 35.468594 seconds. Each execution/audit phase was below its separately declared 900-second cap. Shared-host, instrumented whole-episode timing is not isolated policy latency. Full-map peak serialized state was 219 bytes versus 641/1,213 for last16/last32: retaining raw windows can cost more bytes than a compressed map. These are retained-state measures, not process peak memory or equal-compute architecture comparisons.

No cases, priorities, windows or thresholds were changed, and no environment, model, controller or RNG calls were made for this review. An initial review-only report-sidecar lookup expected a hash string where the schema contains `{bytes, sha256}`; the corrected lookup verified both fields. This did not reveal an artifact failure or alter any experimental data.

Key source receipts:

- Protocol: `b2bbb48c299f102a11857215589ab8af86ebe15d0e73e514e81e89f9845592a6`
- Bindings: `2b05dd3b4b0991a085c1be99b5277bcc9951102433351702647ca70672a3e5a8`
- Execution completion: `d6fd5c35c53e66d6b453b02027bc1e8fc407ae66b05385b95ac093c31536163b`
- Replay completion: `b7e64a15364bf4e4bafaf108b3317743b7394eb19346c3b2f79a6336192d63de`
- Saved summary: `628a5723d1df02ea0de8473d2e0f1a1652817cba54e6df5e2ad4c918694d188f`
- Independent numerical review: `80904ee74a6d63116c5ed960389b32152a2c85f4ca9c2a7dab9cdcf605dd28cb`
