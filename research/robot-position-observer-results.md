# Position-only observer: all six fits complete, scientific criteria fail

**POSITION_OBSERVER_DEVELOPMENT_FAIL: 3 of 5 criteria pass; all five were required.** Starting the correction gain at `[I;0]` allowed all six prescribed fits to complete under the unchanged optimizer and gradient guard. Forecast quality barely improved over the last-two-position initializer: **0.023% lower mean error**, with **36.8% higher request latency** and **11.1% more persistent numeric storage**. This does not establish a useful learned-memory improvement.

The selected learning rate was **0.003**, chosen by pooled H128 squared error on the two original DEV recordings. Both rates, 0.001 and 0.003, completed all three seeds. Inherited control rates remained fixed. The [protocol](robot-position-observer-protocol.md), [full audited results](robot-position-observer-results/audit.json), and [complete family table](robot-position-observer-results/table.md) retain the declared comparison.

## Forecast quality and cost

Errors below are standardized H128 RMSE, averaged across three seeds and then equally across four files; a reference has one instance. Latency is the median of three instance medians, or the reference's own median.

| Method | Mean error | Request ms | Persistent bytes |
|---|---:|---:|---:|
| Position observer, learned gain | 0.649536 | 5.295 | 2,888 |
| Position observer, fixed gain | 0.660341 | 5.339 | 2,888 |
| Last two positions | 0.649685 | 3.872 | 2,600 |
| Frozen-cell local affine | 0.643459 | 3.964 | 4,088 |
| Frozen-cell temporal affine | 0.641713 | 3.843 | 4,088 |
| Jointly trained temporal affine, best control | **0.615141** | 3.845 | 4,088 |
| GRU10 | 0.693458 | 2.086 | 5,488 |
| Causal ridge, penalty 100 | 0.700742 | 0.528 | 3,565,264 |

The candidate's error was **5.59% higher than the best control**. The registered requirement was at least 5% lower, corresponding to an error no greater than 0.584384. Learning the gain improved error by 1.64% relative to the new fixed `[I;0]` control, but left it almost tied with last-two initialization.

The per-file harm criterion also failed. On recording **22H_10M**, candidate error was 0.686064 versus 0.671410 for the best simple initializer, a **2.183% increase** against the 2% limit. The other three files met that limit: increases of 1.506%, 0.786% and 1.537% over their respective best simple controls.

Three criteria passed: complete eligible forecasts and costs, latency within 1.5 times last-two initialization, and no comparator simultaneously matching or improving error, latency and storage with at least one strict improvement. That last criterion preserves a small tradeoff point; it does not override the two failed accuracy criteria.

[Full-range chart](robot-position-observer-results/benchmark.png) · [Clearly marked 0–2 detail](robot-position-observer-results/benchmark-detail.png) · [Seven diagnostic probes](robot-position-observer-results/diagnostic-probes.png)

The full-range chart uses shared symlog error axes because the prior fixed `[I;I]` observer has mean error approximately **1.798 × 10⁸**. The detail chart marks every out-of-range point and mean at its upper boundary. All 19 families, declared rates and failed outcomes remain in the [488 score rows](robot-position-observer-results/scores.csv).

## What the diagnostics establish

The transition's **590 parameters remained frozen**. Only the 72-value gain was trained, starting at `[I;0]`; the fixed control stored the same gain as a 288-byte buffer. Both carry 12 state coordinates. The lower six coordinates are learned latent state, not verified physical velocity. The observer assimilates 30 observed prefix positions before the input-driven forecast.

All six fits completed **4,096 updates each, 24,576 in total**, with no reported nonfinite gradient entries or native clipping norms. The seven prescribed FIT-only probes made zero optimizer updates and left their weights unchanged. Five probes had finite native norms. Two retained failures distinguished finite gradient entries from an overflowing float32 norm:

- Old `[I;I]`, seed 8102, first batch: all 72 gradient entries were finite; the diagnostic float64 norm was **9.256 × 10²⁶**, while native float32 clipping returned positive infinity.
- Archived failed seed 8103 checkpoint, batch 22: all 72 entries were finite; its float64 norm was **2.042 × 10¹⁹**, while the native norm again returned positive infinity.

The new position-only initializations had finite native norms on all three first-batch probes. Their maximum absolute prefix-state coordinates were **2.78–3.40**, compared with approximately **4.08 × 10⁴, 7.06 × 10⁹ and 1.08 × 10⁷** for the paired old initializations. Their diagnostic gradient norms were 0.211–0.351. These paired no-update observations show that the initialization changed forward prefix behavior as well as gradient magnitudes. They do not establish global stability, and no learning-rate effect can be inferred before an optimizer update.

The float64 diagnostics were observational. The original native float32 clipping operation and failure guard were retained. Neither a precision rescue nor a restarted fit was used.

## Execution, accounting and evidence

The original study process completed in **510.357 seconds**; the independent audit process completed in **6.926 seconds** with agreement **PASS**. Individual fresh fits took 78.10–88.30 seconds on the recorded single-thread CPU setup. Training times include diagnostic collection and preservation within their declared scopes. There was no matched diagnostics-disabled training run, so diagnostic overhead cannot be isolated or converted into a training speed claim.

Inference timing included normalization, casts, the complete prefix, operator preparation, H128 rollout, finite checks, denormalization and deadline callbacks. All **46 cost slots** completed three warmups and retained twenty timed samples. Diagnostics were disabled for inference. Persistent bytes include all weights, fixed buffers, caller state and normalizers; temporary workspace, Python object overhead and model loading are excluded. [All cost records](robot-position-observer-results/costs.csv) preserve the instance measurements.

The evidence contains **19 displayed families, 18 eligible families, 488 score rows and 244 forecast attempts**. The 416 parent rows and 208 parent attempts were preserved unchanged, including 24 failed score rows from the prior learned observer. All 36 new forecast banks were saved and replayed exactly by the auditor using qualified model code. New scoring, selection and rules were independently reconstructed. The audit reconciled raw gradients and native norm scalars for seven probes and six last training attempts; earlier diagnostic histories are recorded execution evidence, not independently replayed backward passes.

The [evidence directory](robot-position-observer-results/README.md) and [manifest](robot-position-observer-results/manifest.json) retain 45 model states, six fresh checkpoint/Adam/trace/diagnostic bundles, original qualifications and process closures, source snapshots and parent provenance. Four measured target-window files and external recording arrays remain local under descriptors. The frozen registration SHA is `531663fcb323f0b06d2e8f149ea4b531d3f223e8d8d9609bcc846ddf3a059c0f`; the prefit commit is `4c5f479471a298ebece1a38d8bd13357f60312d9`.

All four evaluation recordings were already exposed development data from the same robot and day. Forecasts receive realized future torques. Official TEST remains closed. The completed run supports this particular numerical intervention's observed behavior, not a stability guarantee, renewed confirmation, calibrated uncertainty, closed-loop control success or biological/architectural novelty. The [prior observer failure](robot-observer-results.md) remains part of the record.

The [subsequent joint observer experiment](robot-joint-observer-results.md) is now complete and independently audited: all twelve fits finish, but its accuracy advantage is insufficient and only four of five criteria pass. The [original rationale](robot-observer-next-direction.md) remains available. A separate [initialization-by-norm factorial](robot-observer-norm-factorial-proposal.md) would clarify numerical attribution, but remains unrun and deprioritized because fixing training completion has not produced useful forecasts.
