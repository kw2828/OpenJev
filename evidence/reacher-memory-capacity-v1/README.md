# Reacher memory engineering capacity

A completed synthetic capacity check projects **10.79 minutes of training plus 4.11 minutes of learned search**, about **14.90 minutes for those components**. This informs consideration of a fixed 3,600-second execution cap. It does not demonstrate end-to-end completion within that cap or scientific effectiveness.

The one recorded attempt completed in 3.115159 seconds. It ran four optimizer updates per architecture on generated CPU float32 data, then three noninteractive CEM searches per architecture. The training batches contained 32 complete 50-step sequences; searches used 64 cases, 256 candidate evaluations per case, and horizon 12. The three GRUs had width 64; the packet MLP had width 107. One paired initialization was exercised, not all three planned fits. No native environment episode, utility measurement, real training corpus, or scientific score was used. The completed attempt is preserved without a rerun.

| Model | Parameters | Four-update trainer wall (s) | Three CEM calls (s) | Projected training, 3 fits (s) | Projected search, 9 rows (s) |
|---|---:|---:|---|---:|---:|
| residual_gru | 36,805 | 0.145644 | 0.136055, 0.141148, 0.109020 | 125.837 | 57.934 |
| current_gru | 36,805 | 0.132972 | 0.112004, 0.122308, 0.114435 | 114.888 | 52.312 |
| bounded_gru | 36,805 | 0.338640 | 0.146565, 0.155692, 0.156833 | 292.585 | 68.863 |
| packet_mlp | 36,599 | 0.132265 | 0.164304, 0.146548, 0.138136 | 114.277 | 67.348 |

Training is projected as each recorded trainer wall divided by four, multiplied by 1,152 updates and three fits. All four update timings, including the first, are retained. Search is projected from the mean of three full-batch calls times 50 decisions, three fits and three panels. The searches use the same real startup root with different saved-plan innovation streams; they do not exercise evolving closed-loop state. Terminal search horizons would shrink in the actual study. These are short engineering samples, not a convergence or latency study.

Twice the projected measured components is 29.80 minutes, leaving 30.20 minutes inside a proposed 60-minute cap for unmeasured execution work. This arithmetic is a planning margin, not an upper bound. **The projections exclude native simulation/data collection, reference controllers and filters, real assimilation and selected-action advances, training preparation and restoration, prediction evaluation, serialization/trace storage, manifest hashing, audit and publication.** A separate enclosing protocol must fix the actual execution and audit limits.

[Raw timings](raw-timings.json) preserve the exact completed measurement JSON, including every recorded per-update and CEM time. [Summary](summary.json) provides independently checked projection arithmetic. [Manifest](bundle-manifest.json) binds all ten original attempt files, including four safe-loadable tensor checkpoints, and all nine source/script snapshot members. The compact source snapshot contains the eight files hashed at execution plus the exact [measurement script](source-snapshot/measure.py). This is the captured source subset, not a claim of a complete transitive dependency archive.

The raw verification archive remains local at `output/reacher-memory-capacity-v1/publication-v1/reacher-memory-capacity-v1.tar.gz`; it is not asserted to be uploaded or publicly downloadable. Every archive member was reopened and checked for exact path, byte length and SHA-256. Binary checkpoints remain outside this evidence directory. [Receipt](receipt.json) binds the report, arithmetic, snapshot and archive. The recorded execution runtime says CPU and two Torch threads. [Post-run context](post-run-context.json) was collected while packaging and identifies an Apple M5 Max host and installed package versions; it was **not captured at execution** and does not establish execution-time host/runtime identity. No attempt was made to infer execution timestamps from file modification times.
