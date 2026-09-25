# Rust rollout comparison: prospective engineering protocol

This is a deployment comparison for the frozen [structured robot models](robot-structured-protocol.md). It cannot change that experiment's 61-condition rule. A faster implementation of an inaccurate model does not establish a useful architecture.

## Question

How much complete-request latency comes from eager Torch execution of tiny recurrent operations, and does a compact Householder transition remain cheaper than dense transitions after both receive native rollout kernels?

The native path is hybrid Python/Torch plus Rust. Python still normalizes inputs, validates and exports parameters, and prepares transition operators. Bounded dense models still compute their spectral norms on every request. Rust performs sequential rollout; the GRU comparator also conditions its recurrent state in Rust. There is no persistent prepared-parameter cache. This is not a comparison between programming languages in general or a pure Rust service.

## Admission and fixed roster

Require successful original structured-study process closure and independent audit, plus successful fabricated qualification of the exact native sources and compiled library. Preserve the failed first native qualification and subsequent repairs. Record compiler identity, build flags, source hashes, library hash, qualification receipts, inherited measurement pins and the closed study manifest before model execution.

Use all five fresh families: Householder, bounded dense, unbounded dense, bounded dense with MLP gate, and GRU32. Use all three seeds at each family's rate selected by the original study. There are 15 checkpoints, not a new model search. Missing or failed selected checkpoints fail admission. No fitting, weight edits, legacy-model substitution, reference refitting or target-based selection.

Read only the authenticated inherited FIT normalizers, initial linear coefficients, selected checkpoints and two already-exposed DEV recordings. Preserve checkpoint array layouts. Reconstruct the same 22 contexts per recording, context32 and horizon128. The first future torque is the input preceding the first predicted position. Measured torque remains a conditional input, not a verified issued action. Internal CONFIRM and official TEST stay closed; no raw measurement decoding or future-position scoring.

All requests carry 32 observed samples. The four structured transition models initialize from only the last two positions; GRU32 also incorporates earlier observations into hidden state. Timing preserves these original functions and cannot establish an equal-history comparison or a memory-quality advantage.

## Numerical gate before timing

For every selected fit and DEV recording, compare native and live Torch standardized forecasts and complete final recurrent states using `rtol=1e-5, atol=1e-5`. Also compare physical forecasts at the same tolerance. The unchanged native interface returns physical forecasts, so parity uses a second call with standardized inputs and identity normalizers to expose standardized outputs without recovering them through physical offsets. Compare both returned states. These additional parity calls are outside timing. All 30 fit/recording comparisons must pass. Retain the reference and native outputs, component errors and every failure. This threshold is fixed before any trained native-model evaluation; fabricated qualification uses the same threshold.

Each of the 30 records includes both the complete 22-window batch and the first-window batch1 request that will be timed. Both shapes must satisfy every forecast/state comparison at the same tolerance; agreement at batch22 alone does not establish agreement for a batch1 Torch kernel.

If any comparison fails, finish recording the finite parity roster and stop before timing. Nonfinite computation or schema failure is retained as a failed attempt. Do not widen tolerances, discard seeds or substitute approximate outputs. Source repairs require a separately named engineering attempt with the original failure retained.

## Complete-request timing

Only after the entire numerical gate passes, time batch1 on the first registered window of each DEV recording. Load models and the shared library outside timing. Each backend gets three warmups followed by 30 paired repetitions. Alternate backend order within each pair. Include physical-input normalization, float conversion, context conditioning, per-request validation and preparation, parameter export and packing, foreign-function overhead, rollout, physical-output conversion and finite-output checks. Retain all raw durations and backend order.

Use the original Torch inference equations with gradients disabled and cleared. Set Torch and the five CPU/BLAS thread controls to one. Start only after the structured training process has closed. Other host workloads are not stopped; record that the Apple Silicon host is shared. Report per-fit medians, all paired durations, and the family median of three fit medians separately for each recording. These are descriptive local latency measurements, not dedicated-host throughput or universal speed guarantees.

Report actual parameter, buffer, normalizer and explicit-state bytes. Native packed weights and input/output copies are per-request work, not hidden persistent storage. List their payload sizes separately; do not equate these with measured peak memory. Exclude shared-library code, Python objects, allocator workspace, optimizer state and model/disk loading consistently and explicitly.

Report each family's Torch/native speed ratio and Householder/native-dense and Householder/native-GRU ratios. Do not turn a favorable ratio into a new pass for the original experiment. Quality remains exactly the original audited development result; there is no new accuracy score, architecture promotion, biological-learning result or confirmation claim.

## Execution boundary

One exclusive output folder, one recorded process attempt, a 900-second cap, complete source/input/output manifests and no automatic restart. Authenticate the original study and source pins again at closure. Qualify the benchmark helper on fabricated data before recording its final pre-run registration. Publish failed qualification evidence alongside any successful comparison.
