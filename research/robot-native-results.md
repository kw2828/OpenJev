# Trained native robot inference: stopped before timing

**The registered native benchmark fails numerical admission. No speed measurements were taken.** All 30 selected-fit/recording cases were retained. The independent audit agrees with the failure, checking 420 saved arrays and 240 comparisons without running a model.

[Complete evidence](robot-native-results/manifest.json) · [Saved-output audit](robot-native-results/audit/audit.json) · [Original outcome](robot-native-results/study/summary.json) · [Protocol](robot-native-protocol.md) · [Registration](robot-native-registration.json) · [Earlier synthetic qualification](robot-native-qualification-results/README.md).

## Where parity failed

All standardized forecasts and final-state comparisons pass the unchanged `rtol=atol=1e-5` threshold. The failures occur after conversion to physical output units. All 30 batch22 physical-output checks fail; eight of 30 batch1 physical-output checks pass. Because every case requires both batch shapes and every output check to pass, the overall result is 0/30 eligible cases and zero timing requests.

| Family | Complete cases passing | Max standardized forecast difference | Max physical output difference, degrees |
|---|---:|---:|---:|
| Householder | 0/6 | 0.000005722 | 0.000176368 |
| Bounded dense | 0/6 | 0.000005245 | 0.000152535 |
| Unbounded dense | 0/6 | 0.000004292 | 0.000169492 |
| Dense with MLP gate | 0/6 | 0.000004053 | 0.000162068 |
| GRU32 | 0/6 | 0.000001907 | 0.000076267 |

These are maximum differences between two implementations of the same fixed model, across both batch shapes. They are not errors against measured future positions. All outputs are finite and there are no backend exceptions. Of the 240 component checks, 180 standardized-forecast/state checks and eight physical-output checks pass.

The fixed absolute tolerance was applied in two different unit systems. Multiplying a standardized forecast by its position scale also multiplies floating-point differences; adding a physical offset changes the relative-tolerance term. The original contract is therefore not invariant to normalization. This explains why passing the standardized checks does not imply passing physical-output checks. It does not establish a material loss in forecasting quality, which this engineering experiment did not score.

The original threshold and failure remain unchanged. Any future comparison needs a separately registered numerical contract that states physical units and how normalization transforms its error bound. Timing a subset or silently relaxing this run's threshold would not satisfy its protocol.

## Scope and evidence

This uses 15 immutable selected checkpoints from the [structured experiment](robot-structured-results.md), covering five families, three seeds and two already-exposed DEV recordings. Each case checks all 22 forecast windows plus the actual batch1 request intended for timing. Each request uses context32 and horizon128. The structured models initialize from the last two observed positions; GRU32 also incorporates the prefix. Future inputs are measured torques, not authenticated issued actions.

The implementation is hybrid Python/Torch and Rust. Parameters are prepared and exported on each request; dense spectral norms remain included. A hypothetical speed comparison would retain that work, but this run never reaches warmups or timing. There is no Rust speedup or architecture-quality claim.

The original benchmark process closes in 2.23 seconds with return code1 and unchanged source/input pins. Its registration was committed before execution at `36838d080e7020b00f0ac1546f50ca1097b275f3`. The earlier fabricated kernel qualification passed 110 cases after a preserved 88/90 first attempt; the benchmark helper passed 22 fabricated tests and its independent auditor passed 24. The actual saved-output audit passes on its first invocation. **Audit PASS means the recorded failure was reproduced, not that the benchmark passed.**

The package preserves every original forecast and final state, scalar comparison, process receipt, qualification log and source snapshot. It includes no source measurements, future-position targets or compiled library. Source measurements remain external under the parent study's exact pins and licensing. Internal CONFIRM and official TEST stay closed.
