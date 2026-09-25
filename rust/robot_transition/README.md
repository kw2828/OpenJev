# Native robot rollout kernels

Inference-only Rust kernels for OpenJev's four structured transition families and its GRU32 reference. The Python wrappers keep the original Torch weights and export them on each request. They do not train, change checkpoints or cache prepared operators.

- Structured models: Python/Torch validates parameters and prepares operators, including dense spectral norms; Rust executes the gate, transition and forcing recurrence.
- GRU32: Rust conditions the 32-value hidden state from the observed prefix, then rolls out the original residual model. The complete explicit state has 50 values.
- Inputs remain observed positions and measured torques. These kernels do not turn the forecasting experiment into a robot controller.

The first compiled qualification passed 88/90 cases. Initialized Householder rollouts at horizons 128 and 512 differed beyond the fixed `rtol=atol=1e-5` tolerance. A source-based diagnostic identified Torch's four-lane float reduction order on this ARM host. Changing only the two reflection reductions resolved those failures. The combined suite then passed all 110 fabricated checks at the same tolerance, including GRU conditioning, causal alignment, long rollouts, saturated activations and live weight refresh.

This establishes numerical qualification on the tested Apple Silicon host, not a speed result or a cross-platform guarantee. The [native benchmark protocol](../../research/robot-native-protocol.md) requires another parity gate on every selected trained checkpoint before any timing.

## Build and test

Use an explicit Rust compiler path and a new output directory. The build helper snapshots sources, records compiler identity and preserves process receipts. It has no dependency downloads or automatic toolchain discovery.

```sh
.venv/bin/python scripts/build_robot_native.py \
  --rustc /absolute/path/to/rustc \
  --output output/native-build-01

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
ROBOT_TRANSITION_LIBRARY="$PWD/output/native-build-01/libopenjev_robot.dylib" \
.venv/bin/python -m pytest -q \
  tests/test_native_robot_transition.py tests/test_native_robot_gru.py
```

The example library suffix is for macOS. Tests require an explicit library and fail if it is missing. The qualified build uses Rust 1.98.1, edition2021, `opt-level=3`, and no fast-math flags. Compilation and library loading are excluded from request timings; parameter packing and operator preparation are included.

`NativeRobotTransition.request()` and `NativeRobotGRU.request()` accept float64 physical inputs and return a float64 physical forecast plus an owned float32 final state. Standardized rollout methods support explicit carried state and an empty horizon. Returned array-work counts describe payloads, not measured peak memory.
