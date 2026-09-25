# Native robot inference qualification

**Attempt 02 passed 110 fabricated tests with zero skips at unchanged float32 tolerances, rtol=atol=1e-5.** This package establishes tested inference parity on the recorded host. It does not establish a speedup, measured-data accuracy, or an architecture improvement.

| Original attempt | Scope | Outcome |
| --- | --- | --- |
| [01](qualification/attempt-01/qualification.json) | Four structured families | 88 passed, 2 failed, 0 skips |
| [02](qualification/attempt-02/qualification.json) | Four structured families plus GRU32 | 110 passed, 0 failed, 0 skips |

The two original failures were initialized Householder rollouts at horizons 128 and 512. Scalar accumulation differed from Torch's four-lane CPU reduction, accumulating enough error to exceed the frozen criterion. The [source-derived diagnostic](qualification/reduction-diagnostic-01/result.json) matched Torch on 1,024 fabricated reductions and all 16 initial reflection reductions with the four-lane order. Only the Householder dot-product accumulation was changed for the repair. The second attempt also added the separately implemented GRU reference. Neither the tolerance nor the original failure was removed.

The tests cover predictions and final states, active bounds, saturated gates, finite affine overflow through saturating activations, invalid inputs, causal/chunked execution, live parameter export, explicit copies, empty horizons, and ownership. This is a hybrid implementation: Python validates/prepares structured weights on every request, including dense spectral norms; Rust executes the sequential rollout. No prepared-weight cache is used. Storage/work counters describe explicit numeric payloads, not peak process memory.

## Evidence and exclusions

- [File manifest](manifest.json): every public payload, SHA-256 and byte count, excluding the manifest itself.
- [Provenance](provenance.json): original source paths and the 17 qualified current source pins.
- [Exclusions](exclusions.json): exact omitted filenames, hashes, reasons, and external source URLs.
- [Runtime setup](runtime/runtime-setup-01.json): Rust 1.98.1, aarch64-apple-darwin, original installation/version/smoke metadata.
- [Build definition](qualification/attempt-02/build/definition.json): source pins and exact compiler commands. Optimization was `opt-level=3`, with no fast-math or target-CPU override.
- [Original process receipts](qualification/attempt-02/qualification.json): lint, build, and test closure joins. Their absolute paths describe the original machine and have not been rewritten.

Compiled libraries, compiler archives/installations, downloaded installer scripts, and downloaded Torch implementation files are intentionally absent. Their exact hashes remain in the receipts and exclusion index. Torch sources remain available at the primary URLs recorded there. Repo-owned build/qualification/diagnostic sources are included. The runtime's installed-file manifest records the excluded toolchain contents.

## Rebuilding

A compatible Python/Torch environment and an explicit Rust compiler are external prerequisites. Restore the repository-owned sources at the paths recorded in provenance, then run the included `scripts/build_robot_native.py` from that repository with `--rustc /absolute/path/to/rustc --output /new/exclusive/build-folder`. Set `ROBOT_TRANSITION_LIBRARY` to the rebuilt library and run both native test files with the five thread variables in the attempt-02 preflight set to `1`.

Do not rerun the historical attempt scripts against their original output directories. A rebuilt library needs its own fresh qualification receipt. The reduction repair follows the recorded Torch 2.14 CPU/ARM implementation; portability and bitwise binary reproducibility across hosts or toolchains are not claimed. No future benchmark result is included here.
