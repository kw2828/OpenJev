# Author FSM baseline: engineering qualification

**Engineering PASS; no empirical author-baseline result.** The pinned author runtime and our causal state initializer passed fabricated qualification. No FSM measurements or supplied trained weights were numerically loaded for these checks. The real pooled FIT-only BLA28 fit remains forthcoming in this report's scope.

## Recorded outcomes

| Check | Original outcome | Evidence |
|---|---|---|
| Source/runtime preflight | PASS: installed author Python sources match the pinned vendor source; CPU and float64 verified | [Receipt](../output/fsm-author-engineering-v1/runtime-preflight-01/receipt.json) |
| Causal initializer qualification 01 | FAIL at Ruff: ten test-formatting errors; pytest did not run | [Original receipt](../output/fsm-author-engineering-v1/linear-context-qualification-01/receipt.json), [log](../output/fsm-author-engineering-v1/linear-context-qualification-01/command-01.log) |
| Causal initializer qualification 02 | PASS: Ruff and **52 fabricated tests**, 0.22 s pytest time, 0.514 s supervised test command; initializer source unchanged | [Receipt](../output/fsm-author-engineering-v1/linear-context-qualification-02/receipt.json), [test log](../output/fsm-author-engineering-v1/linear-context-qualification-02/command-02.log) |
| Author order-4 runtime qualification 01 | PASS on its first numerical execution; 6.395 s supervised process, exit 0 | [Receipt](../output/fsm-author-engineering-v1/author-runtime-qualification-01/receipt.json), [original process](../output/fsm-author-engineering-v1/author-runtime-qualification-01/process.json), [observed completion](../output/fsm-author-engineering-v1/author-runtime-qualification-01/process-observation.json) |

The runtime script also had one import-spacing Ruff precheck failure before numerical execution. Its [preserved transcript](../output/fsm-author-engineering-v1/author-runtime-qualification-01/prelaunch-lint-01.log) is explicitly labeled a transcription, with the original source snapshot beside it; the [corrected precheck](../output/fsm-author-engineering-v1/author-runtime-qualification-01/prelaunch-lint-02.log) passed. Neither numerical fixture nor numerical tolerance was changed after a failed numerical run. There was no numerical retry.

## What the synthetic run established

The fixed fixture used a stable, controllable and observable order-4 system, three inputs and outputs, 256 samples, two orthogonal excitation triplets and two identical periods. The source-derived contract and complete equations are in the [runtime contract](fsm-author-runtime-contract.md). Unweighted subspace initialization used `nx=4,nq=5`; maximum transfer-function error was **1.35e-15** against the independently constructed normalized reference, below the fixed `1e-6` absolute/relative tolerance.

The direct term was then deliberately perturbed by `0.02*I`. BFGS completed the fixed **25-iteration cap**, reducing independently recomputed frequency-response MSE from **1.33333e-4 to 3.33265e-8**. All returned model fields and recorded losses/times were finite. The author stop flag was **false**: this is a finite, improving partial optimization, not convergence or order-28 fit qualification. The author's `converged` field drops the Optimistix result code; even a true flag would require the narrower interpretation documented in the runtime contract.

Public simulation matched a separate NumPy output-before-update loop, including pre-update states, nonzero initial state, direct feedthrough and batches of one and three. The C100/H128 initializer used exactly 99 aligned observed input/output pairs; its largest normalized forecast difference was **2.22e-16**. A separate nonzero-mean, unequal-scale normalization witness differed by at most **3.55e-15**. Both checks used fixed `1e-10` absolute/relative tolerances. The final checkpoint roundtrip preserved all matrix, normalizer and scalar fields and public forecasts bitwise.

These are numerical interface checks. They do not establish FSM prediction accuracy, stability, model-selection quality or runtime competitiveness. The 6.395 s includes this small qualification's compilation, checks and serialization, not an empirical BLA28 training or deployment benchmark.

## Source, environment and independent review

The isolated component is [GPL-3.0-or-later](fsm_author/LICENSE), with [source/dependency notices](fsm_author/THIRD_PARTY.md). It integrates [`freq-statespace` commit a79e8c567b018a6c9462528fc1e10b77fd19b3e2](https://github.com/merijnfloren/freq-statespace/tree/a79e8c567b018a6c9462528fc1e10b77fd19b3e2); author recipe provenance is the [FSM notebook commit 539a12fef384b086a8562b500498b2fa3899ef70](https://github.com/merijnfloren/fsm-benchmark-data/blob/539a12fef384b086a8562b500498b2fa3899ef70/baseline_results/fit_BLA.ipynb). This component does not change unrelated OpenJev licensing. FSM data licensing is separate, CC BY 4.0; no measurement access was needed here.

The isolated runtime used Python **3.12.13**, JAX/JAXlib **0.11.2**, Equinox **0.13.8**, Optimistix **0.1.0**, NumPy **2.5.3**, SciPy **1.18.1**, and author package **0.1.2**, on macOS ARM64 CPU with JAX x64 enabled. Exact resolved dependencies are in [uv.lock](fsm_author/uv.lock), SHA `d7d1c8c38bbcf910bcede299277a21cd58b9da0ad56c7ae9ec75d85dbadcb2a3`. The standalone 52-test NumPy suite used the repository `.venv`, as its original command records state; author integration used `research/fsm_author/.venv`.

Independent source review covered the initializer's causal convention and the fabricated system's rank, the normalized frequency-response oracle, explicit public simulation axes, partial-stop interpretation and checkpoint checks before launch. Reviewed source identities are [linear_context.py](fsm_author/src/openjev_fsm_author/linear_context.py), SHA `342f0656de39f84fb0070f9253c05ac767a8d348519ff608c130bdbdeeb62267`, and [qualify_runtime.py](fsm_author/scripts/qualify_runtime.py), SHA `31857a0c09334d1d9505da73f1f13c6966a996c6084666b68d3ae8ed0df0b216`. After closure, an independent opaque-byte review confirmed the original process, command/environment, definition, source snapshots and every retained-file/log hash. Neither review reran the optimizer or decoded its arrays; this is not a second independent optimizer implementation.

## Reproducibility and preserved records

Commands are relative to the OpenJev repository root. These identify the completed attempts; their exclusive output directories must not be reused:

```sh
research/fsm_author/.venv/bin/python research/fsm_author/scripts/runtime_preflight.py --output output/fsm-author-engineering-v1/runtime-preflight-01
research/fsm_author/.venv/bin/python research/fsm_author/scripts/qualify_runtime.py --prepare --output output/fsm-author-engineering-v1/author-runtime-qualification-01
research/fsm_author/.venv/bin/python output/fsm-author-engineering-v1/author-runtime-qualification-01/launch.py
```

The [frozen definition](../output/fsm-author-engineering-v1/author-runtime-qualification-01/definition.json) contains the exact executed command, source/snapshot pins, fixed fixture, tolerances and environment. Its supervisor enforces 600 s, sets `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=True` and five recorded thread variables to `1`, and retains original stdout/stderr and process exit. The numerical script forbids network connections and external numerical/archive files after imports. Any future repeat uses a fresh output directory and a separately retained definition, rather than overwriting these records.

Receipt SHA-256 values: runtime preflight `4ee148d93eb39c41e2edd3ded97134398f6bc03d991375ff8cb5124a20a6ecc0`; initializer PASS `7cdbd61c84b775db87bafc0a3e52d58270abe6695cb1d6740e06cd00fb15fd64`; author runtime PASS `d054005dcbbdadcf4746c2f216cf62abc909ede3e8bb16c458691c0588fde84b`. Original source snapshots, synthetic inputs, initial/perturbed/final checkpoints, solver trace, parity arrays and failed prechecks remain alongside the receipts. Source and recorded-file hashes were checked unchanged after execution.

The next empirical reference is the separately specified **FIT-only BLA28** using the qualified causal initializer. This report supplies engineering admission evidence, not its result and not an author-score reproduction. NL-LFR reproduction remains a separate qualification task.
