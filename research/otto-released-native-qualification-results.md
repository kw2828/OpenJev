# Original OTTO policy passes the public-input integration check

21 September 2026. **The original TensorFlow value model and published policy give identical decisions through OpenJev's public-input adapter on all eight fixed comparisons.** Its belief agrees exactly with the native environment at every reset and all 446 prescribed transitions. This establishes a working reference integration, not autonomous search competence, a new architecture or a biological-learning advantage.

The [NumPy port's five action disagreements](otto-pretrained-reference-runtime-v2-results.md) remain a failed result. This experiment uses the original TensorFlow model on both sides; it does not loosen that earlier action rule or substitute the port.

## What passed

| Requirement | Completed result |
|---|---|
| Original loaded weights versus extracted tensors | All eight byte-identical |
| Native/public float64 belief | Exact at eight resets and 446 transitions |
| Paired policy inputs, branch masses, values and float32 costs | All eight pairs byte-identical |
| Selected action under the unchanged first-action rule | All eight pairs agree |
| Blocked moves | All four directions checked in both sensing regimes |
| Censoring | Six paths incorporate the final nonterminal observation |
| Found-source lifecycle | Both paths produce the terminal point mass and reject further actions/updates without model work |
| Work accounting | Eight resets, 446 steps, sixteen model forwards, sixteen public actors and 510 public updates; every attempt returned |

The actor receives only current public position, hit, termination, step and boundary metadata, plus the known observation kernel. It owns its posterior and never receives the native environment, sampled source, native posterior or random seed. All four actions remain available, including blocked directions that stay in place and obtain another observation.

These are **mechanical fixtures**. Source positions and observations are prescribed by the evaluator to exercise specific transitions. Six four-step paths cover each positive initial hit in both regimes; two 211-step paths cover boundaries and finding the source. The paired policy checks occur at fixed public prefixes. No autonomous episodes or training updates were run, and the two terminal paths are not a measured success rate.

## Execution and evidence

The [protocol](otto-released-native-qualification-protocol.md), [exact plan](../output/otto-released-native-v1/qualification-plan-01.json), adapter and runner were committed and pushed at `a3101f5` before execution. The plan binds 35 files, eight input roles and all 43 runtime distributions. It preserves the prior reference environment and uses a separate CPU-only runtime with the same distributions plus SciPy 1.18.1: Python 3.12.13, NumPy 2.5.3, TensorFlow 2.20.0 and tf-keras 2.20.1. This does not reproduce the historical TensorFlow 2.8 runtime. Before this run, 62 synthetic tests passed in that runtime.

The one permitted run completed in **4.7153 worker seconds**, enclosed by **5.1448 supervisor seconds**, with **761,479,168 bytes** peak worker RSS. Its sixteen payloads total **13,006,472 bytes**. It stayed within the fixed 600-second, 4-GiB and 64-MiB limits. Construction, graph build and loading each occurred once; all sixteen policy forwards are recorded separately. The worker and supervisor exited successfully and are absent. Source, input, package and payload identities remain unchanged, with no late failure.

- [Summary](../output/otto-released-native-v1/qualification-01/summary.json), [all policy comparisons](../output/otto-released-native-v1/qualification-01/policy-checks.jsonl), [all transitions](../output/otto-released-native-v1/qualification-01/transitions.jsonl) and [per-case results](../output/otto-released-native-v1/qualification-01/cases.jsonl).
- [Complete work ledger](../output/otto-released-native-v1/qualification-01/work.jsonl), [loaded weight checks](../output/otto-released-native-v1/qualification-01/weights.jsonl), [runtime](../output/otto-released-native-v1/qualification-01/runtime.json) and [worker receipt](../output/otto-released-native-v1/qualification-01/receipt.json).
- [Supervisor terminal](../output/otto-released-native-v1/qualification-process-01.terminal.json), [actual execution witness](../output/otto-released-native-v1/execution-witness.json) and [synthetic tests](../output/otto-released-native-v1/runtime-tests-01/tests.log).

The eight `prefix-*.npz` files alongside these receipts preserve both views' actual model inputs, branch masses, values and costs. Weight files remain authenticated external inputs under the [upstream model provenance](otto-learned-reference-and-symmetry.md).

## Independent saved-output audit

The [independent reader](../output/otto-released-native-v1/audit-01/audit.py) reconstructed **all 454 public belief snapshots** and the **eight policy input/branch-mass batches** from the saved public history and kernels. Every reconstructed belief hash, input and mass agrees exactly. It also verifies the native/public saved values and costs byte-for-byte, the original first-action choices, all operation counts, the complete payload set and source/input/runtime/parent identities. Its **2,839,963 checks agree** with the producer result.

The audit completed in 0.5616 seconds with 51,888,128 bytes peak RSS and zero model or simulator calls. Supplemental cost arithmetic in float64 differs by at most 9.404e-6, within the stated float32 reduction-roundoff bound. This does not relax the exact native/public cost or action requirements. Actual TensorFlow execution, loaded-weight truth and the producer's native-posterior equality assertions remain authenticated producer evidence; the auditor independently reconstructs the public side.

[Audit summary](../output/otto-released-native-v1/audit-01/result-01/summary.json) · [Audit receipt](../output/otto-released-native-v1/audit-01/result-01/receipt.json) · [Actual invocation](../output/otto-released-native-v1/audit-01/invocation.json).

## What this permits next

A fresh, frozen autonomous comparison can now use the original released policy through the same public interface as our models. Compare it with the strongest full-belief analytic controller under the same observation kernels, all-four-action contract, episode horizons and paired initialization. Measure search utility and complete decision/update computation. Training or compact recurrent-model claims still need that stronger competence reference and separate untouched evaluation; this integration pass supplies neither result.
