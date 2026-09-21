# Released OTTO model: compatibility check stopped

21 September 2026. **The first numerical qualification failed and remains unqualified.** All eight tensors loaded through the original TensorFlow model match the extracted NumPy tensors byte-for-byte. The first NumPy forward raised `FloatingPointError: overflow encountered in matmul`, before any value or action comparison completed. This establishes weight identity only, not policy equivalence or search performance.

The [protocol](otto-pretrained-reference-protocol.md) and [plan](../output/otto-pretrained-reference-v1/qualification-plan-01.json) were frozen and published at `674e3ec` before execution. The run used the authenticated official 13,390,849-parameter checkpoint, unchanged original inference source and isolated CPU TensorFlow 2.20.0 / legacy Keras 2.20.1 / NumPy 2.2.6 runtime. It started only after the [action-head experiment](otto-action-head-results.md) had terminated, preserving that experiment's timing conditions.

| Work | Attempted | Returned |
|---|---:|---:|
| TensorFlow construction, build and weight loading | 1 each | 1 each |
| NumPy construction | 1 | 1 |
| TensorFlow raw value forward | 1 | 1 |
| NumPy raw value forward | 1 | 0 |
| Completed tensor identity comparisons | 8 | 8 |
| Completed value / policy comparisons | 0 / 0 | 0 / 0 |
| Native simulator calls / training updates | 0 / 0 | 0 / 0 |

The failing case was the first fixed all-zero 105x105 input, batch size one and symmetry averaging disabled. The traceback does not identify the individual layer. The TensorFlow return was not persisted before the NumPy exception, so its value and finiteness cannot be independently checked from saved outputs. The append-only work ledger preserves the returned TensorFlow call and unresolved NumPy call.

## What the exception does and does not tell us

Static bounds from the authenticated tensor metadata place even a conservative final magnitude bound for the zero fixture near `1.27e10`, far below float32's approximately `3.40e38` limit. That makes legitimate activation-magnitude overflow unlikely under correct matrix multiplication. It does not prove what NumPy returned, because this call raised before returning an array.

The earlier synthetic port tests used NumPy 2.5.3; this qualification used NumPy 2.2.6. Both installations identify Apple Accelerate. NumPy's maintainers have documented similar spurious matrix-multiplication warnings on M4 systems in [issue 28687](https://github.com/numpy/numpy/issues/28687) and [issue 29820](https://github.com/numpy/numpy/issues/29820).

A separate [frozen primitive diagnostic](../output/otto-pretrained-reference-v1/matmul-diagnostic-01/plan.json) then ran exactly twelve matrix multiplications with fixed zero/constant arrays and analytically exact answers. It used both hidden-layer matrix shapes, with no checkpoint reads or model calls. All twelve results were finite and exactly correct:

| Runtime | Calls | Exact finite results | Calls emitting numerical warnings |
|---|---:|---:|---:|
| NumPy 2.2.6, before TensorFlow import | 4 | 4 | 4 |
| NumPy 2.2.6, after TensorFlow import | 4 | 4 | 4 |
| NumPy 2.5.3, fresh process | 4 | 4 | 0 |

This reproduces spurious warnings in the isolated NumPy/Accelerate stack without needing TensorFlow. It supports a runtime explanation, but does not establish the failed checkpoint's outputs or parity. The diagnostic completed once with both children exiting zero; the original runtime and failed qualification were unchanged. [Diagnostic receipt and file hashes](../output/otto-pretrained-reference-v1/matmul-diagnostic-01/receipt.json) · [2.2.6 results](../output/otto-pretrained-reference-v1/matmul-diagnostic-01/isolated/receipt.json) · [2.5.3 results](../output/otto-pretrained-reference-v1/matmul-diagnostic-01/main/receipt.json).

The frozen qualification is not rerun, its tolerances are unchanged, and no warning is suppressed in the frozen port. A separately pinned runtime with the corrected primitive behavior would require new numerical qualification before use. Native adapter integration and autonomous evaluation of this released model remain outstanding. The original TensorFlow implementation remains a possible comparator; neither implementation has a new control-quality result here.

## Preserved evidence

The supervisor exited with code one after **4.0566 seconds**, with the worker group absent. This was an exception, not a timeout. Peak worker RSS was **1,038,303,232 bytes**. Post-failure readback confirmed every frozen source and input hash unchanged.

- [Failed worker receipt](../output/otto-pretrained-reference-v1/qualification-01/receipt.json), [actual work ledger](../output/otto-pretrained-reference-v1/qualification-01/work-ledger.jsonl) and [eight tensor comparisons](../output/otto-pretrained-reference-v1/qualification-01/weight-checks.json).
- [Supervisor terminal](../output/otto-pretrained-reference-v1/qualification-process-01.terminal.json), [log](../output/otto-pretrained-reference-v1/qualification-process-01.log) and [independent failure closure](../output/otto-pretrained-reference-v1/qualification-failure-closure-01.json).
- [Actual runtime](../output/otto-pretrained-reference-v1/qualification-01/runtime.json), [reference provenance](otto-learned-reference-and-symmetry.md) and [upstream license notices](../third_party/otto/README.md).

The original HDF5 copy and extracted weights remain retrievable from the pinned upstream source; they are not duplicated in Git. Failure closure records every local output, including the original HDF5 copy. No architecture, speed or gameplay claim follows from this compatibility attempt.
