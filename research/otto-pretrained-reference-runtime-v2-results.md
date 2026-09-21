# Corrected OTTO runtime: values agree, five actions differ

21 September 2026. **All numerical work completed, but the NumPy policy still fails equivalence.** All eight weight tensors, policy inputs and branch masses match exactly. Every value and action cost passes the unchanged numerical tolerance. Selected actions agree in **31 of 36** fixed policy fixtures. Five disagreements at TensorFlow ties keep the overall qualification false.

This is an engineering compatibility result, not search accuracy or model effectiveness. Use the **original TensorFlow model and original RLPolicy** for the next native-adapter check. Do not substitute the NumPy policy, broaden its tie tolerance or discard the disagreeing fixtures.

## What changed

The [first attempt](otto-pretrained-reference-results.md) stopped on a matrix-multiplication warning before any value/action comparison completed. A separate diagnostic reproduced spurious warnings on exact finite primitive answers. The new environment preserves every original distribution except NumPy, updated from 2.2.6 to 2.5.3. It remains CPU-only, with Python 3.12.13, TensorFlow 2.20.0 and tf-keras 2.20.1.

The [new protocol](otto-pretrained-reference-runtime-v2-protocol.md), [plan](../output/otto-pretrained-reference-v2/qualification-plan-01.json) and thin runtime wrapper were committed at `ebcc4e9` before the first new checkpoint call. The original inference code, fixtures, tolerances, action rule and resource limits are unchanged. No warning is suppressed in the port. The first preflight's test collection lacked the repository import path; that failure is preserved. With the explicit source path, **112 synthetic tests passed under the actual new runtime**, alongside eight exact warning-free primitive checks. Neither preflight loaded checkpoint weights.

## Complete comparison

| Requirement | Outcome |
|---|---|
| Original TensorFlow weights versus extracted tensors | 8/8 byte-identical |
| Fixed raw value routes | All 46 batches, 96 scalar predictions within tolerance |
| Policy centered inputs and branch masses | Exact identity on all 36 fixtures |
| Policy values and four action costs | Within tolerance on all 36 fixtures |
| Exact selected action, physical fixtures | 31/32 agree |
| Exact selected action, mechanical floor fixtures | 0/4 agree |
| Overall policy qualification | **FAIL** |

The unchanged numerical rule is `abs(actual-reference) <= 1e-4 + 1e-5 * abs(reference)`. The unchanged action rule selects the first cost within strictly `1e-10` of the minimum. Agreement within a value tolerance does not imply agreement under that much tighter action rule.

| Disagreeing fixture | TensorFlow action | NumPy action | TensorFlow tied minimum actions |
|---|---:|---:|---|
| Baseline, opposite corner, uniform belief | 0 | 2 | 0, 2 |
| Mechanical branch mass 0 | 1 | 3 | 1, 3 |
| Mechanical branch mass `0.5e-10` | 1 | 3 | 1, 3 |
| Mechanical branch mass `1e-10` | 1 | 3 | 1, 3 |
| Mechanical branch mass `2e-10` | 1 | 3 | 1, 3 |

In the physical fixture, the NumPy costs for actions zero and two differ by about **7.63e-6**, while TensorFlow returns equal costs. In each mechanical fixture, NumPy favors action three over one by about **1.91e-6**, while TensorFlow ties them. These small differences explain the recorded choices without exempting them from the exact-action requirement. All fixtures remain in the result, and no runtime or tolerance was selected again after this outcome.

## Execution and preserved evidence

The worker completed all **82 TensorFlow and 82 NumPy forwards**, with every attempted call returned and no unresolved work. Construction, graph build and loading are recorded separately. There were zero simulator calls, training updates or native actor constructions. Worker time was **4.4842 seconds**, enclosed by **4.7510 seconds** of supervisor time. Peak worker RSS was **1,077,673,984 bytes**, within the fixed 4 GiB limit.

The worker and supervisor exited successfully and are absent; complete file closure and all source, input and runtime-package identities were checked afterward. Exit zero means the comparison completed, not that equivalence passed. The saved `qualified` flag remains false.

An independent saved-output reader confirmed complete coverage, all call counts, 42 source pins, the original payload closure and **6,350,976 byte-identical saved input/mass scalars**. It independently recomputed value/cost tolerances and every first-action choice. Maximum absolute differences were **3.8147e-5** for raw values, **4.5776e-5** for policy values and **2.2888e-5** for action costs. It confirms all five action disagreements and the failed qualification. The audit performed no inference or simulator calls; actual model/extraction behavior and timing truth remain inherited. The actual NumPy-call input witness was not separately saved, so that particular producer check is explicitly inherited rather than independently reconstructed.

- [Complete summary](../output/otto-pretrained-reference-v2/qualification-01/summary.json), [raw value comparisons](../output/otto-pretrained-reference-v2/qualification-01/value-checks.json) and [all policy comparisons](../output/otto-pretrained-reference-v2/qualification-01/policy-checks.json).
- [Worker receipt](../output/otto-pretrained-reference-v2/qualification-01/receipt.json), [work ledger](../output/otto-pretrained-reference-v2/qualification-01/work-ledger.jsonl), [supervisor terminal](../output/otto-pretrained-reference-v2/qualification-process-01.terminal.json) and [execution witness](../output/otto-pretrained-reference-v2/execution-witness.json).
- [Independent audit receipt](../output/otto-pretrained-reference-v2/audit-01/receipt.json) and [recomputed results](../output/otto-pretrained-reference-v2/audit-01/summary.json).
- [Losslessly compressed policy arrays](../output/otto-pretrained-reference-v2/array-delivery-01/policy-arrays.npz.gz) and [hashes/restoration instructions](../output/otto-pretrained-reference-v2/array-delivery-01/manifest.json). Decompression restores the original 53,142,662-byte NPZ. The original HDF5 remains available from its authenticated upstream source rather than duplicated in Git.

Native/public-adapter qualification and autonomous control performance remain separate outstanding requirements. The [native integration protocol](otto-released-native-qualification-protocol.md) preserves all four actions, the original filtering threshold and evaluator-only access to hidden state. Passing that future check would establish integration, not a novel learned architecture advantage.
