# Native query-gate integration qualification

**PASS for integration:** all 18 fixed short probes completed with the original neural policy and the public-state gate connected correctly. This admits a learned-gate experiment; it establishes no learning benefit or full-horizon control competence.

| Fixed mode | Cases | Actual steps | Policy queries | Actual TensorFlow forwards |
|---|---:|---:|---:|---:|
| Always query | 6 | 56 | 56 | 112 |
| Never query | 6 | 62 | 0 | 0 |
| Alternate queries | 6 | 56 | 29 | 58 |
| **Total** | **18** | **174** | **85** | **170** |

Each policy query triggers two actual forwards: the gate backend and a separate original-policy reference calculation. The 170 total therefore contains **85 reference forwards**. Never-query makes none. These totals describe this instrumented qualification, not deployment cost or a speed comparison.

## What passed

The six cases combine sensing length 3 with seeds 1110001-1110003 and sensing length 4 with seeds 1120001-1120003; initial hits are respectively 1, 2 and 3. Each case is reused across all three fixed modes, stopping on discovery or after 16 steps. The alternating mode carries a counter, querying on steps 1, 3, 5 and so on. It is a fixed schedule, not a learned recurrent model.

The original run recorded **3,033 comparisons**. It required exact analytic score/action agreement on every decision, exact original restricted-neural score/action agreement whenever queried, and byte-identical native/public posterior arrays after reset and every update, including terminal updates. Independent reference actors only scored the actual shared history. The gate received its 31 public features and carried state before the current neural calculation. Callable purity remains a trusted interface contract, not a process sandbox.

A separate [saved-log publication check](../output/otto-query-gate-v1/publication-check-01.json) verifies all 18 case identities, both operation journals, every query schedule, paired source/overlapping-uniform witnesses, all 12 payload hashes, all 99 frozen sources, three metadata inputs, eight native inputs and the original parent-process closure. It reconciles 1,392 outer/nested operations, 817 gate operations, 192 reset/step witnesses and 170 recorded forwards. **It does not recompute posterior arrays, neural values or the 3,033 original numerical comparisons**; those remain evidence from the completed native probe.

## Cost and limits

The worker took **11.068282 s**; the original supervising process, including exit and cleanup, took **11.307257 s**. Peak worker RSS was **761,757,696 bytes**. The 12 payloads total **2,649,691 bytes**, excluding the receipt. Physical setup took 3.867644 s and the probe loop 6.899346 s; nested call timers overlap and should not be summed. The inherited `setup.json` allocation over 192 episodes is historical metadata, not an allocation for these 18 probes.

The frozen cap was 180 seconds, one numerical thread, 4 GiB RSS and 64 MiB output, with at most 18 resets, 288 steps and 288 actual TensorFlow policy forwards. The run used the unchanged 13,390,849-parameter checkpoint and made **zero training updates**. All pending operations were closed; the original supervisor exited successfully without timeout and confirmed the process group was absent.

These short paths do not estimate full-horizon success, a compute-utility frontier, the value of learned recurrence, or an architecture advantage. The next scientific comparison needs a separately frozen learned recurrent gate versus a matched stateless gate and the fixed analytic/neural endpoints. See the [protocol](otto-query-gate-native-protocol.md), [qualifier](../scripts/qualify_otto_query_gate_native.py) and [all saved cases](../output/otto-query-gate-v1/native-run-01/episodes.jsonl).

| Evidence | SHA-256 |
|---|---|
| [Frozen plan](../output/otto-query-gate-v1/native-plan-01.json) | `65d92b9f6c54ea29218c9b3f904af3579f7817b2072bee2696e76cea643f1d15` |
| [Worker receipt](../output/otto-query-gate-v1/native-run-01/receipt.json) | `9196580258c04b9e38b058085a33bd7964c2116613363822262adbdf75055f5b` |
| [Original parent terminal](../output/otto-query-gate-v1/native-supervision-01.terminal.json) | `9b336c339ff9dab9feca55df1c225622dc26170ba283fef5529a18a9df7c89dc` |
