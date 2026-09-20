# Paired objective comparison: completed, continuation failed

The six-fit study launched once on **20 September 2026 at 10:52 UTC**, after
the coordinated CPU allocation became available. All six fits completed in
**4,480.1063 seconds (74.67 minutes)** before quality scoring. The saved-output
reporter and independent primary audit agree: **continuation failed, with
6 of 13 checks passed**. See the [completed results](dialogue-objective-results.md)
for every requirement and seed variation. Frozen inputs and all 56 scientific
source files remained unchanged; there was no retry.

The [published protocol](dialogue-objective-protocol.md) compares ordinary
row-uniform training with the existing stratum-weighted objective in the same
token-aligned model. Six fresh fits use three paired seeds, identical initial
tensors within each pair and the same saved training orders. Four fixed
readouts separate changing the training objective from changing scores after
training. Historical corrected flat, mean and aligned controls remain visible.

Continuation requires all thirteen checks, including at least a two-point
changed-accuracy gain over the fresh corrected stratum control, without worse
retention error or overall log loss, under both row and equal-service weighting.
The practical checks also require improvement over historical corrected flat.
These are development requirements; a pass would support a training choice,
not establish a new architecture or untouched confirmation.

The [result plotter](../scripts/plot_dialogue_objective.py) displays all three
seeds for every fixed fresh and
historical readout, with separate panels for changes, retention, overall
accuracy and log loss under both weighting schemes. One
[synthetic layout check](../output/dialogue-objective-v1/figure-preflight-01/receipt.json)
verified the 240 displayed seed values and readable PNG/PDF output. Those
invented values are prominently marked and are not research results. The
plotter reads saved report summaries only. The completed independent primary
audit agrees on 3,566 scalar comparisons, 60 primary cells, 30 paired cells
and all thirteen decisions; it does not replay training or model inference.

## What is verified

- **17 runner cases** cover paired initialization, loss and gradient arithmetic,
  tail batches, row orders, public-input isolation, completion and failure
  handling. [Latest receipt](../output/dialogue-objective-v1/preflight-03/receipt.json).
- **22 reporter cases** cover the four readouts, thirteen decision rules,
  synthetic runner compatibility and saved report output.
  [Receipt](../output/dialogue-objective-v1/report-preflight-02/receipt.json).
- **20 independent-auditor cases** cover separate correction arithmetic,
  primary metrics, rules, authentication and rejection of invalid technical
  claims. [Latest receipt](../output/dialogue-objective-v1/audit-preflight-02/receipt.json).
- A [static integration check](../output/dialogue-objective-v1/source-preflight-01/receipt.json)
  verifies the source closure and test receipts. Independent source review
  found no remaining blocker.

Those 59 focused cases passed across the preserved attempts, not in a single
combined invocation. Initial fixture and lint failures remain in the earlier
receipt directories. The final runner differs from its tested version only
by its source-file inventory; deleting those added inventory lines restores
the exact reviewed source hash. The static check records that equality.

The metadata-only freeze completed in **4.635 seconds**, with **555.14 MiB**
process-lifetime peak RSS, **64 files** and **16,101,019 bytes**. It binds
56 source files, 29,211 fitting rows and 13,599 evaluation rows. It made zero
model or encoder calls. Its inputs remain exposed official TRAIN development
data, with the correct previous value supplied; no autonomous memory or
official DEV inference is measured.

| Frozen artifact | SHA256 |
|---|---|
| [Plan](../output/dialogue-objective-v1/protocol-01/plan.json) | `ea5660f97f75256246808b0971e80ab3ec086baa614ea2080c893716838d4baa` |
| [Freeze completion](../output/dialogue-objective-v1/protocol-01/completed.json) | `c2ea94560792c9d6b84e8e6f10fb0aef2a3135918c17bbeb09334ff0b3c3f6ea` |

## Completed execution

The single run stayed within the frozen whole-run limits of **6,000 seconds,
6 GiB peak RSS and 512 MiB output**. Actual time was 74.67 minutes, peak
process-lifetime RSS was 1,901,625,344 bytes, and execution output was
73,697,480 bytes across 32 files. The earlier 76-minute figure was a historical
estimate, separate from this measurement.
The launch command is recorded below for provenance, not as a retry instruction.

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python scripts/study_dialogue_objective.py train \
  --plan output/dialogue-objective-v1/protocol-01/plan.json \
  --plan-sha256 ea5660f97f75256246808b0971e80ab3ec086baa614ea2080c893716838d4baa \
  --out output/dialogue-objective-v1/training-01
```

All six final fits were saved before quality scoring, with 13,800 optimizer
updates completed. The [report](../output/dialogue-objective-v1/report-01/report.md)
and [independent audit](../output/dialogue-objective-v1/audit-01/receipt.json)
are complete. No replacement seed, outcome-based checkpoint selection or retry
occurred. Completed technical validation does not override the failed
scientific rule.

The earlier token-alignment result remains **FAIL, 18/22**. The subsequent
[fixed weight correction](dialogue-weight-prior-results.md) improves retention
while missing more real changes; it does not overturn that result. The broader
goal of a strong novel architecture remains unachieved.

The [source-specific memory note](dialogue-source-memory-decision.md) records
one conditional fallback and its established prior work. It admits no new
experiment. The objective did not satisfy the prescribed tradeoff; that
failure neither proves source confusion nor automatically admits the fallback.
