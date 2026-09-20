# Independent primary-result audit

Prepared before fitting. Do not run against an incomplete study or before the
parent task supplies final execution and report pins.

From the OpenJev checkout:

```sh
.venv/bin/python output/dialogue-token-alignment-scientific-v1/audit-01/audit.py \
  --run RUN_DIRECTORY \
  --plan-sha256 FINAL_PLAN_SHA256 \
  --completed-sha256 FINAL_EXECUTION_COMPLETED_SHA256 \
  --report REPORT_DIRECTORY \
  --report-receipt-sha256 FINAL_REPORT_RECEIPT_SHA256 \
  --out output/dialogue-token-alignment-scientific-v1/audit-01/result-01
```

The output directory must not exist. One invocation has a 60-second wall limit.
A mismatch preserves a failed receipt and any completed audit payloads. Do not
retry silently or replace the failed directory.

The auditor independently recomputes all nine fits' held-out-service all,
changed and retained accuracy, raw NLL, Brier and selected-candidate error
partitions with NumPy float64 arithmetic on saved float32 log probabilities.
It applies no probability floor or normalization repair. It reconstructs the
22 exact count-fraction behavioral checks, public-type false-positive supports,
and paired correctness quadrants against both controls. Primary support is
578 changed and 7,241 retained rows; rare changed supports are reported without
an additional efficacy gate.

Before opening predictions, it authenticates the external plan, execution and
main-report receipts, all 44 execution files, source snapshots, direct metadata
and schema-cache inputs, original paired orders and saved normalization/work
count ledgers. Checkpoint bytes are hashed but never deserialized. Recorded
initializer digests, neural normalization extrema, timings and RSS are
source-bound execution witnesses, not independently rerun measurements.

Model configuration/parameter validity, complete operation-geometry formulas,
literal-reference generation and descriptive secondary subgroups inherit the
authenticated main reporter. This audit is deliberately narrower. Technical
agreement is separate from whether the fixed behavioral rule passes.

The accompanying tests use small artificial arrays only. Neither script
imports a model, encoder, training module or reporter metric implementation.
