# Independent saved-output audit

The recorded [audit completion](result-01/completed.json) passed once and agreed with the reporter. Technical validity passed; scientific continuation remained **FAIL, 7/13**. It used execution completion `768376e24824523b9a5c26923c14206e567fda4f3578ed1babdf7cbdfb1d89f4` and report receipt `fa4e47979b9fab18c40f009423ed1aeb0603991aa4944b29cfa8ffb5e6002a2a`. No training or neural replay occurred.

Run only after all 15 fits and the frozen reporter have completed. Use the externally verified final completion and report-receipt hashes, not hashes captured from a running attempt. From the OpenJev repository root:

```sh
.venv-robotics/bin/python output/dialogue-copy-v2/independent-audit-01/audit.py \
  --run runs/dialogue-copy-v2/study-01 \
  --report output/dialogue-copy-v2/report-01 \
  --plan-sha256 9c39c3b27c7219190bc3f45fc342bc4da4eb408e622402a92ce51efb68ed3905 \
  --completed-sha256 "$V2_COMPLETED_SHA256" \
  --report-receipt-sha256 "$V2_REPORT_RECEIPT_SHA256" \
  --out output/dialogue-copy-v2/independent-audit-01/result-01
```

Set the two environment variables to the supplied final SHA-256 values before running. The report path must identify the completed frozen report. The output directory must not exist. Failures retain `started.json` and `failed.json`; no automatic retry or overwrite is performed. Success creates `summary.json` and `completed.json`. Audit success means agreement and completeness, even when scientific continuation fails.

## Independent checks

- Recalculate all 15 fits' accuracy, NLL, Brier, transition bins, revisions, three-stratum macro metrics, all/seen/unseen panels, and equal-three-seed family summaries from saved probabilities.
- Recalculate both deterministic references' accuracy and the original 13 checks. Accuracy thresholds use exact integer fractions. Numerical metric comparison allows `2e-12` absolute/relative roundoff; gate comparisons receive no tolerance.
- Authenticate the fixed plan, 20 source files, all 64 execution files, report payloads, feature/lexical inputs, original initialization receipt identities, and paired initialization/order records.
- Reconstruct each saved batch's public layout, including real questions and executed dummy-query slots, then verify complete training/evaluation normalization and released-mass witness totals.

## Limits

Model configuration and parameter validation are inherited from the authenticated frozen reporter. This second audit also does not separately compare that reporter's `execution_members` field; it independently hashes the actual execution tree against the execution completion manifest instead.

Initialization tensor generation, optimizer execution, neural state arithmetic, and literal-reference generation remain authenticated source-bound records. The audit does not deserialize weights, replay a neural model, reconstruct the lexical rule from raw text, call an encoder, or access official test data. It imports no main-reporter metric or gate code.

The standalone auditor and its tests are reporting artifacts outside the frozen 20-source experimental closure. Source-only peer reviews found no material blocker. Fifteen synthetic tests and Ruff passed before any real audit invocation.

Reviewed identities:

- `audit.py`: `145f7d8619ae3ed04101a07ad0a41dc97489e2b2aead33d7cf1f1da084933477`
- `test_audit.py`: `cbd64c0f0ae21cf4bcb6da4372a69965ea9bbb6e57737eb52642f7dd11c3c32f`
