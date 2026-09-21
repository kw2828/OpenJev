# Audited V2 report figures

This presentation directory is outside the 64 frozen scientific sources. It
reuses the qualified V1 two-figure layout and all twelve fits, four arms, three
seeds and seven original conditions. No analysis rule changes.

The generator requires a completed V2 production report **and** a successful
independent V2 audit. Supply externally verified hashes for both receipts and
both summaries. It verifies exact file membership and every payload hash, the
audit's producer-report binding and matching plan/run/supervision identities
before decoding either aggregate summary. There is no partial-result fallback.

After all twelve fits, reporting and independent auditing finish, run from the
repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
.venv/bin/python output/dialogue-observation-learning-v2/figure-01/plot.py \
  --report /absolute/path/to/completed-v2-report \
  --summary-sha256 VERIFIED_REPORT_SUMMARY_SHA256 \
  --receipt-sha256 VERIFIED_REPORT_RECEIPT_SHA256 \
  --audit /absolute/path/to/completed-v2-audit \
  --audit-summary-sha256 VERIFIED_AUDIT_SUMMARY_SHA256 \
  --audit-receipt-sha256 VERIFIED_AUDIT_RECEIPT_SHA256 \
  --out output/dialogue-observation-learning-v2/figure-01/render-01
```

The output directory must not exist. Limits remain 60 suspend-inclusive seconds, 1 GiB process
RSS and 32 MiB output. Outputs are two PNG/SVG pairs, exact plotted values and
source/input/output-bound receipts. Failed attempts are retained without retry.
The qualified, hash-pinned native clock helper enforces a strict deadline,
including a final check after receipt publication. A late completion is retained
as invalid; clock failure has unavailable timing, never a fallback clock. The
emergency signal handler never reads the clock. The helper's identity and timing
scope are recorded alongside the plotter source. Authenticating that small
helper precedes deadline initialization.

`observation` shows seen/unseen three-stratum macro accuracy and unseen micro
NLL/Brier. `retention-and-conditions` shows seen/unseen assigned-retention error
and the seven exact logical conditions. All 72 seed points remain visible with
descriptive mean ticks and panel support counts. Dots use explicitly zoomed
axes. Seeds repeat the same exposed DEV examples and do not justify confidence
intervals. No new architecture or calibration claim is made.

The plotter compares displayed values with saved independent audit aggregates:
counts/statuses agree exactly; floating values use the audit's existing 1e-12
comparison tolerance. It does not recompute predictions or reinterpret the
seven scientific thresholds. Authentication inherits the completed reporter
and independent auditor's source, execution and data validation. It does not
replay their computation or independently audit service/type/factorial tables.

No timing is plotted. Successful V2 root/parent execution time is suspend-inclusive.
Nested fit/update/evaluation/checkpoint durations remain performance-counter
diagnostics that may exclude suspend. They are not alternative admission clocks.
The fixed published V1 failure manifest is hash-bound separately in the figure
receipt, using `41384aeab0d108992906f9f8d0ffbe341751bf7d061288aa98a7441c96f4476f`.
Its five completed-fit records and interrupted sixth remain failed-attempt
spend, not successful V2 work. Old parent monotonic and civil durations are
distinct measurements and are never added together. Only that metadata manifest
is opened; its linked predictions, labels, weights and journals are not read.
`--failed-manifest` may locate the same byte-identical manifest in another checkout.

`smoke.py` uses the existing invented V1 aggregate fixture, relabeled V2, with a
synthetic audit envelope. Synthetic figures prominently say **SYNTHETIC FIXTURE -
NOT RESULTS**. The bounded smoke also rejects wrong pins, extra members, failed
or missing audits and wrong producer joins before either quality summary is
decoded. Its audit envelope tests presentation plumbing and is not a real
independent scientific audit. No actual V1/V2 quality result is read.

The initial `smoke-01` and `qualification-01` remain unchanged. They exercised the
inherited performance-counter presentation lifecycle before the additional V2
suspend-clock requirement. `smoke-02` qualifies the revised native-clock lifecycle
and binds its final source. Both render only invented aggregate metrics.
