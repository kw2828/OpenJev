# Paired objective comparison: frozen, not yet trained

Status on 20 September 2026: implementation, synthetic checks and the metadata
freeze are complete. **No empirical fit has started and no new performance
result exists.** A separate local experiment currently occupies the coordinated
CPU allocation; release must be verified before this study launches.

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

The [result plotter](../scripts/plot_dialogue_objective.py) is ready for the
completed report. It displays all three seeds for every fixed fresh and
historical readout, with separate panels for changes, retention, overall
accuracy and log loss under both weighting schemes. One
[synthetic layout check](../output/dialogue-objective-v1/figure-preflight-01/receipt.json)
verified the 240 displayed seed values and readable PNG/PDF output. Those
invented values are prominently marked and are not research results. The
plotter reads saved report summaries only; the independent metric audit
remains a separate required step.

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

## Execution still required

After confirming CPU availability, run the six fits once within the frozen
6,000-second, 6 GiB peak-RSS and 512 MiB output limits. The historical workload
projects about 76 minutes; that is an estimate, not a measured new run.

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python scripts/study_dialogue_objective.py train \
  --plan output/dialogue-objective-v1/protocol-01/plan.json \
  --plan-sha256 ea5660f97f75256246808b0971e80ab3ec086baa614ea2080c893716838d4baa \
  --out output/dialogue-objective-v1/training-01
```

Save every fit before quality scoring. Then run the saved-output reporter and
independent primary check, publish all paired results and a figure, and retain
any failed or partial outcome. There is no replacement seed, outcome-based
checkpoint selection or retry within this frozen campaign.

The earlier token-alignment result remains **FAIL, 18/22**. The subsequent
[fixed weight correction](dialogue-weight-prior-results.md) improves retention
while missing more real changes; it does not overturn that result. The broader
goal of a strong novel architecture remains unachieved.

The [source-specific memory note](dialogue-source-memory-decision.md) records
one conditional fallback and its established prior work. It admits no new
experiment and is canceled if the simpler objective resolves this tradeoff.
