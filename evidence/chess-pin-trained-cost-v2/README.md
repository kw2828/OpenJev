# Native timing for the fresh pin-quality study

Prepared before quality-v3 evaluation, with zero completed fits and no
evaluation outputs. Execution and replay audit are now complete. Joint takes
6.95 ms versus WLDN's 6.11 ms median complete decision time; its median paired
ratio is 1.131, about 13.1% slower. All 31,104 timings, 54 warmups and 3,456
audit decisions are retained. There are zero changed native choices. These
shared-host timings do not rescue quality-v3's failed 2/16 quality criterion.

![All methods and paired timing ratios](figures/native-cost.png)

[All values and measurement scope](figures/report.md) ·
[Verified raw bundle](https://github.com/kw2828/OpenJev/releases/download/research-chess-pin-quality-v3/cost-v2-verification.tar.gz) ·
[Published hashes](publication/release-verification.json).
The [frozen plan](protocol/plan.json) has SHA-256
`46bd0965241aa8fa8e0a54c5f19a925dfd6f5102a93b97eefc8abe4832abe464`.

This recovery binds the unchanged cost-v1 decision, authentication, timing,
aggregation and replay kernels to quality-v3. Every scientific protocol field
except the version equals cost-v1, including the original 128 roots, nine
methods, three seeds, nine rotating repeats, 31,104 timings, 54 warmups and
3,456 audit decisions. Each phase retains its 30-minute ceiling.

Forty-four constructed-fixture tests passed: twenty-nine recovery cases and
all fifteen original cost cases. They cover versioned checkpoint identities,
predecessor and source hashes, unchanged inputs and scientific protocol,
native equivalence, failed-quality-gate retention and complete audit
prerequisites. This validates the machinery, not model quality or speed.

Run only after quality-v3 completes and passes its full replay audit:

```sh
.venv/bin/python scripts/chess_pin_trained_cost_recovery.py run \
  --plan evidence/chess-pin-trained-cost-v2/protocol/plan.json \
  --out runs/chess-pin-trained-cost-v2/execution
.venv/bin/python scripts/chess_pin_trained_cost_recovery.py audit \
  --plan evidence/chess-pin-trained-cost-v2/protocol/plan.json \
  --execution runs/chess-pin-trained-cost-v2/execution \
  --out evidence/chess-pin-trained-cost-v2/audit
```

The authentic quality auditor remains `chess_pin_quality_study_v2.py`, and
the cost auditor remains `chess_pin_trained_cost.py`, because those are the
executing kernels. The new frozen plan binds both recovery wrappers separately.
Failed scientific or numerical quality criteria remain failed regardless of
descriptive timing. No retries, resumed partial checkpoints or excluded slow
methods are permitted.

The [completed audit](audit/receipt.json) has SHA-256
`80415131be41da01af1e8befa1e2263be29db4be229438d3e048c08a03cdc53b`.
Execution took 261.70 seconds and replay validation took 86.12 seconds;
external process times and the one-attempt supervisor are preserved separately.
