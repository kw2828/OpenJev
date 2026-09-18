# Fresh matched pin-factor quality study v3

**Completed and audited; the scientific continuation criterion failed.**
Joint passed 2/16 quality comparisons and trailed all trained comparators in
mean agreement on both panels. Native/cached numerical equivalence passed.
[Results](../../docs/chess-pin-quality.md) · [Audit receipt](audit/receipt.json).

The [launch observation](launch-observation.json) records the detached
supervisor and worker alive with 27 completed updates on the first fit.
This dated observation is not a claim of continuing liveness or completion.
The [plan](protocol/plan.json) has SHA-256
`b0b3da433d1f507d012208624c4231ee777febe7d6843685f9dacd169c691582`.

The [v2 interruption review](../chess-pin-quality-v2/interruption-review.json)
retains five completed fits and 8,239 logged updates, with no evaluation.
Its 3,147.981 seconds of logged updates overlap the completed-fit costs and
must not be added to them. Exact termination time and cause are unknown.
V1's earlier schema stop also remains preserved separately.

V3 uses the unchanged v2 training and evaluation kernels, models, data,
initialization recipes, schedule and acceptance criteria. Every head starts
fresh; none of the interrupted fits is reused. The new version binds the
retained attempt and detached launcher to the protocol. It still requires
24 fits, 36,864 updates, 110,592 evaluation records, every numerical check,
all sixteen quality checks and the full evidence audit. Its fixed run/audit
ceilings remain twelve/two hours. The update profile projects 5.715 hours.

[Sixty tests pass](tests.json), including tiny-fixture update parity with v2,
interruption preservation and detached lifecycle tests. These are engineering
checks, not evidence of model quality.

The one-shot launch command is:

```sh
.venv/bin/python scripts/launch_chess_pin_quality.py launch \
  --plan evidence/chess-pin-quality-v3/protocol/plan.json \
  --out runs/chess-pin-quality-v3/launcher \
  --execution runs/chess-pin-quality-v3/execution \
  --audit evidence/chess-pin-quality-v3/audit
```

The supervisor records regular logs and process receipts and invokes the
frozen audit only after a successful complete execution. Its terminal status
describes lifecycle integrity; the scientific acceptance criterion remains in
the audited study. Check live processes and recent learning receipts before
claiming ongoing work. Do not restart or resume because an observation times
out. The separately frozen [cost-v2 protocol](../chess-pin-trained-cost-v2/README.md)
retains the original native timing comparison and binds it to v3 checkpoints.
It was launched after the complete quality study and replay audit finished.
Timing results remain pending.
