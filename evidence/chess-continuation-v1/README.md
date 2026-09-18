# Does learning action outcomes improve the recurrent actor?

The comparison completed and the original audit passed. **The performance
criteria failed:** no required engine-loss reduction passed, and continuation
scored 50.52% against policy and 50.00% against teacher-action value.
All nine fits and all 192 games completed, with no failed or unfinished games.

- [Result and figure](../../docs/chess-continuation.md).
- [Complete audited summary](audit/summary.json) and [audit receipt](audit/receipt.json).

- [Exact plan](protocol/plan.json), SHA256 `5f3a1d25fcc2da1c1f59a41b91bd4dbc6f1de29bf34e27b3c0c710e6a72cc9b8`.
- [Scientific description](../../research/chess-continuation-supervision.md).
- [210 passing implementation tests](tests.json).
- [Synthetic update profile](../chess-continuation-preflight-v1/profile.json), with no real training examples or retained weights.

Nine fresh fits compare the unchanged recurrent actor under policy-only,
teacher-action value, and distinct continuation-action supervision. They use
88,408 training roots and the same paired seeds and minibatches, totaling
49,752 updates. All nine fits precede evaluation. Epoch 8 is primary; all
eight epochs are reported on fixed diagnostic subsets after fitting finishes.

The main comparison retains the old ordinary and shifted panels, fixed
20,000-node engine grading and 192 color-paired games. These are exposed
development conditions. A positive result would improve a reference model;
it would not establish a novel architecture, world-model planning, biological
wiring advantage, Elo or strength against Astra.

Local lifecycle records are in `runs/chess-continuation-v1/launcher`. The
detached supervisor records process exits and only starts the saved-output
audit after successful primary completion. Primary execution has a four-hour
limit and audit a 45-minute limit, with no retries or replacement fits.
Completed audit evidence is in this directory's `audit/`. Primary execution
took 1,036.68 seconds and the saved-output audit took 47.71 seconds with no new
model or engine calls. The original thresholds were not changed.
The separate pin-quality-v3 run continues unchanged.
