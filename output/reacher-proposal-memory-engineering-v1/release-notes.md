The completed nine-run robot-reaching comparison found no consistent benefit from carrying the previous plan into the next search. It passed 2 of 6 fixed full-episode comparisons; no state/dynamics setting passed against both a fresh search and a last-action control. This is one reused engineering case, not a scientific qualification or learned architecture result.

[Results, chart and next decision](https://github.com/kw2828/OpenJev/blob/main/research/reacher-proposal-memory.md).

All nine 200-action runs passed independent native replay. All three fresh-search runs exactly reproduced the earlier episodes and 9,600 saved decision arrays. The protocol and source were published before execution at commit `a9b689396207d62cd5285403296f05aa12744126`.

The archive contains every new raw row, original shared search inputs, both generations of source snapshots, replay reports, independent reviews, and the three complete earlier fresh-search reference rows. The other earlier raw rows are in the [preceding release](https://github.com/kw2828/OpenJev/releases/tag/research-reacher-tracking-engineering-v1). Every archive member was reopened and checked against `members.json`; archive and manifest checksums are in `SHA256SUMS`.

Execution took 89.63 seconds and replay 84.82 seconds. These shared-host timings include evidence recording and do not establish a deployment speedup. No model training occurred and no weights are included. Original path bindings and runtime assumptions are preserved; this is research evidence rather than a standalone application installer.
