# Connectome spatial-mapping comparison

Completed once on 2026-09-18: **30 of 30 fits completed; 27 of 40 continuation
checks passed. The study did not pass its continuation rule.** Both execution
and saved-output audit exited successfully. No game-strength follow-up is
justified by this result.

![All thirty fits, unchanged direct references, costs and every continuation check](figures/figure.png)

[PDF](figures/figure.pdf) · [Complete results](audit/summary.json) ·
[Audit receipt](audit/receipt.json) · [Centipawn tails](figures/engine_cp_tails.csv) ·
[Figure provenance](figures/provenance.json) ·
[Study design and interpretation](../../research/chess-connectome-mapping-study.md)

## Result

All means below retain the three paired seeds. These signed bounded engine
losses use the frozen 128-position subset of each historical development panel;
lower is better.

| Topology | Ordinary fixed → learned | Shifted fixed → learned |
|---|---:|---:|
| Biological | .101983 → .093204 | .103960 → .104099 |
| Rewire 151 | .095392 → .101010 | .097407 → .107061 |
| Rewire 163 | .090986 → .097719 | .104259 → .105320 |
| Rewire 179 | .096636 → .102945 | .102657 → .106654 |
| Node-local | .092884 → .094729 | .104899 → .110014 |

The unchanged direct-model means are .097387 ordinary and .110094 shifted.
Biological mapping improves ordinary mean loss by **8.61%**, below the frozen
10% threshold, and worsens shifted loss by **0.134%**. All eight mean-reduction
checks fail; three paired rewire comparisons and two direct no-degradation
checks also fail.

All eight difference-in-differences checks pass. This records a topology-by-mapping
interaction, but in the shifted panel it reflects learned mappings hurting
controls more, not biology improving. It is not evidence of reliable biological
superiority or game strength. Some fixed controls also outperform the biological
learned mapping. All seeds, controls and previous negative results are retained.

## Work and cost

- Five topologies × fixed/learned mappings × three paired seeds: 30 fresh fits,
  plus three unchanged direct references. Each fit uses the same 32,768 roots,
  frozen backbone, six epochs and 1,536 optimizer updates.
- Exactly 320 proposal pairs per fit: 9,600 total. Learned interfaces accepted
  2,110 of 4,800 proposals. Fixed controls accepted zero while performing the
  same two forwards per proposal. All 30 fits finished before evaluation.
- Primary execution: **2,395.74 s** against a 7,200 s cap. Audit: **833.92 s**
  against a 1,800 s cap. No retry, replacement fit, resume or extension.
- Summed full fit time: **1,518.43 s**, including **477.49 s** of proposal work.
  Proposal time is already included. Backward, optimizer and bookkeeping costs
  are charged in wall time; forward-operation estimates are not total FLOPs.
- Stronger grading: 748 deduplicated calls, 14,960,000 requested nodes and
  14,885,370 reported nodes. No model or engine calls were made during audit
  or figure rendering.
- Shared-host mean full decision latency: biological learned **1.752 ms**;
  direct **0.656 ms**. This is not isolated hardware benchmarking.

The [prior connectome comparison](../../docs/chess-connectome.md) remains
negative. The current study tests a hard 64-square spatial permutation shared
across channels and tied between graph input/output. It is not an intact fly,
a cross-move recurrent world model or a new architecture claim. These exposed
panels are not independent confirmation data. No games or Elo were measured.

## Evidence and availability

[Plan](protocol/plan.json) SHA-256:
`bef6862e84733c6436b09dc3d693037247895bc017132b388f692fabf74aac04`.
The [preparation receipt](protocol/prepared.json) records its pre-training freeze.
The [launch observation](launch-observation.json) is the earlier point-in-time
launch record, not the terminal result.

The audit authenticated all 30 training/proposal journals, checkpoints, full
saved score vectors, engine coverage and timing. It did not rerun inference,
Stockfish or optimizer execution. Public artifacts include the plan, audit
summary/receipt, PNG/PDF figure and a CSV retaining all 66 per-model/panel
centipawn means, p95 values and maxima. Figure provenance records exact input
hashes; all means are descriptive, with no confidence intervals inferred from
three fitted seeds.

Biological arrays, rewires, derivative weights, original training data, raw
execution records and launcher logs remain local under `runs/` and the original
source directories, subject to their original terms. This package is not a
self-contained raw execution archive. The public figure can be reproduced from
the public authenticated audit without model weights or biological assets.

The [135-check engineering receipt](tests.json), [full input verification](input-check.json)
and [56 synthetic figure checks](figure-tests.json) remain available. Full input
verification took 1,562.58 s before launch and reproduced all 32,768 training and
both 2,048-position evaluation caches without neural or engine calls. That is
preparation cost, separate from the measured execution above. The earlier
[synthetic workload profile](../chess-connectome-mapping-preflight-v1/README.md)
contains no study-performance result.

To reproduce the derived figure in a new, unused directory, using
`matplotlib==3.11.2`:

```bash
.venv/bin/python scripts/plot_chess_connectome_mapping.py \
  --audit evidence/chess-connectome-mapping-v1/audit \
  --plan evidence/chess-connectome-mapping-v1/protocol/plan.json \
  --expected-plan-sha256 bef6862e84733c6436b09dc3d693037247895bc017132b388f692fabf74aac04 \
  --expected-receipt-sha256 3c7a8d342733e7d5518ca02f7ab9045f6dc13f03f91df7a6b2bff649fde6870e \
  --out runs/chess-connectome-mapping-figure-reproduction
```

This command renders saved results only. It does not restart training or evaluation.
