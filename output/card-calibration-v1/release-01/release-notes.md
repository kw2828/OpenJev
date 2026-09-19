Completed probability calibration comparison: **3,584 fresh games, fixed neural weights, scientific continuation FAIL (11/12)**.

All 15 associative-memory fits finish 64/64 games after temperature scaling, totaling **960/960**, versus 407/960 with original probabilities. The three GRU fits finish none; their mean return declines 0.01171875, exceeding the allowed 0.01 regression. The original proposed architecture remains failed, 1/6. No criterion was relaxed and no fit was removed.

On identical fresh baseline histories, mean NLL falls **36.29%** and Brier **26.85%**. Hard decisions also finish 960/960 associative-memory games, but retain infinite NLL on every wrong prediction. Simple public-memory references remain more efficient than the learned families. This is a confidence/controller result, not a new architecture or RL algorithm.

- [Results and embedded replay](https://github.com/kw2828/OpenJev/blob/main/research/card-calibration-results.md)
- [Prospective protocol and source freeze](https://github.com/kw2828/OpenJev/tree/c464120cbd4acfde1f3a21fdf36db3eec324aa30/evidence/card-calibration-v1)
- [Independent audit](https://github.com/kw2828/OpenJev/blob/main/output/card-calibration-v1/independent-review.md)

The archive contains all new scalar fits, 3,584 native trajectories, reports, figures, independent audit and prospective source snapshots. `manifest.json` gives every payload hash; `receipt.json` records the archive digest and full streamed verification.

Full lineage also requires the [original memory training release](https://github.com/kw2828/OpenJev/releases/tag/research-card-memory-pilot-v1) for unchanged checkpoints and the [controller release](https://github.com/kw2828/OpenJev/releases/tag/research-card-controllers-v1) for the old C trajectories now explicitly used as calibration training. The new test decks are disjoint. All scientific runs completed once, with no retry or neural-weight update.
