# ChessBench transfer results

All twelve frozen OpenJev candidate models completed the same **4,096-position public ChessBench panel**, without further training or checkpoint selection. Mean agreement with the supplied move ranges from **26.03% to 26.79%**. This is a transfer measurement, not an Elo estimate, a novelty result, or a replacement for the [original candidate study's failed continuation gate](chess-candidate.md).

![ChessBench transfer agreement and NLL for all four architectures, showing each of three seeds and their equal-seed mean](assets/chessbench-transfer-results.png)

Markers show seeds 97, 109 and 127; dark ticks show equal-seed means. Source-game IDs are unavailable, so seed variation is descriptive and no independent-game confidence interval is claimed. [Figure provenance](assets/chessbench-transfer-results.json).

| Architecture | Move agreement, higher is better | Legal-menu NLL, lower is better |
|---|---:|---:|
| Direct | 26.0335% | 2.6254 |
| Action only | 26.3590% | 2.6194 |
| Native delta | 26.4730% | 2.5932 |
| Full afterstate | 26.7904% | 2.5735 |

These are arithmetic means across all three seeds, with 4,096 positions per fit. Native delta's mean agreement difference is +0.4395 percentage points against direct and +0.1139 against action only; both comparisons include a negative difference at seed 97. All per-seed metrics and paired differences remain in the [official summary](../evidence/chessbench-transfer-v1/results/summary.json). The table alone does not establish an architectural advantage.

## What was measured

The [frozen plan](../evidence/chessbench-transfer-v1/results/plan.json) bound all twelve published checkpoints, source files, environment, acquired dataset bytes and prior exposures before decoding the benchmark. A fixed, label-independent hash order selected 4,096 eligible positions from the acquired behavioral-cloning test file. The selection examined 4,211 of its 62,561 records. It excluded prior natural and mirrored roots or legal successors, and FEN-detectable automatic terminal positions. Accepted roots are mirror-unique; successor overlap within this panel remains possible. Repetition history and source-game IDs are not supplied.

Each model scored every native legal move once, using a four-step recurrent root and the candidate arms' fixed two-step branch. Delta and full-afterstate use native chess-rule consequences. They do not learn a world model. The evaluator preserved legal-move order, raw logits, choices and probability arithmetic, and verified that checkpoint tensors were unchanged. The official report audited these saved outputs and reconstructed selection without another model forward pass.

Agreement measures matching one reference move; other strong moves can disagree. NLL measures the probability assigned to that move under an uncalibrated legal-menu softmax. This panel supplies no value target, so there is no value MAE, engine regret or calibrated winning probability. It is a selected public development panel, not a sealed test of universal generalization.

Execution completed in 306.99 seconds, including selection and all twelve evaluations. Per-fit receipts retain CPU batch construction, forward and output-writing time. These are **not single-decision latencies**. Brief concurrent synthetic code tests were observed, so this run does not support a clean speed comparison.

## Weights and audit evidence

Every evaluated checkpoint is linked below. No fit was retrained or selected after seeing this panel.

| Architecture | Seed 97 | Seed 109 | Seed 127 |
|---|---|---|---|
| Direct | [weights](../models/chess-candidate-v2/direct-97/weights.pt) | [weights](../models/chess-candidate-v2/direct-109/weights.pt) | [weights](../models/chess-candidate-v2/direct-127/weights.pt) |
| Action only | [weights](../models/chess-candidate-v2/action_only-97/weights.pt) | [weights](../models/chess-candidate-v2/action_only-109/weights.pt) | [weights](../models/chess-candidate-v2/action_only-127/weights.pt) |
| Native delta | [weights](../models/chess-candidate-v2/delta-97/weights.pt) | [weights](../models/chess-candidate-v2/delta-109/weights.pt) | [weights](../models/chess-candidate-v2/delta-127/weights.pt) |
| Full afterstate | [weights](../models/chess-candidate-v2/full_afterstate-97/weights.pt) | [weights](../models/chess-candidate-v2/full_afterstate-109/weights.pt) | [weights](../models/chess-candidate-v2/full_afterstate-127/weights.pt) |

The [verified evidence archive](../evidence/chessbench-transfer-v1/results/chessbench-transfer-v1.tar.gz) contains the complete execution, original report, selection and exclusions, unchanged source data and download receipt, all twelve weights, frozen source snapshots and figure provenance. The [manifest](../evidence/chessbench-transfer-v1/results/manifest.json) hashes every member. Byte-identical plan, summary and official completion receipts are also available beside the archive, bound by the [publication receipt](../evidence/chessbench-transfer-v1/results/completed.json).

Verify the package without model calls or additional dependencies:

```bash
python -S evidence/chessbench-transfer-v1/results/reproduce_package.py audit \
  evidence/chessbench-transfer-v1/results
```

The unchanged dataset comes from [ChessBench by Ruoss et al., NeurIPS 2024](https://github.com/google-deepmind/searchless_chess/tree/90ae0e6b121673fc3079aaeffa047580bb600c0a). Its source identifies CC0 portions from Lichess and CC-BY-4.0 for the remainder. Those terms remain attached to source and derived data; see the [attribution and modification notice](../evidence/chessbench-transfer-v1/results/source-attribution.md). Original OpenJev weights retain their [MIT license](../models/chess-candidate-v2/LICENSE).
