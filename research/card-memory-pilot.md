# Learned decision memory: a controlled card-matching pilot

This experiment trains small recurrent memories on the public observations of [POPGym ConcentrationHard](https://github.com/proroklab/popgym/blob/410d5aa626dae8024f498354d8781a0d1870c399/popgym/envs/concentration.py). It asks whether an error-sensitive memory update improves actual decisions, beyond conventional recurrent updates and a matched correction control. It is a single-task pilot, not a claim of biological learning or a new state of the art.

## Experiment fixed before training

Six update rules share learned position keys, rank values and a readout: delta, gated delta, diagonal Kalman, local innovation, gain-matched diffuse innovation and GRU. The proposed local rule increases a diagonal covariance scale in the coordinates of a key whose prediction has an error. The diffuse control increases it everywhere while matching the immediate correction along the current key from the same input state. The control does not match total trace or future memory histories.

Existing work already covers the main building blocks: [DeltaNet](https://arxiv.org/abs/2406.06484), [Kalman Delta Networks](https://arxiv.org/abs/2609.07816), and [HOLA](https://arxiv.org/abs/2607.02303). Innovation-adaptive filtering has further prior art. This pilot tests a specific interference hypothesis; it does not establish novelty by combining names or equations.

- 128 native training episodes; 32 separate development episodes.
- Six modes and three paired initializations, with the same 16 episode-order permutations per pair. Batch 16, 128 Adam updates per fit, final checkpoints only.
- All 18 fits finish before development evaluation. Final controllers then play the same 64 fresh seeds each, plus an exact public-memory table and a last-32-event reference: 1,280 native games total.
- Training labels come only from previously exposed card ranks that are currently hidden. Predictions precede the next reveal; each actual selected reveal causes one write. No hidden deck inspection.
- Every controller receives the same public seen/matched/phase bookkeeping and fixed probability-based action picker. This is supervised memory learning plus a supplied planner, not end-to-end RL.

The GRU matches the associative matrix's 512 mean-state floats but has many more parameters. Kalman variants add 32 covariance floats. Reported full-controller timing includes game logic, bookkeeping and saved episode evidence. Neither parameter count nor compute is matched across all families.

## Continuation rule

Continue the proposed local rule only if it clears every prespecified criterion: mean native return at least 0.03 above the strongest conventional learned arm; a positive difference in every paired fit; at least 0.02 above the gain-matched control, also positive in all three fits; and at least two percentage points better recall of hidden cards last visible more than 32 actions ago, with adequate development coverage. Missing, incomplete or nonfinite results fail the rule. No alternative seeds, checkpoint selection or automatic retries.

A pass would justify fresh-seed and scenario-shift validation. It would not establish ICLR readiness, calibrated uncertainty, robotics transfer, connectome efficacy or a leaderboard result.

[Exact protocol](../evidence/card-memory-pilot-v1/protocol.json) · [Fixed seeds and episode orders](../evidence/card-memory-pilot-v1/inputs.json) · [Prior-work review](../output/decision-memory-direction-v1/related-work.md) · [CPU sizing](../output/decision-memory-direction-v1/sizing.json)

Results: pending the prospectively fixed run.
