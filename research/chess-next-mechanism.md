# Choosing the next chess mechanism

This is a conditional research note written while the candidate comparison runs. It does not select an architecture from partial results or alter the current protocol.

## If exact afterstates help

Test whether their useful decision information can be compressed into a learned, action-conditioned low-rank update of the root latent. The root is encoded once; each candidate gets a small correction before the existing scorer. Legal move generation remains native, but candidate inference avoids child board copies and child encodings.

Compare identical students trained with decision labels alone, latent-MSE targets, and the oracle's complete candidate distribution. Include a direct policy receiving the same distillation targets, plus the exact-afterstate oracle. This extra direct control matters: better teacher targets must not be attributed to latent transition structure. Retain the original decision loss and count all oracle computation used in distillation.

A proposed target is 25% lower complete decision latency while retaining 90% of the oracle's bounded-regret improvement over action-only on both panels. These margins are not frozen yet. The relevant falsifier is that distillation or an ordinary direct head explains the gain equally well.

[Value-Aware Loss](https://proceedings.mlr.press/v54/farahmand17a.html) already studies learning models for downstream value estimation. The [Value Equivalence Principle](https://arxiv.org/abs/2011.03506) formalizes models that preserve relevant Bellman updates. Matching candidate probabilities alone would not prove that formal property. Together with TreeQN and MuZero, these precedents mean the broad learned-transition idea is established; a contribution would require a specific demonstrated mechanism and tradeoff.

## If all four controls remain weak

Test whether explicit long-range piece relations help the root representation. Use 64 square nodes, the same observed board features, typed attack/defence relations for both colors, and a shared recurrent update. Construct relations once per root and keep the legal candidate menu unchanged.

Compare rule-derived edges, grid edges, degree-preserving rewired edges and dense edges with matched node features, recurrent cell, dimensions, labels, updates and seeds. Count graph construction and message passing in complete latency. Rewiring must retain edge counts and relation-type counts; it must not alter labels or legal actions. Fix its randomization before training.

The test is whether the chess relationships improve fresh ordinary and shifted decisions and paired games beyond grid and rewired controls. If rewiring preserves performance, the claimed relational mechanism fails. If dense propagation has a better quality-cost tradeoff, sparsity has no demonstrated benefit. Missed mates and repetition draws motivate this question but do not establish the cause.

[AlphaGateau](https://arxiv.org/html/2410.23753v1) already represents chess with square nodes, move edges, rule features and edge policy outputs; its authors qualify comparisons against a simplified AlphaZero baseline. [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493) already uses shared GRU propagation. Neither a chess GNN nor recurrent sparse message passing alone is a novelty claim. This would be a controlled inductive-bias experiment before any claim about a connectome-inspired advantage.

## Decision discipline

Publish the complete current comparison first. Select one follow-up from its final outcomes, freeze its controls and margins, then train. The separate [ChessBench transfer proposal](chessbench-transfer.md) offers an external distribution check; it cannot rescue a failed current gate or turn a development result into Elo.
