# Candidate-conditioned chess refinement

**Status: proposed, not implemented or run.** Proceed conditionally on the audited capacity results. Freeze a separate protocol, sources, budgets and fresh panels before execution. No improvement or novelty is established.

Hypothesis: candidate-specific processing across the board improves on the current source-square, destination-square and pooled-feature head. This proposes an inductive bias, not a proof of greater expressivity.

For each legal move, give a shared recurrent branch the root representation, action encoding and exact board changes. Score its output in the legal-move softmax. Use two internal refinement steps for one successor position, not two chess plies or cross-move memory. Train branch and ranking head jointly with legal-move cross-entropy plus 0.5 root-value MSE; omit reconstruction loss.

If the capacity gate passes, use width 128 as reference. Otherwise, limit this to a width-32 diagnostic before attempting multi-step rollouts. Do not select seeds or depths after results.

## Exact transition contract

Use native `python-chess` transitions for every legal candidate. Save the root player's color and encode root and successor in that **fixed root-player perspective**. The default successor encoder would incorrectly switch to the opponent. Include captures, promotion, castling, en passant, rights and counters through the existing encoding. Retain original UCI candidate IDs.

Provide no checkmate/tactical flags, engine scores or teacher shortlist. Use no tactical guard. Native game termination remains common to all arms.

## Required comparisons

1. **Direct policy:** the matched-width current candidate policy.
2. **Action-only refinement:** candidate-specific recurrent processing conditioned on the action, without exact successor changes. This tests whether a richer decision head explains the gain.
3. **Exact-change refinement:** the proposed branch with native successor changes.
4. **Full successor encoding:** re-encode each native successor and train its score with the same legal-candidate cross-entropy used by the other arms.

Control 4 differs from the failed inference-only `-V(successor)` experiment: its scorer is trained for the actual sibling-move decision. Keep root-value supervision identical across arms.

Match training positions, labels, minibatches, updates and paired seeds. Hold teacher calls and node budgets constant. Native successors add rule consequences, not engine labels. Report active/stored parameters; do not claim matched parameter counts, FLOPs or wall time.

## Decision rule and falsification

Freeze fresh positions, scenario shift, paired openings and numerical thresholds before fitting. Measure stronger-engine signed bounded regret and paired-game points, retaining unresolved games in the denominator. Best-move agreement is secondary.

The transition-specific hypothesis fails without gains over both direct policy and action-only refinement. If full successor encoding matches or dominates the quality-versus-cost tradeoff, cached refinement has no demonstrated advantage. Predeclare a test-time permutation of candidate changes to probe use of the correct action-to-consequence mapping; this is a mechanism diagnostic, not independent efficacy evidence.

Measure complete latency: legal-move enumeration, copies, transitions, encoding, batching, inference, synchronization and output handling. Record candidate evaluations, training time and memory. Extra computation must improve decisions or the measured tradeoff, not merely preserve quality.

Attempt learned latent transitions only after exact decision-trained controls help. Their approximation must justify itself against chess's exact simulator through quality and full cost.

## Closest primary sources

- [TreeQN and ATreeC, ICLR 2018](https://arxiv.org/abs/1710.11417) and [author implementation](https://github.com/oxwhirl/treeqn): action-conditioned latent trees trained end to end for their decision objective. The general idea of decision-trained latent refinement is established.
- [MuZero](https://arxiv.org/abs/1911.08265): action-conditioned recurrent dynamics trained to predict planning-relevant policy, value and reward, including chess. It does not establish that our proposed small branch or current supervision is sufficient.
- [Amortized Planning with Large-Scale Transformers, NeurIPS 2024, Appendix B.5](https://papers.nips.cc/paper_files/paper/2024/file/78f0db30c39c850de728c769f42fc903-Paper-Conference.pdf) and [official ChessBench implementation](https://github.com/google-deepmind/searchless_chess): the action-value advantage over state-value learning disappeared when labeled examples were matched. Additional action labels and teacher computation must not be mistaken for an architectural improvement.

Any positive result initially supports a local policy-design finding, not novelty, superior learned dynamics, calibration or Elo.
