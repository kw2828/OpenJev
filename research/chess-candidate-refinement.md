# Candidate-conditioned chess refinement

**Status: implemented; the separately frozen v2 recovery attempt is running.** The audited capacity gate failed, so this is a width-32 diagnostic. [Architecture](../src/openjev/research/chess_candidate.py) · [Prospective study](chess-candidate-study.md) · [Failure and recovery accounting](chess-candidate-recovery.md). The v1 attempt stopped after 128 updates without a checkpoint or neural evaluation. No improvement or novelty is established.

Hypothesis: candidate-specific processing across the board improves on the current source-square, destination-square and pooled-feature head. This proposes an inductive bias, not a proof of greater expressivity.

For each legal move, give a shared recurrent branch the root representation, action encoding and exact board changes. Score its output in the legal-move softmax. Use two internal refinement steps for one successor position, not two chess plies or cross-move memory. Train branch and ranking head jointly with legal-move cross-entropy plus 0.5 root-value MSE; omit reconstruction loss.

If the capacity gate passes, use width 128 as reference. Otherwise, limit this to a width-32 diagnostic before attempting multi-step rollouts. Do not select seeds or depths after results.

The prototype shares the existing encoder, residual core and source/destination/pooled scoring head. It adds a 1-by-1 action projection with 128 parameters at width 32. All four variants store 43,854 parameters, including the unused auxiliary decoder; the direct control has 33,185 active parameters and each refinement arm has 33,313. Exact delta reuses the encoder's convolution weights without its bias or activation to project the complete encoded input change. Full afterstate runs the complete four-step trunk on every child before the two shared refinement steps. It is native one-ply lookahead, not learned dynamics.

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
- [DeltaCNN, CVPR 2022](https://arxiv.org/pdf/2203.03996): reusing convolution weights on input differences is established. Exact propagation through a nonlinearity needs the difference of activations at cached states. Our one linear correction to an already nonlinear root latent does not reproduce full successor encoding exactly.
- [Stockfish NNUE accumulator documentation](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html#accumulator): chess evaluation already adds and removes changed features from a cached first-layer accumulator. Reusing move consequences is therefore not novel by itself. Our current delta convolution is dense, and broadcast rights/counter channels can change every square; sparse speedups cannot be assumed.

Any positive result initially supports a local policy-design finding, not novelty, superior learned dynamics, calibration or Elo.

## Conditional next experiment

Only if the current controlled probe succeeds, test a learned action-conditioned residual in place of the exact successor correction. Distill the successful oracle's candidate distribution using the same training roots, while retaining the original decision loss. At inference, keep legal-move generation but remove native successor copying and encoding. Compare with an identically sized action-conditioned residual trained only on decision labels, one trained with latent-MSE supervision, and the exact-successor oracle. Count all extra teacher/branch inference used for distillation.

A possible prospective target is at least 25% lower complete decision latency while retaining at least 90% of the oracle's bounded-regret improvement over action-only on both fresh panels. This is a proposal, not a frozen criterion or observed result; margins, budgets and uncertainty requirements must be frozen before that separate experiment. It would test whether useful consequences can be compressed into a learned transition. The broad idea remains closely related to TreeQN and MuZero.
