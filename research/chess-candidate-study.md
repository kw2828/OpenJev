# Candidate-conditioned chess study

**Status: v1 failed during its first fit after 128 updates; no completed checkpoint or neural evaluation.** The original protocol and all failed evidence are preserved. See the [explicit recovery amendment](chess-candidate-recovery.md) for the separate v2 attempt. The architecture and evaluation controls below remain the original frozen specification.

## Question and controls

Does processing the consequences of each legal move help a small recurrent policy beyond processing the action alone? All four arms start from the same per-seed root encoder, tied residual core and policy/value heads. The root trunk performs four internal updates; candidate branches perform two more. These are computation steps within one chess decision, not future chess plies or memory between moves.

| Arm | Candidate state before two shared-core updates |
|---|---|
| Direct | No branch; use the current root candidate scorer unchanged |
| Action only | Root latent plus projected source/destination/promotion planes |
| Exact delta | Action-only state plus a linear projection of the full encoded successor-minus-root difference |
| Full afterstate | Re-encode the successor with the full four-step trunk, then add the same action projection |

Use the existing encoder convolution weights, without its bias or activation, to project the exact delta. This shares parameters across the refinement arms. It is an exact change in encoded inputs, **not** an exact update of the nonlinear latent state. Reuse the root policy head on each branch's source, destination and pooled features. Train these scores directly in the root legal-move softmax; do not substitute a negated successor value.

Keep the pre-move player's perspective for every successor. Native transitions must include castling, en passant, promotion, castling rights and counters. Keep all legal UCI candidate IDs. No engine scores, mate/check flags, tactical guard or teacher-selected shortlist enter the policy. The root value loss is counted once per training position.

## Planned bounded fit

Use the original 32,768 spatial training positions, six epochs, logical batches of 128 and paired seeds 97, 109 and 127. This is 1,536 optimizer updates per fit and twelve final fits. All arms receive the same labels, logical minibatches, microbatch boundaries and optimizer settings: Adam at 0.001, betas 0.9/0.999, epsilon 1e-8, no weight decay, gradient norm clipped at 1. The loss remains legal-menu cross entropy plus 0.5 bounded root-value MSE.

The synthetic backward-only preflights tested 16 and 128 positions without changing weights. The executable protocol uses **128 positions per microbatch**, matching the logical batch, and chunks candidate computation at 128 actions. The largest allocation sampled in the 128-position preflight was about 415 MiB for full afterstate; these samples are not measured memory peaks or training-speed results. Both [16-position](../evidence/chess-candidate-v1/preflight.json) and [128-position](../evidence/chess-candidate-v1/preflight-128.json) records are retained.

Accumulate gradients by complete positions and weight each microbatch by its share of the logical batch. Do not split the softmax into separately normalized candidate subsets. Derived encodings are cached with construction time, memory, source hashes and actual native-transition counts recorded. No additional successor teacher labels are acquired.

All final fits finish before fresh neural evaluation. No best-epoch or best-seed selection. Match data, labels, logical updates and shared initialization, while reporting actual active/stored parameters, candidate computations and wall time. Full-afterstate processing is deliberately more expensive than reusing the root latent.

## Planned evaluation and decision rule

Generate 2,048 fresh ordinary and 2,048 fresh shifted positions using the established 50% and 10% random-rollout generators. Generator seeds start at 111000000 and 112000000, with 600 games per panel and 64/96 plies per game. Exclude all prior inputs and recorded successors/games, mirrored equivalents, and **every legal successor of this study's training positions**. Every admitted root and all its legal afterstates must avoid prior exposure sets and the opposite panel's accepted exposure set. Roots remain globally mirror-unique; within-panel successor overlap is allowed so consecutive rollout positions remain possible. This can leave dependence across source games beyond the game-cluster bootstrap. The same generators have been examined before, so fresh positions still constitute development evidence. All visited teacher calls count, including discarded positions; exhaustion of a fixed quota is a failed attempt.

Use the existing 2,000-node labels for teacher agreement. Grade every fit on the same fixed 128 positions per panel using 20,000-node Stockfish calls. Retain all signed bounded score losses, raw centipawn losses, mate scores, calls and node costs. Best-move agreement is secondary.

For a transition-specific positive result, exact delta must reduce mean signed bounded score loss by at least 20% against **both direct and action-only** on both panels, with strictly positive reference means. It must also earn at least 60% of all available points against each of those controls, counting unresolved games as zero in the lower point bound and requiring zero failed games. Preserve failures as unresolved outcomes; never replace them. Each pairing uses sixteen fixed six-ply openings, three paired seeds and both colors: 96 games per opponent. Also run the same 96-game comparison against full afterstate, for 288 scheduled games in total. No Elo follows from this panel.

Compare exact delta with full afterstate on quality and complete decision cost. If full afterstate dominates, there is no demonstrated benefit from reusing the root latent. If only action-only improves, report a richer scorer result rather than an exact-transition benefit. Numerical tolerances, conditional game-cluster intervals and the first scheduled replay selection belong in the frozen executable plan.

Predeclare a deterministic non-identity test-time permutation of candidate deltas within each root, shared across fits, as a mapping-use diagnostic. Report unchanged one-legal-move positions separately. It gets its own result label, no training, no seed selection, and cannot replace any primary criterion. A response to corruption is not proof of learned dynamics or of superior latent reuse. Native copies, pushes, encodings, batching, inference, synchronization and response construction all count toward latency. Also report candidate counts and core iterations; model-only timing is insufficient. Delta adds a convolution per candidate, and full afterstate also encodes every child, so equal core-iteration counts would not establish equal FLOPs.

Only after these controls show useful decision improvements should a learned transition replace the exact simulator. The broader ideas are established in [TreeQN/ATreeC](https://arxiv.org/abs/1710.11417) and [MuZero](https://arxiv.org/abs/1911.08265). This study tests a specific small-policy mechanism; it does not itself establish a novel world model or ICLR-level contribution.
