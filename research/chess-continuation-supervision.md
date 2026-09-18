# Test the learning targets before adding another small chess head

Status: completed and audited; the continuation criteria failed.
All nine fits and 192 scheduled games completed. Continuation scored 50.52%
against policy and 50.00% against teacher-action value. None of the four
required engine-loss reductions or two game-score thresholds passed.
[Full result and figure](../docs/chess-continuation.md).

The current 24-fit pin-factor comparison continues unchanged. The
[frozen plan](../evidence/chess-continuation-v1/README.md) bound the exact data,
implementation and limits before fitting. All 210 focused implementation tests
passed before launch; the separate synthetic update profile is engineering
evidence. The experimental rationale and original protocol follow below.

## Why this is the next question

The [capacity comparison](../docs/chess-capacity.md) increased active parameters
13.23-fold without a convincing gameplay gain. A local saved-log diagnostic
(`evidence/chess-training-dynamics-v1`, outside this publication package) finds
that width128 improves last-epoch online policy cross entropy by 0.2349 over
width32, but final ordinary NLL improves by only 0.0126 and shifted NLL worsens
by 0.0362. These are not fixed-checkpoint train/test gaps: the training losses
were observed across changing weights. Every final epoch still improves its
online loss. Neither convergence nor a particular cause of weak generalization
has been established.

Gameplay supplies a separate warning: the capacity games missed 74 of 104
recorded immediate-mate opportunities. Earlier mate-specific training improved
the selected task while degrading general decisions. This motivates testing
broader action-outcome supervision, rather than treating another width increase
or a tactical-puzzle score as a solution.

WLDN is the strongest established adapter in the inspected development studies,
but its original head uses frozen root piece features for both root and child
attack graphs. Its 37.01% / 31.43% agreement has not established gameplay
strength. Explicit successor information and joint fine-tuning separately
failed in other model families, so neither is assumed to solve the problem.

## Reuse already paid-for training labels carefully

The original and capacity training generators retain accepted roots, behavior
actions, complete game histories and teacher analyses for visited states.
A [train-only continuation inventory](../evidence/chess-transition-value-inventory-v1/README.md)
has completed with 96,758 usable labels from 98,304 accepted roots: 96,474
nonterminal teacher continuations and 284 exact native terminal wins. It
retains 1,546 skips, including 1,468 unobserved children, 32 excluded children
and 46 other unaccepted children. Thirty-nine synthetic tests passed, and all
source, code and output hashes were checked. No new teacher calls or training
occurred. A nonterminal
pair is usable only when the exact successor is itself an accepted training
root in the same source directory and game at the next ply. Its teacher value
is negated into the original player's perspective. Actual terminal children
use the result from native rules applied to the full recorded history.

Unobserved, capped, excluded and non-training successors do not acquire an
invented target. No development or shifted labels may enter the join. Source
directory, game, ply and exact FEN jointly identify a transition; matching a
board string alone is insufficient. Retain every skip and all inherited
teacher costs. These are additional targets from existing exposed training
data, not free original supervision, new independent evidence, calibrated
winning probabilities or exact game-theoretic action values. Differences
between successive finite searches must remain visible.

On nonterminal moves matching the teacher choice, the mean absolute difference
between root value and negated child value is 0.02825 in the original source
and 0.02885 in the capacity source; their 95th percentiles are 0.09406 and
0.09654. Those differences are descriptive finite-search consistency, not
exact Bellman errors. All eligible pairs remain available, without a
quality-based filter. The inventory contains only one behavior continuation
per root, not a full action-value vector or a multi-step search target.

## The controlled first comparison

The same width32, depth4 residual actor is freshly initialized for three
objectives and paired seeds 193, 211 and 227. All retain the original legal-move
cross entropy and root-value MSE with coefficient 0.5. An independent scalar
critic shares the policy head's hidden activation; the raw actor logits and
actor-only move selection remain unchanged.

- **Policy:** the original objective; the critic is stored but unused.
- **Best value:** add critic MSE on the teacher action against the root value.
- **Continuation:** add the mean critic MSE over the teacher action and the
  available behavior action when it differs from the teacher action.

The auxiliary coefficient is 1 per root in both supervised arms. Duplicate
actions receive only the original teacher target. Missing outcomes are
selected out before arithmetic. Finite-search discrepancies remain visible
and are not clipped, reordered or filtered. Separate policy and critic
outputs avoid forcing confident policy logits to represent absolute losing
position values.

The deterministic source-game split preserves all 98,304 roots: 88,408 training
roots from 1,577 games and 9,896 diagnostic roots from 177 games. Canonical order
is the original spatial source followed by the capacity source. All natural
and mirrored root/continuation state intersections across groups are empty
under the first four FEN fields. This excludes clocks and repetition history;
the games are historical and exposed, not independent confirmation.

Each fit receives eight epochs, batch128 with the tail retained, 5,528 updates
and 707,264 examples. Within each seed all arms share minibatch order and
initial actor and critic parameters. Adam uses learning rate .001, betas .9/.999,
epsilon 1e-8, no weight decay and gradient clipping 1. Training uses MPS float32
without CPU fallback. CPU evaluation uses two threads. Stored parameters are
43,791 in every arm; active training parameters are 33,185 for policy and 33,250
for the other arms. All actor inference uses 33,185 parameters. This is not a
matched-compute comparison; actual costs are recorded.

All nine fits finish before evaluation. All eight epoch checkpoints are then
measured on the fixed first 2,048 roots of each partition. Epoch 8 alone is the
primary checkpoint; no retrospective stopping or selection is allowed.
Primary panels are the existing 4,096 ordinary and 4,096 shifted capacity
positions, with 128 prespecified positions per panel graded using 20,000-node
Stockfish searches. The ceiling is 2,560 calls and 51,200,000 requested nodes;
reported nodes, score differences and raw centipawn tails are retained.

The arena contains 192 games: continuation against each control across three
paired seeds, 16 fixed common opening prefixes and both colors. The original
300-second/no-increment clock and 240 additional-ply cap apply. Complete native
history is retained. Unfinished games have 0-1 point bounds, not imputed draws.
Post-game immediate-mate checks do not influence any move or policy clock.

Promotion requires at least 20% lower mean signed bounded engine loss against
**both** controls on **both** panels, with positive control means, and at
least 60% lower all-game points against **each** control with no failed games.
These are development criteria, not significance or Elo guarantees. Stronger
adapters and fixed-engine gameplay remain subsequent opponents before a broad
strength claim. The study stops after four hours of primary execution and
45 minutes of saved-output audit, without retries, replacement seeds or budget
extension. A separate supervisor enforces the phase timeout and preserves
failure evidence.

The audit checks sources, all recorded updates/checkpoints, scalar prediction
arithmetic, engine coverage and complete native game replay without new model
or engine calls. Scalar records do not independently prove complete score
vectors or exact gradient execution. A positive result still requires fresh
source games and a complete exposure boundary before confirmation.

This is preparation for a strong research baseline, not a novel architecture
claim. Recurrent prediction, adaptive computation or a learned connectome
interface should subsequently compete against it under matched information
and measured compute. A positive outcome from richer labels must not be
attributed to biological wiring or recurrent world modeling.

The completed [connectome comparison](../docs/chess-connectome.md) failed its
biological-topology criterion. Its artificial board-to-neuron mapping motivates
testing a learned interface later, but any such trial must pair the biological
graph with degree-preserving rewires and the same interface learning. Neither
this supervision study nor that potential interface change overturns the
existing negative result.
