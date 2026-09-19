# One conditional architecture direction after the controller correction

This is a prospective, post-outcome design memo. It changes no completed criterion and contains no new model, environment, RNG, training, or inference calls. Finish the fixed-checkpoint confidence diagnosis first. If a development-only confidence transformation accounts for the remaining deficit, another memory architecture fit is not justified.

## What the completed evidence supports

With controller C, ordinary diagonal Kalman memory is the strongest learned family: mean native return **0.370092**, with **83/192** successes. Local innovation inflation gives **0.360577, 80/192**; matched inflation **0.351062, 78/192**; gated delta **0.351362, 68/192**. Exact public memory gives **0.692308, 64/64** and last-32 public memory **0.671575, 64/64**. These are exposed comparisons across three fits on the same 64 decks, not confirmation of an architectural advantage. The successful controller continuation does not reverse the original failed architecture gate.

The current model predicts from `decoder(S.T @ key)` and does not read covariance `P` when producing class probabilities (`card_associative_memory.py:126-133`). Covariance affects subsequent writes, while local/matched variants modify that write covariance (`:178-193`). Calling their softmax outputs calibrated uncertainty would therefore require additional evidence. Remaining failures alone do not identify whether the problem is probability overlap, rare wrong classes, interference, exploration, or their interaction.

## Three closest primary precedents

1. **Test-time regression** organizes associative sequence models by regression weighting, function class, and optimizer, and derives familiar delta-style updates from online regression. It makes changing the fast-memory learning problem an established direction, rather than a new principle. Its discussion of feature maps and online solutions also cautions against attributing a representation bottleneck solely to the update rule. [Wang, Shi, and Fox, 2025](https://arxiv.org/html/2501.12352v1).
2. **Kalman Delta Networks** derives fast-memory mean and covariance updates from a linear-Gaussian state-space model and develops a diagonal approximation. The uncertainty concerns the assumed value-memory model; it is not, by itself, a guarantee about categorical probabilities in this card task. Another residual-dependent diagonal inflation would remain close to this precedent and the existing local/matched experiment. [Bui, Huang, and Ying, 2026](https://arxiv.org/html/2609.07816v1).
3. **Miras** explicitly separates memory architecture, internal learning objective, retention, and optimizer, including gradient writes under alternative memory objectives. This is the closest broad precedent for the candidate below. I am not claiming that Miras presents this exact card-specific formula, or that a categorical instantiation is novel. [It's All Connected, 2025](https://arxiv.org/html/2504.13173v1).

## Candidate: a categorical-error write through the existing decoder

Change one mechanism: replace the learned-value squared-residual write with a categorical prediction-error write. Keep the normalized position key, shared decoder, retention gate, and 32-by-16 fast state. For an actually revealed public rank `y`, let `D` be the 13-by-16 decoder weight and `b` its bias:

```text
S_prior = alpha * S
p       = softmax(D @ (S_prior.T @ k) + b)
e       = D.T @ (one_hot(y) - p)
S_new   = S_prior + beta * outer(k, e)
```

With the event-dependent gate fixed during this inner step, this is one gradient step in fast state on categorical cross entropy. Outer training can differentiate through the entire update, including the residual, identically for every paired implementation. The rank embedding remains available to the same event gate; there is no second optimizer, replay buffer, hard address table, or extra persistent covariance.

This directly aligns each write with the eventual class decision rather than an embedding-space reconstruction. A confident wrong prediction produces a larger categorical correction than a confidently correct one. That could reduce sequential interference or improve separation, but it can also damage other keys and become overconfident. It supplies neither calibrated uncertainty nor a biological mechanism. It is an online categorical-regression/test-time-learning adaptation; a possible contribution would be a reproducible task-relevant improvement under matched controls, not invention of error-driven memory.

Preserve prediction-before-write and exactly one write from the actual selected reveal. No unseen rank, failed-pair counterfactual label, or future observation enters the update. Extra arithmetic is one small decoder evaluation and its transpose projection per write, plus the existing state contraction and outer product. Persistent state remains 512 float32 scalars; actual runtime and arithmetic must be reported, not assumed compute-matched.

## Smallest informative comparison and stopping rule

Only if the confidence diagnosis leaves evidence of a write/interference deficit, prospectively compare **categorical write, ordinary gated delta, and ordinary diagonal Kalman memory**, with three paired initializations and shared data/order per arm. Keep the fixed C controller and shared initial key/value/decoder/gate tensors. Include a prospectively specified temperature transformation of the ordinary baselines as a control that fits no neural weights. The exposed 64 decks may be explicitly repurposed as calibration training data for a new study, but cannot also be its test set; calibration choices must be frozen before evaluating fresh decks. Exact public memory and last-32 remain native references.

Suggested ceiling: nine fits, the existing 128-update recipe per fit, **1,152 total updates**, final checkpoints only, no sweep or failed-recipe retry. New data/deck splits, wall caps, step size recipe, and all selection rules must be fixed before execution. This memo allocates no seeds and authorizes no run. Equal update count is not equal compute; retain wall time, state bytes, parameter counts, and decoder/write work.

A falsifiable proposed continuation rule is at least **+0.03 mean native return over both ordinary learned comparators**, positive paired-family differences for all three pairs, and no worsening of held-out categorical NLL or the predeclared high-confidence-error rate. Use every test deck and retain failures. Reject the architectural explanation if temperature alone removes the improvement, if only selected pairs benefit, or if state norms/finite checks fail. These would be new prospective requirements, not reinterpretations of either completed study. A single failure stops this candidate rather than triggering another covariance or gating variant.

## Why this is probably the wrong flagship task

Static cards have exact addresses and noiseless revealed labels. An explicit public address-to-rank table already solves every evaluated deck without training. In principle, 52 four-bit entries including an unseen sentinel need only 26 packed bytes for rank memory, excluding controller bookkeeping and implementation overhead; the learned fast state alone uses 2,048 bytes. This is a representation comparison, not a measured Python-memory claim.

There is also a fixed representation constraint: for all 52 queries, logits have form `K S D.T + 1 b.T`; each centered logit column lies in the learned key matrix's at-most-32-dimensional column space. Changing the update cannot remove that constraint. This does not prove classification impossible, but a larger or orthogonal address code could explain an improvement without a better recurrent mechanism.

Use Concentration as an implementation and decision-interface check, not evidence of a necessary new architecture. For an architecture claim, a separately motivated noisy-cue or changing-association task would be more informative, with an explicit categorical Bayesian/filter or finite-memory table baseline under the same observation and storage budget. That task must be designed prospectively for a real uncertainty/adaptation question, not adjusted until the proposed model wins.

## Local evidence binding and limits

- Completed controller report receipt: `output/card-controllers-v1/report-01/receipt.json`, SHA256 `f1543d5c87c4ae9f85db14155a21fc2583713dc6eb4918c9081cc3aefc2a48ed`.
- Its authenticated `summary.json`: SHA256 `3343959b8df20680de12557a89652843459100121cfcd2513db8f3e944fe2e51`.
- Inspected model source: `src/openjev/research/card_associative_memory.py`, SHA256 `2163a892354f13ce6c1a7b59f278f32fbc48aee4472e772a82ce1d6bdd6b7451`.
- Existing training dimensions/update count read from `evidence/card-memory-pilot-v1/protocol.json`; they are a proposed budget template only.

This review verified the report-summary binding and inspected source arithmetic. It did not re-audit all episode members, fit any model, infer any new prediction, or evaluate the proposed candidate. Three primary sources constrain the recommendation; they are not an exhaustive novelty search.
