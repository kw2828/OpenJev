# Recurrent memory prior art and one conditional hypothesis

Source review, 20 September 2026. No model, test, corpus, timing or numerical experiment was run for this note, and no predictions from the live token-alignment campaign were inspected. This is a research hypothesis, not an implementation or training authorization.

**Recommendation: finish the observation comparison before selecting another recurrent architecture.** The local evidence establishes semantic decision failures even with the correct previous value supplied. It does not yet establish a missing recurrent capability. If a history-dependent deficit remains with adequate observations, the narrow hypothesis below tests delayed predictive feedback while retaining explicit candidate identity.

## Five primary mechanisms, with their boundaries

| Primary source | Exact mechanism and nearest overlap |
|---|---|
| **Gated Delta Networks**, [arXiv:2412.06464v1](https://arxiv.org/html/2412.06464v1), §3.1 | Combines scalar decay with a key-directed delta write: `S_next = alpha*S*(I-beta*k*k^T) + beta*v*k^T`. It joins established gated linear recurrence and associative error correction. A new forgetting or surprise gate would overlap directly; overlapping keys can still interfere. The paper's chunked GPU training and language-model results do not establish small-CPU dialogue speed or useful state tracking here. |
| **Learning to (Learn at Test Time)**, [arXiv:2407.04620v1](https://arxiv.org/html/2407.04620v1), §§2.1-2.6 | Makes the recurrent state the weights of a linear model or MLP, updated by a self-supervised reconstruction loss on learned input/target views; a separate query view reads it. The outer task learns the views and initialization. Linear squared-loss updates closely connect to delta memory. Nonlinear fast weights and differentiating through their updates are established, not a new architecture by themselves. |
| **Titans**, [arXiv:2501.00663v1](https://arxiv.org/html/2501.00663v1), §§3-4 | Updates neural associative memory through gradients of key-to-value prediction, adding momentum and forgetting; combines it with a short-context attention core and persistent parameters. This extends online fast-weight learning rather than providing semantic error labels. Gradient magnitude measures the learned objective's surprise, not calibrated confidence that a dialogue interpretation is wrong. |
| **Mamba-3**, [arXiv:2603.15569v1](https://arxiv.org/html/2603.15569v1), §§3.1-3.3 | Uses an exponential-trapezoidal recurrence containing previous and current state-input terms, complex state rotations, and a multiple-input/multiple-output formulation. The first part is a width-two convolution on the state-input inside the recurrence. Second-order discretization accuracy needs stated assumptions that the empirical parameterization need not enforce. These are substantive SSM mechanisms, but state-tracking expressivity and hardware results do not show that dialogue errors require them. |
| **DeltaProduct**, [arXiv:2502.10297v3](https://arxiv.org/html/2502.10297v3), §3 | Generates distinct keys, values and update strengths for several delta steps per input. Their generalized Householder product permits richer, potentially nonsymmetric transitions; using identical keys collapses the product to one generalized Householder update. The effective recurrence becomes longer. Multi-write transitions and reflections are therefore explicit prior art, not a new contribution from renaming several writes as deliberation. |

These papers establish mechanisms and results in their own settings. None establishes the proposed OpenJev hypothesis, Bayesian semantic calibration, reinforcement learning, or a dialogue world model. The separately reviewed queryable-belief literature is outside this note's scope.

## What the local evidence excludes

The [first dialogue memory study](dialogue-memory-results.md) already implemented gated delta, dense Kalman updates and surprise-based covariance inflation in [the frozen module](../src/openjev/research/dialogue_fast_memory.py). The innovation proposal passed 8/11 checks and failed its rule; literal carry remained substantially stronger on unseen services. The streams had a median of ten USER turns, so a compressed recurrent state does not automatically offer a memory advantage.

The normalized [copy V2 study](dialogue-copy-v2-results.md) failed at 7/13: selective versus scalar unseen macro accuracy was 72.89% versus 72.58%, and unseen revision accuracy remained below literal carry. This does not prove that scalar updates are universally inadequate: their learned writer can produce arbitrary categorical distributions and receives belief-dependent features. The earlier unnormalized scalar implementation is not valid evidence for a new operator.

The [closed conditional diagnostic](dialogue-conditional-error-results.md) found all nine models choosing NOT_MENTIONED on all 31 unseen TRUE updates despite a supplied correct previous value. That supports an observation/decision problem without recurrent error accumulation. It does not establish that the previous value causally caused those predictions. The current [token-alignment comparison](dialogue-token-alignment-design.md) still uses privileged previous labels and cannot establish autonomous memory efficacy.

A proposal/commit buffer and competitive categorical evidence updates already appear in the [local evidence design review](dialogue-evidence-source-review.md). They should not be presented again as new. The separate [pose innovation screen](pose-innovation-results.md) also found summary ridge stronger than the recurrent error head; that is a reason to demand a linear control, not evidence that dialogue must behave identically.

## One conditional extension: delayed predictive plasticity

**Falsifiable hypothesis:** after controlling current semantic evidence and exact candidate belief, a completed, out-of-sample prediction error about the next public observation contains decision-relevant history that a fixed predictor and a finite-window linear adapter do not capture.

Keep an explicit categorical belief `b` with the same candidate identities and observation scorer in every arm. Add a small fast predictor `W` per supplied question, shared across its candidates. Before USER turn `t`, construct each `k_t,c` only from schema/candidate information, past public context, the model's own previous belief and the already available SYSTEM turn. Predict a fixed, schema-conditioned USER feature `v_hat_t,c = W_(t-1) k_t,c`. After USER `t` arrives, obtain `v_t,c` from the fixed observation representation and form the signed pre-update residual. The decision head receives the same current evidence and belief as the controls, plus this residual. Only after forming that decision, update the fast predictor for subsequent turns, for example:

```text
L_t(W) = mean over valid candidates c of 0.5 * ||W k_t,c - v_t,c||^2
W_t = alpha_t W_(t-1) - eta_t grad_W L_t(W_(t-1))
```

The target is a newly observed public feature, never a state label, correctness flag, service frame or future utterance. Fixed target features avoid a jointly learned target collapsing to zero. Padding performs no update; candidate pooling is permutation invariant; supplied questions have independent state. Any persistent literal-history registers remain ordinary shared scorer inputs, not repeatedly counted as newly observed target evidence.

The distinction from the local associative module is the **delayed, pre-update prediction target**, not the delta algebra. The distinction from the old pose correction head is an online state update used by later decisions, rather than a root-only correction of a fixed forecast. TTT, Titans and adaptive prediction are the nearest prior art. This is a locally untested causal contrast, not a claim of a novel learning rule.

The essential controls are the same-feature competitive scalar and structured categorical transition/observation filter, the same predictor with updates disabled, and a causal finite-window ridge/RLS predictor feeding the identical decision head. A current-input reconstruction update with the same fast-state size isolates delay from generic fast-weight capacity. Report actual state, update and end-to-end costs; matching registered parameters alone is insufficient.

**Do not launch this from the present evidence.** Predictable next-observation structure is an unproven prerequisite: user responses reflect external choices, and no agent action dynamics are modeled. A training-only qualification would first need to show predictive information beyond the current SYSTEM turn and simple recent-history controls. Later decision evidence must beat the structured belief and linear controls on changed states without merely trading away retention or improving NLL while making more wrong decisions. Failure of either condition closes the hypothesis. Better future-embedding prediction alone supports neither a useful decision mechanism nor a world-model claim. Previously exposed panels remain development evidence.
