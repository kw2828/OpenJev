# Candidate evidence correction: a conditional next comparison

Status: source-only design, 2026-09-19. No new fit, model call, data selection, or implementation. Final choice waits for the saved-error diagnostic. The [completed copy study](dialogue-copy-results.md) remains failed: 7/13 requirements passed. Its unseen macro scores were 72.89% selective and 72.58% scalar; unseen revision accuracy was 73.56% selective versus 79.80% literal. Those aggregates do not identify the cause of the remaining errors.

## The concrete limitation

In [the current source](../src/openjev/research/dialogue_copy_memory.py), each candidate independently produces write score `e_i` and departure probability `r_i` from its own public features, previous probability `b_i`, and the previous distribution's entropy. The normalized writer `w=softmax(e)` is computed only after both heads. The departure head therefore does not see the current writer's confidence or alternative candidates' literal matches.

The scalar update is `b'=(1-m)b+m*w`, where `m=sum_i b_i*r_i`. Holding the shared utterance embedding, previous belief, and all other candidate features fixed, a change to candidate `j`'s lexical features affects `m` only through `b_j*r_j`. At a one-hot prior on candidate `k`, an alternative's lexical features cannot directly change `m=r_k`. With `b_k>=1-epsilon`, the alternative candidates collectively contribute at most `epsilon` to `m`. A confident correct write can consequently be blocked by a departure decision that lacks its schema-specific evidence. The shared utterance embedding remains an indirect path; this is not proof that the complete model cannot learn replacement.

There is a second, conditional limitation: for fixed `b,w,m`, scalar output lies on their line segment, so each coordinate cannot exceed `max(b_i,w_i)`. It cannot treat agreement as stronger evidence than either distribution alone. Selective transport is not confined to that segment: candidate-specific rejection already permits sharpening and correction. Moreover, both old scorers inspect predicted belief and can learn a sharper `w`. It would be incorrect to claim that either trained model universally cannot reject, accumulate evidence, or represent the desired answer.

Both categorical methods also discard any history distinction not represented by their belief and supplied literal register. Equal beliefs plus equal future public inputs yield equal outputs. This does not yet justify a larger evidence-state architecture: the diagnostic must establish a useful lost distinction, not merely another error bucket.

## One proposed alternative, with the missing control supplied

Test a **competitive scalar** against a **competitive normalized-product correction**. Preserve the current projections, ten lexical features, own prior probability, entropy, data, loss, and candidate indexing. First compute `e` and masked `w=softmax(e)`. Then expose `w_i` and normalized `H(w)` to the departure head. A minimal implementation adds two shared scalar coefficients to its existing logit:

```text
r_i = sigmoid(a_i + theta_1*w_i + theta_2*H(w)/log(max(C,2)))
h = sum_i b_i*r_i
u_i = 1/C over valid candidates

competitive_scalar:  b' = (1-h)*b + h*w
competitive_product: pi = (1-h)*b + h*u
                     log b' = log(pi) + e - logsumexp(log(pi)+e)
```

Here `C` counts valid candidates. Initialize the two added coefficients to zero and pair every initial tensor. This adds two registered parameters to the existing non-GRU layout. It gives the occupied candidate direct access to rejection by the current writer and to whether the alternative evidence is concentrated. It adds no cross-query operation or candidate-ID embedding.

Use three trained arms if this design is selected: scalar with the two writer-statistic inputs masked, competitive scalar, and competitive product. All instantiate identical parameter shapes and receive identical public inputs; disclose the two inactive coefficients in the masked arm. The first contrast tests the missing gate input. The second tests the update factorization after that input is supplied. Retain literal carry and candidate GRU as strong controls under a prospectively fixed comparison, rather than selecting the favorable historical result for either.

The product has a distinct fixed-factor behavior. In a two-candidate example with `b=(.9,.1)`, `w=(.9,.1)`, and `h=.1`, scalar remains `(.9,.1)`. Product first forms `pi=(.86,.14)` and returns first-candidate probability `387/394`, about .9822. Conversely, contradictory evidence changes relative odds according to `log(b'_i/b'_j)=log(pi_i/pi_j)+e_i-e_j`. These are operator identities, not evidence of improved learned reasoning.

The two uses of `h` are deliberately different: total overwrite mass for scalar, support-renewal hazard for product. There is no equal-released-mass claim. At `h=1` both reduce to `w`; with uniform writer both return `(1-h)b+h*u`. At `h=0`, product can still revise a nondegenerate prior through evidence, whereas scalar carries it. Product carries exactly only when the hazard is zero and evidence is candidate-constant. Finite sigmoid logits do not attain exactly zero hazard in real arithmetic, so that identity is a kernel endpoint, not a guaranteed learned no-evidence behavior. Positive hazard admits candidates with zero prior support without an undocumented floor. Implement these statements in log space and test zero-support endpoints explicitly.

This is a discriminative predict/correct recurrence, not an exact generative Bayesian filter. Current USER features can affect both heads; scores are not independently observed likelihoods. The [prior source review](dialogue-copy-source-review.md) already proposed an odds-filter option, explicitly excluded from the completed study. The local new test would be this option against a scalar control that can see current candidate competition, not the invention of Bayes or copying.

## Behavioral checks and the main risk

- Candidate permutation, including reserved-state indices, must permute outputs. Adding or reordering unrelated questions must not change a question's result.
- Full-prefix and stepwise evaluation must agree; future turns and gold labels must not affect an earlier update. Only padding pauses a stream. Masked candidates never enter normalization.
- Verify the fixed-factor identities above, finite gradients, normalization, and recovery from zero support. A stronger writer must be able to alter the competitive gate even when its candidate has zero prior probability; the masked scalar must retain the old conditional invariance.
- Shared SYSTEM/USER text, lexical flags, and state inputs stay identical across arms. No gold correction, reliability, or operation annotation may enter the actor.
- Preserve old literal-register semantics and report that it is persistent derived history. Product must not be described as multiplying independent evidence: repeatedly reusing that register or an echoed SYSTEM statement can amplify the same evidence. An irrelevant turn is not automatically neutral merely because it has no exact match.

The greatest risk is overconfident persistence, not only under-replacement. Strong positive evidence for the wrong identity can be reinforced repeatedly. NLL/Brier and retained-state errors are therefore necessary counterweights to revision accuracy. One-hot accuracy alone cannot distinguish useful evidence integration from harmful confidence growth. A shared lexical feature improvement must not be credited to product topology.

## Decision after the diagnostic

Prefer the competitive-scalar correction alone if remaining mistakes concentrate on confident stale values despite visible alternatives. Product is worth the additional matched arm only if there is evidence of recoverable ambiguous or contradictory beliefs, rather than primarily missing semantic cues or ontology-specific language interpretation. Final saved output probabilities can describe those symptoms; without saved writer/departure states they cannot prove a gate bottleneck.

If competitive scalar explains an improvement and product does not improve corrections and probability quality over it, reject the product mechanism. If both fail on the same public interpretation errors, do not respond by adding more posterior factors or calling the problem calibration. Revisit the text representation or stop this recipe. No new numerical continuation thresholds, seeds, official-test access, or fit budget are selected here. The already exposed development set can support a development comparison only.

## Inspected identities

- `src/openjev/research/dialogue_copy_memory.py`: `182c71a944c6d787533bf129bca1631ff3b366b8559e3db7567e7fd9b19a8717`.
- `research/dialogue-copy-protocol.md`: `75c7eaad7d421d18a2872fa2af7a458a72f5aa6722c2e888df2301dcae2519c0`.
- `research/dialogue-copy-results.md`: `89d8e8d34219b06bd2f0e828f33b4a783ccd751861f58372f022ea8359182e89`.

These identify the inspected source and completed comparison, not a frozen prospective experiment. No existing file was changed.
