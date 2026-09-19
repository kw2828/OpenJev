# Dialogue evidence memory: prior art and a bounded next hypothesis

Source review, 2026-09-19. This is a provisional design, not a new experiment or a novelty claim. It uses the [published copy-study results](dialogue-copy-results.md), its [frozen protocol](dialogue-copy-protocol.md), and the primary papers below. No individual examples, predictions, checkpoints, or new diagnostic outcomes were inspected; no model, data, or training calls were made. Final selection waits for the separate saved-error diagnostic.

The completed selective update failed its continuation rule, with **7/13 checks passed**. Its unseen macro accuracy, 72.89%, barely differs from scalar memory's 72.58%; scalar wins on seen services and has better seen NLL. The shared lexical/history feature ablation supports retaining those observations for the next comparison, but does not isolate the benefit of exact candidate storage. The study does not establish that finer probability retention is useful. A useful next comparison must beat the stronger scalar model, not the earlier compressed-memory baseline.

## Four primary sources and what they establish

| Primary source and version | Established mechanism and relevant boundary |
|---|---|
| Mrkšić and Vulić, **Fully Statistical Neural Belief Tracking**, ACL 2018, proceedings version, [paper](https://aclanthology.org/P18-2018.pdf), §2.1, equations 4-6. | Learns `softmax(W_current*y + W_past*b)` and a value-independent version tying diagonal/off-diagonal weights. Differentiable categorical belief updates and parameter sharing across unseen values are established. Its WOZ setting uses preceding system-act representations; that is not our raw-text-only supplied-schema boundary. A learned odds recurrence is not novel merely because its state is an explicit probability vector. |
| Chen, Chang and Mehta, **Differentiable Filtering for Learning Hidden Markov Models**, [arXiv:2511.10571v2](https://arxiv.org/html/2511.10571v2), 24 April 2026, accepted L4DC 2026, §3/Algorithm 1. | Belief Net learns initial, transition and emission logits through the forward filter: `posterior ∝ likelihood * prior`, followed by transition and next-observation prediction. Its evidence concerns HMM identification and small character-level language modeling; the larger tested nanoGPT has lower text perplexity. It does not establish DST transfer, calibrated neural likelihoods, or a replacement for general transformers. The HMM conditional-independence assumptions matter. |
| Heck et al., **TripPy: A Triple Copy Strategy for Value Independent Neural Dialog State Tracking**, SIGDIAL 2020 proceedings version, [paper](https://aclanthology.org/2020.sigdial-1.4.pdf), §§3.3-3.8. | Copies from user spans, a system-inform memory, or other tracked slots; a gate selects the source. This directly precedes retaining proposal identities separately from committed state. The paper's structured system-inform operations and cross-slot memory cannot silently become inputs to our independent raw-text questions. A local implementation would have to derive proposal evidence from public text and charge its extraction. |
| Dey et al., **Know Your Mistakes: Towards Preventing Overreliance on Task-Oriented Conversational AI Through Accountability Modeling**, ACL 2025 proceedings version, [paper](https://aclanthology.org/2025.acl-long.1399.pdf), §§3.2-3.3/Algorithm 1. | Adds a supervised slot-presence classifier to an LLM and corrects generated state by deleting suspected false positives or generating missing values. Explicit state correction is established. Its validation-tuned thresholds, extra decoder calls, slot-presence objective and full generation task differ from a compact candidate residual head. Do not transfer its empirical gains or cost claims to OpenJev. |

The [earlier review](dialogue-copy-source-review.md) already covers TRADE, SOM-DST and Bayesian dialogue tracking. These four sources extend that boundary rather than supporting a claim that copying, Bayes, a recurrent belief, or a residual correction is new.

## Option 1: expose competing evidence, then test the recurrence

The current departure head sees its own candidate's lexical features, belief and shared turn embedding, but not the normalized writer distribution. With a one-hot belief on old candidate `k`, its release mass is `m=r_k`. Changing another candidate's lexical flags has no direct effect on release, even when the writer strongly favors that alternative. The shared utterance embedding is still an indirect route, so this is a conditional information-flow limitation, not universal lack of expressivity.

Use the [matched design](dialogue-evidence-design.md): expose each candidate's normalized writer probability and writer entropy to the departure head. Compare the old scalar, this competitive scalar, and a competitive product update with identical scorer inputs and initial tensors:

```text
w = softmax(e)
h = sum_i b_i * r_i
competitive scalar:  b_next = (1-h)*b + h*w
product:             prior = (1-h)*b + h*uniform_valid
                     b_next = softmax(log(prior) + e)
```

For fixed factors, the scalar cannot exceed the larger of its two input probabilities for any candidate; the product can reinforce agreement and reverse relative odds. This is an inductive-bias contrast only: the trained scalar's belief-conditioned writer can itself sharpen. Moreover, `h` means overwrite mass in one arm and support renewal in the other, so equal tensors do not imply equal operations or equal released mass.

The product is **discriminative**, not an exact Bayesian posterior: the same utterance affects both factors, and its score can depend on prior belief or a persistent literal register. Reusing those features may repeatedly count one observation and increase erroneous confidence. Constant evidence returns the renewed prior, not the old belief unless `h=0`; a sigmoid hazard is not exactly zero. Positive renewal permits escape from zero support, while literal probability floors would change the model and must not be hidden.

**Falsification:** if competitive scalar accounts for the improvement, reject the product-specific explanation. If product improves accuracy while worsening correction errors or proper scores, do not describe that as better evidence integration. This is the smallest justified next mechanism comparison if the diagnostic shows confident stale beliefs despite available competing evidence. Saved final probabilities alone cannot prove the internal gate cause.

## Option 2: retain evidence separately from the committed answer

Use this only if the diagnostic instead indicates useful older public context lost by both scalar and candidate GRU. Maintain a fixed four-turn buffer per supplied question containing already available turn embeddings, candidate lexical observations, source indicators and age. Keep a frozen scalar baseline's own recurrent belief `b_scalar` independent. A small candidate-equivariant attention head reads that buffer and predicts a signed residual:

```text
output = softmax(log(b_scalar) + residual(current_query, candidate, buffer))
```

Initialize the residual head to zero. It then exactly reproduces the scalar wherever support is positive. It cannot revive a true zero through multiplication; zero-support behavior must be specified before implementation. Do not feed corrected outputs back into the baseline for this first isolation test. Supervise only current state CE, with no operation, relevance, correctness, or gold-prior input. This can reconsider a proposal without making the baseline's earlier commitment the only summary of history.

The necessary control is the **same buffer and attention representation feeding an ordinary scalar read/write head**, plus a current-turn residual head with the same output parameterization. If the buffer helps both equally, the result is improved context access, not a new correction operator. Extra memory, attention, and baseline evaluation all count. Source-aware external memory and correction are established; the only prospective contribution is a demonstrated compact implementation that resolves a specific failure under matched public information. Do not launch this option alongside option 1 as an outcome-selected search.

## Constraints before choosing either

Require candidate permutation, unrelated-query independence, exact padding no-op, causal prefix agreement, and no reset when a slot is unscored. Distinguish a truly neutral numerical evidence input from a natural-language turn with no literal match. Include repeated irrelevant/echoed turns and conflicting evidence in synthetic behavior tests, without claiming these fixtures establish real efficacy.

Keep the shared encoder, lexical observations, state loss, candidate set, training examples and paired initialization fixed wherever the comparison permits. Report all seeds, retained/changed/revision outcomes, NLL/Brier, parameters, retained state and full measured work. No annotated system acts or cross-question gold state are allowed. The repeatedly exposed development set remains development evidence; neither option repairs the failed 13-check study or establishes RL, a world model, or architectural novelty.

**Recommendation pending the diagnostic:** prioritize the three-arm competitive-evidence comparison if its causal symptom is supported. Otherwise pursue the bounded evidence buffer only with a concrete missing-history finding. If the remaining failures are semantic information missing from the frozen inputs, changing recurrence algebra is not a supported next step.
