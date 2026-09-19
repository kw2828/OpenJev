# Frozen-memory confidence comparison

**Prospective design and pure arithmetic implementation only.** No calibration corpus was opened, scalar fitted to real data, model called, environment run, or new seed allocated for this note. The original architecture pilot and subsequent controller comparison retain their own results and criteria. This experiment would test confidence use in the supplied controller, not a new memory architecture.

## Calibration source and information boundary

Prefer **all 64 completed prior C episodes for each of the 18 fixed memories** as explicitly repurposed calibration **training** data. This uses the actual C state distribution and requires no additional memory inference. Authenticate the prior completed evaluation and independent review, all 18 weight/configuration identities, and every selected C episode member before fitting. The old C decks were held out for their original evaluation; they are not held out for this new study. Preserve that lineage and the original published results.

The old 32-episode development API was inspected: `card_memory_training.evaluate` predicts all 52 ranks before the current selected reveal, uses public seen/currently-hidden targets, and reduces queries within boundaries, then boundaries within episodes, then eligible episodes. Its aggregate development files do not retain the logits needed for temperature fitting; returning to the packed development corpus would require new frozen-model inference and would use the older teacher/random behavior distribution. Do **not** fit both sources and choose whichever temperature looks better. Select the prior-C corpus prospectively; retain the original development artifacts as historical evidence only.

Reconstruct each C episode's labels with `PublicTeacher` from recorded public frames, before the action/reveal at that boundary. A target must have been publicly observed previously and be hidden in the current frame. Exclude unseen cards, currently visible cards, and post-terminal padding. Public ages measure time since last visibility. Never use the deck identity array, future reveals, eventual matches, or the hidden environment state to supply missing targets. Save the reconstructed masks and their identities or a reproducible hash-bound reconstruction contract.

## Exactly one scalar per fixed memory

For saved raw probabilities `p`, fit inverse temperature `beta=1/T` in **[0.05, 20]**:

`q_beta(r) = exp(beta*log(p(r))) / sum_s exp(beta*log(p(s)))`.

Minimize categorical NLL with the same query/boundary/episode weighting as the original loss. Each eligible episode has equal total weight. Freeze all model tensors and recurrent update rules. Fit on all eligible public targets, not a selected age group, rank, family or successful-episode subset. Record every eligibility denominator; no eligible targets for a fit is a failed calibration, not an invented temperature.

The objective is convex in beta. Evaluate both endpoint derivatives and beta=1; if the optimum is strictly interior, perform exactly **64 derivative bisections** and one final midpoint evaluation. This is at most **68 objective/derivative passes per fit, 1,224 across all 18 fits**. Select identity if the final candidate's computed NLL is not strictly lower; exact ties use identity. These numerical rules are fixed before fitting. The helper returns the full scalar-evaluation trace, final beta and temperature, endpoint status, counts and weighting.

The raw values are saved float32 softmax outputs represented in float64. Power scaling is mathematically logit temperature scaling before finite rounding, but cannot recover discarded pre-softmax logits. Use float64 log-space arithmetic, never an epsilon floor. A calibration target with raw probability zero has infinite NLL at every finite beta and fails fitting. Zero probabilities at other ranks are supported. Beta=1 is an exact raw-value copy; the existing C picker then performs its unchanged normalization and public overrides.

## Three confidence conditions, one picker

After all 18 scalar receipts are complete and authenticated, evaluate every memory under:

1. **C baseline:** unchanged model probabilities and current C picker.
2. **Temperature:** the fit's single frozen beta, then the same C picker.
3. **Hard:** one-hot belief at the model's largest raw rank probability, smallest rank index on an exact tie, then the same C picker.

All conditions retain identical public unseen-uniform/currently-visible-one-hot overrides, legal actions, tie tolerance and unseen-first preference. The hard condition is a nonprobabilistic decision control, not a calibration method. Positive temperature preserves rank ordering; its intervention changes confidence and therefore expected-match scores. Neither condition updates weights or adds historical labels to the actor.

Use **64 entirely fresh paired decks**, disjoint from original train/development/evaluation, the prior A/B/C evaluation, and declared engineering inputs. No identities are selected here. Retain the exact public-table and last32 references with unchanged C **once each**, because neither receives a fitted temperature or an artificial hardening of its unknown-uniform entries. Coverage is **54 learned rows plus two references = 3,584 episodes**, at most **372,736 native actions**. Learned prediction and write caps are each **359,424**; references account for at most 13,312 actions. Restore each of 18 models once and initialize a fresh recurrent state per episode. Freeze balanced condition order before execution; later actions and observations may diverge.

## Generalization and native utility are different endpoints

Use each fit's **fresh baseline-C trajectory** as its fixed-prefix predictive test. From each stored raw prediction, compute all three confidence transforms against the same pre-action public targets. This needs no extra model calls or alternate trajectories. Fit-specific baseline histories can differ across memories, so this isolates transforms within a memory, not identical cross-family state distributions.

Report episode-balanced NLL, multiclass Brier and accuracy on all eligible targets, with age>32 descriptive strata and explicit denominators. Macro-average all 18 fit metrics equally, including all three pairs per family. No test label changes temperature, belief state, actions or model selection. Own-trajectory prediction scores may also be descriptive, but do not substitute them for this fixed-prefix comparison.

Hard beliefs incur **infinite proper NLL on every wrong queried rank**. Report an explicit infinite-NLL flag and error count, plus Brier and accuracy; do not smooth or average away those mistakes. Finite-beta NLL is computed in log space even if an exposed float64 probability underflows to zero. Report such underflow separately from a true raw-zero likelihood. A temperature's improved proper loss is evidence for this predictive distribution, not proof that it helps the native controller.

The proposed calibration-control continuation rule is clear if frozen as all of: temperature-minus-C native return at least **0.03** averaged across all 18 fits; positive mean difference in each pair index after averaging its six families; no family mean loses more than **0.01**; fixed-prefix macro NLL improves by at least **5%**; and identically weighted Brier does not worsen. For the relative NLL condition, require a finite positive baseline and compare `L_C-L_T >= 0.05*L_C` directly. Missing required metrics fail, with no favorable-stratum substitution. The three pair-index averages are not a requirement that all 18 fits individually improve, so show all individual differences too. Hard results remain descriptive and cannot be adopted as a winning controller by this rule.

## Implementation hooks and costs

The additive pure helper is `card_probability_calibration.py`: `fit_temperature(episodes)`, `transform_probabilities(raw, mode, beta)`, and `score_episodes(episodes, mode, beta)`. It uses only NumPy/stdlib, with no I/O, environment, model or random generator. Inputs are actual-length arrays for raw probabilities, targets, masks and ages. Its 45 synthetic tests cover an analytically known convex optimum, endpoint optima, flat identity ties, hierarchical weighting, exact beta=1 identity, hard infinite loss, stable temperature underflow, zeros, malformed masks and defensive copies. The caller still owns authenticity, split membership, public-label reconstruction, completion receipts and budgets.

Reuse the previous evaluator's fit authentication/restoration, public event loop, one prediction and one actual-reveal write per step, isolated layout digest, paired case schedule and failed-prefix preservation. Add confidence transformation between the one model prediction and the one C decision. Save underlying raw probabilities and final picker probabilities; the frozen beta/transform recreates intermediate beliefs without another stored probability cube. Reuse the original zero-new-training boundary, but count all scalar-fit passes, calibration input reads/decompression, proper-score arithmetic, restore/setup, transformation, native stepping, serialization, final hashing and complete wall time. Calibration uses **zero** new model calls; it is not zero computation.

Set explicit calibration, evaluation and whole-output caps after source review and ordinary engineering sizing, before scalar fitting or fresh evaluation. The helper's pass/action caps do not establish a wall-time guarantee. Preserve failed fits, cap failures and all attempted output; no automatic retry or budget extension. Success supports a bounded confidence/controller result. An architectural claim still requires useful learned memory relative to last32 and a separate transfer result, not relabeling scalar calibration as recurrent-model novelty.
