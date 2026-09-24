# Draft: tie blind dynamics to observed decision moments

**Unregistered prospective mechanism, not an admitted experiment or a novelty claim.** Written while the conditional-label study is running, without reading its live records. That study first tests averaged labels in the existing GRU. This proposal changes recurrent structure and would require a separate source, data, compute and acceptance protocol.

The [current model](../src/openjev/research/otto_action_latent_model.py) assimilates all nine observed public rows into a 28-dimensional GRU state, then applies an action-conditioned GRU through the observation gap. The input preserves the observed history. Whether the learned state loses relevant information remains unmeasured. Its blind transition has no enforced relationship to probability-weighted observation updates.

## One mechanism: a shared observation operator

Test a small learned positive operator recurrence. Its state `u` is a nonnegative vector of surviving latent mass, not four current action scores. Initialize `u0 = softmax(P * encoder(prefix))`. For each action `a`, learn matrices `B[a,o]` for the four nonterminal odor categories and a found-probability row. A column-wise softmax over all next-coordinate/odor entries plus found makes the total outgoing mass exactly one in real arithmetic. These are learned operators, not the known native sensor law.

For an observed nonterminal odor, assimilate the observation by conditioning:

```text
v = B[a,o] * u
u_observed = v / sum(v).
```

For an unobserved step, propagate the nonterminal mass without normalization:

```text
u_blind_next = sum_o B[a,o] * u_blind
survival = sum(u_blind_next)
cost_contrasts = C * W * u_blind_next,
C = I - 11^T / 4.
```

Observed found sets `u = 0` and every subsequent cost to zero. Missing mass during a blind rollout is the cumulative probability of found. Never divide the blind cost by survival. Use a linear readout without an unweighted bias; an intercept, if required, must multiply surviving mass. The readout can have signed coefficients. It therefore commutes with branch summation and avoids evaluating a nonlinear cost head at a mean latent state. Declare numerical treatment of zero observed evidence and underflow before any data execution; silent renormalization or probability clipping would change the proposed model.

This identity is exact **inside the learned model**, not proof that its state accurately represents the environment. Four cost contrasts alone need not be closed under observation updates. Moreover, the legacy teacher depends nonlinearly on its history-dependent belief filter. A finite mixture over physical source locations alone does not automatically make teacher cost linear: an exact representation may need the filter state as well. This finite operator model is an approximation whose external moment error must be measured.

## What the literature already establishes

[Littman, Sutton and Singh, Predictive Representations of State](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf) represent state through action-conditional future tests and derive normalized action-observation updates. Their finite-POMDP representation result does not establish that a chosen small basis is sufficient for our teacher. General linear PSRs can use signed operators; the positive construction above is a more restrictive learned finite-state model, not an implementation of that full representational result.

[Boots, Siddiqi and Gordon, Closing the Learning-Planning Loop with Predictive State Representations](https://www.cs.cmu.edu/~ggordon/boots-siddiqi-gordon-closing-loop-psrs.pdf) learn transformed predictive operators and use them for planning. Their consistency argument needs finite rank and sufficiently informative features. They also discuss invalid finite-sample probability estimates. Operator learning and recurrent predictive state are established techniques.

[Grimm et al., The Value Equivalence Principle](https://arxiv.org/abs/2011.03506) define equivalence through matching Bellman updates for specified policy and function classes. Our finite-horizon teacher-cost moments are a narrower target. Low teacher-cost MSE or an internal branch identity would not establish value equivalence, optimal control or novelty.

## Necessary comparison and discriminating test

Use three arms: the GRU with averaged labels; the tied operator model; and the same operator model with a separately learned blind transition `A[a]` instead of `sum_o B[a,o]`. The untied model keeps the observed operators, linear readout, survival treatment and all supervision. This is the necessary ablation: improvements that survive untying cannot be attributed specifically to marginalization structure.

Endpoint-only labels cannot identify the individual observation operators. A future training set must supply matched observed-branch and multihorizon moment supervision to every arm, including the GRU. Any likelihood or filter-derived targets must be available to all controls and charged equally. At inference, none receives a known filter, hidden source, future odor or extra teacher call. Preserving history in 31 input features does not give the model the known likelihood computation for free.

Keep latent width equal between tied and untied models. The extra untied matrix makes that control larger; disclose this conservative asymmetry rather than calling parameter counts identical. Before registration, qualify widths and fixed training schedules on fabricated data for comparable effective parameters and measured training/inference budgets across all three arms. Preserve an equal-update comparison and predeclare any separate compute-matched comparison. If actual resource bands fail, retain the results but withhold a compute-matched architecture claim.

Train short-gap and observed-update moments, then test fresh longer committed gaps and a sensing-law shift. Predeclare the horizon split, seeds, survival weighting and regret criterion. Report teacher-based external moment error and survival error alongside unconditional decision regret; the model's internal consistency residual alone is a tautology. Architectural evidence would require the tied model to improve long-gap decisions over both controls at comparable cost, survive the shift, and retain observed-update performance. Failure would reject this particular finite moment closure, not demonstrate that recurrence or predictive-state representations cannot work.
