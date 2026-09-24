# Full-belief targets for compact recurrent prediction

Prospective OpenJev development pilot, `otto-belief-distillation-v1`. Freeze this
protocol, sources, inputs, qualification and seed allocation before collection.
Previous TEST and confirmation panels stay closed. No Astra calls, model search,
replacement cases, retries, early stopping or selected checkpoints. Failure is a
terminal outcome for this registration, including a technical or budget failure.

## Question

Does training on the full conditional outcome distribution improve a compact
recurrent predictor relative to matched sampled-label training? Does action
recurrence add value beyond proposed-action-blind and direct horizon controls?
The preceding action-latent pilot failed its gate, with only124 retained TRAIN
cases. This new study increases the fresh allocation for all arms and changes
the probability supervision. Cross-study differences cannot isolate either
change. Only the matched comparisons within this study support attribution.

This is privileged TRAIN distillation using a known simulator likelihood. It is
not Bayesian weight inference, RL, autonomous planning, connectome evidence or a
new model architecture. The existing28-state model source remains unchanged.

## Cases and public trajectory

Use the authenticated original OTTO sampled-source simulator and TensorFlow
teacher,53x53 grid, four odor categories, R_dt2 and Euclidean Poisson sensing.
Initial odor cycles1,2,3. Original public filter and analytic behavior remain
unchanged. Collect TRAIN1536 starting cases at lambda3, DEV128 at lambda3 and
DEV128 at lambda4. Exact seeds are324000001..324001536,
325000001..325000128 and326000001..326000128. Fit seeds327000001..327000003.
Metadata-only seed-review02 covers all1795 exact seeds; it is a scoped historical
scan, not a proof of global seed uniqueness. Earlier review01 remains preserved.

Each case has eight observed analytic-policy transitions and nine prefix feature
rows. Exclude and log cases found during the prefix, without replacement.
Require at least800 retained TRAIN and64 DEV in each regime before fitting.
One originating case contributes one block to one split. Precommit actions from
PCG64(SeedSequence([seed,911])) before the block, uniformly over four directions.
TRAIN horizon4, DEV horizon8. Center-origin paths are geometrically in bounds.

Outcomes are odor0..3 or absorbing-found4. Found is native hit=-2. After found,
retain found labels and zero feature/cost placeholders, without simulator or
teacher calls. Annotate surviving states with original four-action teacher
costs and public features. Annotation and Bayesian computation costs are charged.

## Probabilistic teacher

A separate shadow posterior starts from the actual reset source-draw law of the
initial public prior. It does not reuse the legacy posterior after the prefix.
That filter has a low-evidence normalization floor and a vectorized likelihood;
we log differences while preserving its original policy and teacher inputs.

Reconstruct the sampler's scalar likelihood at every integer displacement in
[-52,52]^2, separately for lambda3 and lambda4. Use the authenticated original
scalar mean-hit and Poisson functions, Euclidean scalar distance, categories0..2
and max(0,1-sum) tail3. Co-location is found and has no odor draw. Persist both
raw and effective105x105x4 lookup tables. Validate actual native draw vectors
against the primitive laws as an audit only. Hidden source coordinates cannot
enter shadow updates, neural inputs, actions or predictive targets.

The sampler normalizes a cumulative sum and uses right-sided search. Installed
NumPy source defines uint64-to-double as (rnd>>11)/2^53. Under the declared
uniform-grid model, effective category masses are differences of
ceil(2^53*normalized_CDF)/2^53. Record normalization and quantization changes,
including positive raw masses rounded to zero. This models the categorical law;
it does not claim fixed PRNG histories are independent physical randomness.
Only reset prior mass gets an explicit bounded roundoff normalization (1e-10
admission tolerance). Every later positive evidence is divided exactly in
floating arithmetic; no smoothing, evidence floor or arbitrary renormalization.
Impossible observations or underflow fail visibly.

Before each normal action, marginalize its odor/found distribution from the
shadow belief, then assimilate that observation for the next prediction. Once
found was observed earlier, normal forecasts are exactly one-hot found.
Blind marginals start at the common observed prefix, integrate unobserved odors,
and accumulate source mass of unique visited positions into absorbing-found.
Repeated positions cannot double-count found mass. No realized future outcome,
feature or found status enters the blind teacher or neural rollout.

Save initial_belief, eight prefix actions/outcomes and prefix position so an
independent audit can reconstruct both posteriors and targets. These arrays,
full beliefs and probability labels are never neural inference inputs.

## Matched models and training

Four arms, three seeds each,12fits:

* recurrent_soft: unchanged action-conditioned GRU transition F and GRU
  assimilation G,28-state,8,299 parameters, trained on probability targets.
* recurrent_sampled: identical initialization, inputs, network and optimizer;
  categorical training targets are the sampled outcomes instead.
* action_blind_soft: same8,299 parameters but zero proposed-action inputs to F;
  its336 action-input weights are inactive. It still sees past observed actions.
* direct_soft: same G/heads and8,107 parameters, with causal positional action
  pooling and a direct horizon decoder instead of learned action recurrence.

All arms see the same31-feature prefix. Gap inference receives only prefix,
lengths and proposed actions. Normal inference predicts before assimilating each
new feature row. Sampled and soft recurrent fits share seed initialization and
epoch permutations. Rotate arm execution order by fit-seed index. Every fit uses
80epochs, Adam0.003, batch32, clipping5, no schedule. Complete every fit before
first DEV array decoding. Fix the final epoch checkpoint without selection.

Average blind and normal objectives equally. Outcome loss is categorical cross
entropy, using either the hard outcome or full distribution. Average over all
cases and horizons. In normal mode only rows AFTER already-observed found have
zero categorical loss, retaining the same denominator; the first found event
is still scored. Gap suffixes remain predicted and scored. Cost and auxiliary
losses are unchanged across all arms: equal-case mean over surviving target
rows, with zero contribution from unsupported cases. Center all four teacher
costs and divide by64. Cost_scale is float32 sqrt(max(TRAIN mean squared centered
cost,1e-6)); the zero-initialized cost head emits in those units, and its MSE is
divided by scale squared. Auxiliary MSE weight0.1 uses future columns19:21.

At inference, take softmax of float64-converted logits. No probability clipping
or smoothing. For all arms, normal forecasts AFTER previously observed found are
analytically replaced by exact one-hot found. Unresolved prediction probabilities
must remain positive. Zero predicted probability on positive oracle support is
an explicit infinite-loss failure. This common terminal shortcut differs from
the previous study and prevents attributing its effect to soft-target training.

## Gate and supplementary diagnostics

Keep the strict hard-outcome continuation criterion. For each fit seed, regime
and each of three controls, recurrent_soft must have at least1% lower
case-weighted sampled log loss and5% lower teacher-action cost gap at horizons5..8.
Normal all-horizon losses/gaps cannot worsen more than1%. All18cells must pass,
with at least48 originating cases supporting decision targets in each compared
group. Undefined/zero comparator gaps cannot prove strict relative gain.

Select the lowest-index legal predicted minimum cost. Average surviving decision
rows within case then supported cases. Report unavailable cases and the full-case
zero-contribution alternative. Outcome scores include all suffixes. Report
horizons1..8, short1..4, long5..8, Brier, sample sizes and compute. Fit seeds share
DEV cases and are not independent experimental samples.

Supplementary oracle-relative KL, expected log score, entropy and Brier excess
use exact marginal targets, with all-row and unresolved-row denominators.
Zero oracle masses contribute zero. Count KL roundoff corrections within1e-12;
larger negativity fails. These diagnostics cannot change the gate.

For every DEV prefix also predict an unobserved opposite-action block pi(a)=a
XOR1, using both oracle and each model. No actual alternate outcomes or normal
counterfactual observations are invented. Report oracle JS/TV, squared signal
S=sum((p(A)-p(piA))^2), and model effect error
E=sum(((q(A)-q(piA))-(p(A)-p(piA)))^2). Retain every case, including S=0; no ratio,
filter or effect-based selection. Charge all12additional rollout views. The
proposed-action-blind control should give E=S up to floating summation tolerance.

## Bounds, admission and audit

Each original supervised process has1,800s suspend-inclusive deadline,4GiB RSS
and512MiB output. Single numerical CPU threads. Native maximums are1792resets,
22528steps,8192teacher/value calls,24320feature/analytic calls. Additional table,
lookup and Bayes call limits are fixed in collector source. No neural calls in
collection. All costs include checks and serialization; timings are specific to
this implementation and machine, not general speed claims.

Pre-execution qualification uses fabricated data only: CDF boundary enumeration,
Bayes tree identities, recurrence/input causality, terminal semantics, loss
weights, independent scalar metrics and capacity projection. Native preflight
reads runtime and source metadata without importing numerical frameworks or
stepping the environment. Pin installed NumPy grid-conversion sources too.

Audit original collection/fit closure before loading saved DEV/prediction/sensor
arrays. Independently reconstruct effective sensor tables, prefix posterior,
blind/normal/opposite marginals, hard and expected scores, action effects and the
18-cell gate. Verify12fit checkpoints and960matched epoch permutations from
saved artifacts. No neural, native, teacher, optimizer or solver calls during
audit. Publication requires separate validation of the audit's own original
terminal receipt, process reaping and absent process group. Archive sources,
checkpoints, raw saved evidence and failed engineering attempts.
