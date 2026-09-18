# Bayesian calibrated decisions: implementation and research direction

This update runs actual Doom experiments with a learned Bayesian outcome model. It does not reproduce TypeSafe's proprietary RLCD, train a language model with RL, or establish an ICLR contribution. The original demonstration still uses its existing models; research policies are evaluated through a separate runner.

## RLCD naming and usable public implementations

TypeSafe calls its method **Reinforcement Learning for Calibrated Decisions**. Its [launch article](https://typesafe.ai/blog/introducing-system-one-models-and-jev) describes typed probabilities and calibrated decisions, but the checked material does not supply a training loss or an implementation sufficient to reproduce that method. Do not infer its objective, Bayesian architecture, or internal reward from its name.

The public `harshatheg/Qwen-2.5-1B-RLCD` repository provides parallel constrained
inference around existing Qwen2.5-1.5B weights. Our [pinned source review](../research/qwen-parallel-source-review.md)
found no new weights or training pipeline in the inspected release. Its MLX
collision fallback floors confidence at 0.75, so that output should not be
treated as calibrated probabilities. Shared-prefix batching remains a useful
engineering idea to evaluate separately.

[Yang et al.'s RLCD](https://arxiv.org/abs/2307.12950) means **Reinforcement Learning from Contrastive Distillation**, a different approach using positively and negatively prompted outputs to create preference pairs. It is not evidence for Jev's training recipe.

The closest usable calibration-reward baseline found in this review is [RLCR's official implementation](https://github.com/damanimehul/RLCR), associated with [Beyond Binary Rewards](https://arxiv.org/abs/2507.16806). It combines correctness and calibration rewards for language-model training. Its published training setup uses multiple A100 GPUs. We reviewed its documented approach but did not import its code or launch that training job.

Other important comparisons:

- [Calibrated Model-Based Deep RL](https://proceedings.mlr.press/v97/malik19a.html) already studies calibration in learned dynamics and control.
- [Laplace Redux](https://arxiv.org/abs/2106.14806) describes established approximate Bayesian methods. A Laplace posterior is not a new contribution here.
- [Thompson sampling](https://arxiv.org/abs/1707.02038) is established posterior-based exploration. The package exposes a sampling primitive, but this pilot does not evaluate online Thompson learning.
- [Calibrate-Then-Act](https://arxiv.org/abs/2602.16699) already addresses cost/uncertainty tradeoffs in agent exploration using inferred environment priors.
- [Verifiable Rewards for Calibrated Probabilistic Forecasting](https://arxiv.org/abs/2607.00164) studies outcome noise and calibrated forecasting rewards. Adding a Brier reward alone is not a new research claim.

## What is implemented

`openjev.research.bayesian.BayesianLogistic` fits a Gaussian-prior logistic outcome model using Newton optimization and a Laplace posterior. Twenty-node Gauss-Hermite quadrature approximates the posterior predictive mean and separates uncertainty into expected Bernoulli variance and variance of predicted probability across posterior samples. It also returns logit-space posterior credible bounds and supports posterior sampling.

Those bounds are conditional on the approximate model, prior, and likelihood. Correlated episode observations, omitted state, a misspecified linear predictor, and distribution shift can make them overconfident. They are not conformal intervals or safety guarantees.

A standard `brier_reward(p, outcome) = -(p-outcome)^2` primitive is included for future calibration-reward experiments. **It is not the training loss in this pilot.** The pilot fits a Bernoulli likelihood and performs one-step model-based control using its forecasts. It does not run PPO, GRPO, Bellman backups, or proprietary RLCD.

Keep two meanings of probability separate:

- `choice_probabilities`: a distribution over mutually exclusive action choices, which sums to one.
- `success_probability(action, horizon)`: the probability of a specified outcome if an action is issued. These probabilities across different actions need not sum to one.

The current candidate API returns the former. The Bayesian research module returns the latter and never silently substitutes one for the other.

## Frozen development experiment

Each of five models is trained on a separate set of 16 real `defend_the_center` episodes. Collection randomizes firing and occasionally steering to gather both hits and misses. The predictor has seven numeric features, including visibility, aiming error and target width; it does not encode English or pixels. Features use only the observation available before the action window.

Five firing policies are evaluated with identical rule steering and seven-tic action windows:

1. Existing firing rule.
2. Existing tiny imitation model's firing choice.
3. Learned logistic MAP probability above the fixed cost threshold.
4. Bayesian posterior mean above the same threshold.
5. Bayesian lower credible bound above the same threshold.

The two existing controllers share a single evaluation per episode seed. Learned controllers use five training replicates and ten paired evaluation episodes in each of two scenarios. Training and evaluation episode IDs are disjoint. There is no online refitting on evaluation episodes, threshold sweep, or choice of hyperparameters from these results.

A shared randomized-behavior audit supplies identical firing-window labels to MAP and Bayesian mean for calibration comparisons. Per-policy selected-event calibration is logged separately because different action choices generate different observed datasets. The audit is still limited to its behavior policy and does not reveal every counterfactual action outcome.

### Measurement failure retained in v1

`bayesian-doom-v1` ran 440 episodes, but net ammo change was not a valid shot detector in `defend_the_line`: ammo replenishment masked firing. It also exposed a missing check for empty audit datasets, resulting in NaN audit values. The raw run, original source snapshot and a quality-review receipt are retained. Its cross-scenario utility/calibration results are invalid. Its center-only development result did not establish a Bayesian gain.

The problem is related to the [reported ViZDoom ammo behavior](https://github.com/Farama-Foundation/ViZDoom/issues/709). The local diagnostic confirmed increasing ammo during firing; it did not rely solely on the external issue report.

`bayesian-doom-v2` explicitly changes the estimand to **hit-count increase during an issued seven-tic fire-command window**, charging 0.25 per issued fire-command window. Waiting does not fabricate an outcome label. Cooldown windows with no hit are failures for this particular target. This measures command-window success, not per-bullet accuracy. Raw net ammo change is retained only as a diagnostic and is not a valid shot count in the line scenario.

V2 uses fresh evaluation seeds 40000-40009, unchanged learner hyperparameters and the same training episodes. It is a new development study, not a repaired identical replication or sealed confirmation. Its runner fails if an audit set is empty and rejects nonfinite result JSON.

## Reproduce

```sh
uv sync --frozen --extra dev
uv run python research/bayesian_doom.py --output runs/bayesian-doom-new
uv run python research/analyze_bayesian_doom.py runs/bayesian-doom-new \
  --output runs/bayesian-doom-new-analysis.json
```

The output directory must not already exist. The runner records source, protocol and artifact hashes, posterior weights, episode summaries, and decision-level traces. It limits a run to 600 episodes and 30 wall-clock minutes; the planned count is 440. No paid model API is used. Research source snapshots are available in the git commits recorded in the result report. A frozen replay should check out the matching source version.

## Measured development results

The valid v2 run completed **440 real ViZDoom episodes in 85.18 seconds** on this workstation: 80 training-collection episodes, 20 common-audit episodes, and 340 policy-evaluation episodes. Five independently collected training sets produced five fitted models. Each learned policy has 50 evaluation episodes per scenario, while each baseline has ten; baseline episodes are shared across comparisons, not counted as independent copies.

| Firing controller | Center utility | Shifted line utility | Shifted line kills |
| --- | ---: | ---: | ---: |
| Rule baseline | 3.175 | 13.900 | 26.90 |
| Tiny imitation baseline | 3.150 | 13.100 | 25.40 |
| Logistic MAP | 0.855 | 5.715 | 16.34 |
| Bayesian mean | 0.870 | 5.140 | 16.96 |
| Bayesian lower bound | 1.000 | 0.575 | 4.76 |

Utility is the v2 command-window objective defined above, not the environment's default reward or a score comparable to the original GIF benchmark. All policies here share rule steering and seven-tic steps.

**No Bayesian mean control gain was established.** Its mean paired utility difference from MAP was +0.015 in the center scenario (exploratory 95% interval -0.705 to +0.725) and -0.575 in the line scenario (-3.745 to +1.960). These intervals do not establish equivalence or rule out smaller benefits.

**Conservative gating performed worse under the scenario shift.** The Bayesian lower bound's line-scenario utility difference from MAP was -5.140, with an exploratory interval of -9.120 to -1.720. This is a local failure of this particular model, threshold, credible bound and objective, not a general theorem against Bayesian control.

**Prediction quality and control quality differed.** On the identical line-scenario audit events, mean Brier score across models decreased from 0.230733 for MAP to 0.226479 for Bayesian averaging (lower is better). That small descriptive improvement did not translate into better mean utility. The center audit was essentially unchanged: 0.122786 versus 0.122836. There were 349 center audit events and 700 line audit events. These are dependent events from ten episodes per scenario, not 1,049 independent trials.

![Paired utility differences against MAP in the two scenarios](../evidence/bayesian-doom-v2/utility-comparison.png)

Intervals use 10,000 paired crossed-bootstrap draws over training replicate and evaluation seed. They are exploratory and unadjusted for multiple comparisons, with only five training sets and ten evaluation seeds. The linear model omits weapon cooldown and temporal state; posterior intervals use an approximate conditional-independence likelihood despite within-episode dependence. The inference timing uses a shared belief routine even for MAP, so it is not an optimized MAP-versus-Bayes speed comparison. No method is promoted on these results.

Artifacts: [analysis](../evidence/bayesian-doom-v2/analysis-001.json), [run manifest](../evidence/bayesian-doom-v2/run-001/manifest.json), [episode results](../evidence/bayesian-doom-v2/run-001/results.json), [lossless trace archive](../evidence/bayesian-doom-v2/run-001/trace.jsonl.gz), and [the v1 quality exception](../evidence/bayesian-doom-v1/run-001/quality-review.json). The v1 and v2 execution-source snapshots were committed as `4007e64` and `f59ca5c`. Artifact hashes refer to original trace bytes; decompressing the archive reproduces those bytes. Original v1 JSON containing NaN is preserved as invalid evidence, not repaired into successful observations.

All five v2 MAP fits were numerically converged at the recorded solution (largest gradient norm below 8e-8), and all posterior covariance matrices had positive minimum eigenvalues. These checks validate optimization mechanics, not model specification or scientific novelty.

## A defensible next research hypothesis

**Separate honest outcome forecasting from action selection, then allocate additional observations or recurrent computation based on expected decision improvement rather than confidence alone.**

For a fixed action and outcome distribution, the expected Brier score is optimized by the true event probability. Once action selection changes the observations received, that fact alone does not establish calibration or good control. Rewarding self-reported confidence can also change what the controller chooses to report. Keep the forecast head's proper-score learning signal separate from the utility decision rule, and test the complete loop.

A follow-up could combine a frozen text/candidate encoder, a Bayesian outcome head, and optional recurrent state or world-model rollouts. The first controls must include the same outcome head without Bayes, the same model with fixed extra compute, scalar probability and categorical-entropy gates, and matched-data/compute ensembles. Random audit actions should have known propensities and a fixed budget; selected-feedback correction needs support/overlap checks. Approximate posterior uncertainty must be compared with actual error under shift.

A potential contribution would be evidence or theory about calibration under changing candidate sets, computation policies, and selective outcome feedback. The components above are established. A new method claim would require showing why the complete procedure differs from the closest work and survives the matched ablations, plus a second environment and an English decision task. The pilot below is a way to reject weak ideas cheaply, not proof that such a contribution has been achieved.
