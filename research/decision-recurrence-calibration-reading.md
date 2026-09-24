# Recurrent decisions and calibration: useful prior work

Reading note, 24 September 2026. These are possible follow-ups, not changes to
the completed action-error experiment or claims of a new method.

| Primary paper | Mechanism worth testing | Important boundary |
|---|---|---|
| [Learning Probabilistic Filters with Strictly Proper Scoring Rules](https://arxiv.org/abs/2606.26497), Bach et al., 2026, Sections 3-4 | Learn an ensemble update using a proper distributional score; cross architecture and loss as separate factors. | Its population result assumes realizability. It uses simulated state-observation trajectories and a transformer ensemble map. Replacing that map with recurrence needs new evidence. |
| [Conformal Decision Theory](https://arxiv.org/abs/2310.05921), Lekeufack et al., 2024 revision, Section IV | Adjust a controller's conservatism from observed decision losses. | The online result controls average loss and requires an eventually safe family of decisions. It does not automatically certify each action or give a safe fallback to every task. |
| [Conformal Risk-Averse Decision Making with Action Conditional Guarantee](https://arxiv.org/abs/2606.05551), Zhu et al., 2026, Sections 3-4 and proofs | Measure risk conditional on the action actually chosen, instead of reporting only pooled coverage. | Its exchangeability-based guarantees cannot simply be transferred to dependent, policy-selected gameplay observations. Rare actions also need adequate support. |
| [When Should a World Model Move? Loss-Conditioned State Execution](https://arxiv.org/abs/2609.15801), Xu et al., 2026, Sections 3-4 | Compare a fixed proposal's decision loss with retaining the current state; use independent calibration units for a gate. | Certification concerns fixed proposals/groups, bounded losses and independent target-population units. It does not establish compute savings; those require a separate cost measurement. |

Our interpretation: separate three questions. Does the recurrent state retain
the relevant uncertainty? Does its decision minimize the application's actual
cost? Does a calibrated gate improve the resulting cost-compute tradeoff?
Combining these paper names is not a contribution. Each added mechanism needs
an ablation against the same backbone and a simple control with comparable cost.

A concrete later study could test an ensemble of recurrent states on a task
with genuinely ambiguous, multimodal posterior beliefs. Our current model
already carries an eight-coordinate probability-mass latent; its coordinates
are not established as true-state probabilities. The proposed ensemble must
represent a clearly specified additional uncertainty, rather than merely being
called distributional. Compare mean-target MSE with an energy score and cross
that choice with the state architecture. MSE is already proper for a conditional
mean; the question is whether learning more of the distribution helps decisions.
Keep observation history, latent-state supervision, teacher information,
training budget and candidate actions matched. Simulated latent-state labels
used by the filter paper are not current learner inputs. Charge ensemble members
and any extra inference steps.

Evaluate blind forecasting, filtering after new observations, and decision
regret separately. Use an untouched change in observation noise and a second
transition family. A filter that knows the true noise level is an informative
reference, but a deployable model must receive that information through declared
inputs or infer it from allowed observations.
In robotics, use independent episodes as calibration units where the selected
theorem requires them, and state how loss feedback and a safe fallback are
obtained. An Astra-generated label is a teacher prediction, not ground truth
for an empirical safety or calibration claim.

Connectome wiring would be one controlled structural factor. It needs matched
dense and degree-preserving randomized-wiring controls, together with measured
parameter, update and inference costs. The existing negative biological-wiring
results remain part of the evidence. This reading note does not reopen them.

The completed [loss comparison](finite-action-range-results.md) failed its rule
at 17/54 conditions. Twice-MSE had lower mean regret than the proposed range
loss in all four regime/horizon settings. All three rounded loss variants failed
the short-horizon and observed-filtering criteria in the same four of five
cohorts under increased observation noise.

That result makes observation reliability a more direct next hypothesis than
adding a connectome or ensemble immediately. A separate study could compare the
current fixed observation model with global softening fitted on TRAIN, a model
that infers reliability from its observation history, a simpler window-based
estimator, and an oracle supplied the true noise level. Update reliability only
when a new observation arrives, and freeze it during blind forecasts. Training
must include the information needed to identify
reliability; merely adding a recurrent state cannot supply it. Test untouched
noise levels and within-episode changes, measure filtering and decision costs,
and charge adaptation work. If a history-dependent model cannot improve over
global softening, adaptation has not earned its cost. If it improves KL alone,
that is not evidence of better decisions. The oracle diagnoses the potential cost of noise
mismatch; it is not a deployment baseline with equal information. This is a
prospective hypothesis, not a demonstrated cause or an admitted new run.

## Completed reliability-memory follow-up

The separately registered [reliability-memory study](reliability-memory-results.md)
now tests part of that hypothesis with public-event likelihood training. It
fails its rule at 8/13 conditions. The recurrent model has 1.83-3.59% lower regret
than the two prespecified bank controls, consistently across all five datasets, but falls
short of the required 10%. Normal-noise regret is 3.80x the unchanged model's.
The one-parameter global adapter beats recurrent on every higher-noise and
changing-noise dataset. These results support a modest descriptive difference
between gates, not continuation or a new architecture claim. The static bank,
with no added trainable parameters, has lower mean regret in all four settings.

Any follow-up needs to explain that tradeoff before adding biological wiring.
A useful next diagnostic would separate imperfect backbone learning from
uncertainty about the observation process, using a supplied-physics reference
and an exact public-history mixture over the specified noise schedules.
Compare both to the already disclosed private-noise-path reference. That could
estimate how much improvement public evidence permits, without assuming that
the privileged reference is attainable. It would require a new protocol and
fresh evaluation episodes; it must not tune this failed recipe on its DEV set.

Exactness would require an explicit, correctly supported schedule prior and
correct physical, emission and cost laws. Supply the same declared schedule
prior to the true-law and frozen-backbone references, without revealing an
episode's stratum, realized noise path or switch time. A prior learned only on
0.12/0.48 noise does not cover the 0.30 evaluation regime; adding that prior
knowledge is a diagnostic assumption, not evidence of robustness to unknown
distribution shift. Use a fixed-boundary or additive causal loss for a strict
attainable-floor claim: the current per-episode late-boundary average has a
denominator that depends on future survival. Keep that earlier metric
descriptive. If the correctly specified public filter offers little headroom,
stop this adaptation line. If only the true-law reference helps, investigate
backbone mismatch before another learned gate.

## Completed exact-reference diagnostic

The [separately registered diagnostic](schedule-headroom-results.md) now closes
that proposal. On five fresh datasets, exact schedule tracking with learned
fields reduces mean additive regret by 40.95% versus recurrent, 21.86% versus
global, and 12.18% versus learned static-two, with all five cohorts improving
in every one of seven control comparisons. The true-law exact filter has
40.57% lower mean regret than learned exact; this combines T/O/h/C mismatch,
not just transition error. True exact also improves 17.54% over true static-two
and passes both registered history conditions.

Both candidates nevertheless pass only 14/15 conditions. Normal-noise regret
is 2.55x unchanged for learned exact and 1.89x for true exact. The ordered
classification is therefore `NO_REGISTERED_HEADROOM`: no continuation under
the specified preservation requirement. It does not mean that exact history
tracking has zero benefit, prove that the tradeoff is unavoidable for every
controller, or make the privileged reference attainable. The exact references
also receive a correct episode prior and retain 288 joint probabilities;
learned exact inference takes 4.28x as long as the GRU. No new model was trained.

Stop this observation-reliability adaptation line for this population, history
length and requirements. Further architecture work needs a separately justified
task and evaluation, not another gate tuned against these evaluation episodes.
The earlier 8/13 and 17/54 failures remain unchanged.
